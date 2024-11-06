import sys
import os
from functools import partial

import h3
import pandas as pd
import numpy as np
import geopy.distance

import haven.spark as db 

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf, pandas_udf, explode,
    row_number, dense_rank,
    lead, log, exp,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
    ArrayType,
    FloatType,
)
from pyspark.sql.window import Window

# GLOBAL FUNCTIONS -----------------------------------------

def apply_to_create_columns(dataframe, func, input_cols, new_fields):
    """
    Inputs:
    - dataframe: DataFrame
    - func: function
    - input_cols: list of str, input columns
    - new_fields: list of StructFields

    Returns a new DataFrame with the new fields created by applying
    the function to the input columns.
    """
    args = [dataframe[col] for col in input_cols]
    if len(new_fields) > 1:
        func = udf(func, StructType(new_fields))
        dataframe = dataframe.withColumn("_tmp_struct", func(*args))
        for field in new_fields:
            dataframe = dataframe.withColumn(
                field.name, dataframe["_tmp_struct"][field.name]
            )
        dataframe = dataframe.drop("_tmp_struct")
    else:
        func = udf(func, new_fields[0].dataType)
        dataframe = dataframe.withColumn(new_fields[0].name, func(*args))
    return dataframe

def vectorized_create_column(dataframe, pandas_udf, input_cols, field):
    """
    Inputs:
    - dataframe: DataFrame
    - pandas_udf: function, pandas UDF
    - input_cols: list of str, input columns
    - field: str, new field name

    Returns a new DataFrame with the new field created by
    applying the pandas UDF to the input columns.
    """
    return dataframe.withColumn(
        field, pandas_udf(*[dataframe[col] for col in input_cols])
    )

def join(dataframe, other, on, how="inner"):
    """
    Inputs:
    - dataframe: DataFrame
    - other: DataFrame
    - on: list of str, column names to join on
    - how: str, join type

    Returns a new DataFrame with the two DataFrames joined on the specified columns.
    Repartitions the DataFrames on the join columns before joining.
    """
    dataframe = dataframe.repartition(on)
    other = other.repartition(on)
    return dataframe.join(other, on=on, how=how)

# APPLY FUNCTIONS -----------------------------------------

def get_h3_index(lat, lon):
    return h3.geo_to_h3(lat, lon, 4)

def find_neighbors(max_km, h3_index):
    """
    Input:
    - h3_index (str): the H3 index

    Finds all the h3 indices whose centroids are 
    within `max_km`. 
    """
    h3_coords = h3.h3_to_geo(h3_index)
    checked = set([h3_index])
    neighbors = set([h3_index])
    distance = 1
    found_neighbors = True

    while found_neighbors:
        found_neighbors = False
        candidates = h3.k_ring(h3_index, distance)
        new_candidates = set(candidates) - checked
        for candidate in new_candidates:
            if geopy.distance.geodesic(h3_coords, h3.h3_to_geo(candidate)).km <= max_km:
                neighbors.add(candidate)
                found_neighbors = True
            checked.add(candidate)
        distance += 1
    return list(neighbors)

def add_distance(origin_lat, origin_lon, lat, lon):
    distance = geopy.distance.geodesic(
        (origin_lat, origin_lon), (lat, lon)
    ).km
    return distance

@pandas_udf(FloatType())
def add_water_heading(velocity_north, velocity_east):
    return np.arctan2(velocity_north, velocity_east)

@pandas_udf(FloatType())
def add_movement_heading(origin_lat, origin_lon, lat, lon):
    return np.arctan2(lat - origin_lat, lon - origin_lon)

# VECTORIZED UDFs -----------------------------------------

@pandas_udf(StringType())
def strftime(time):
    return time.dt.strftime('%Y-%m-%d')

@pandas_udf(TimestampType())
def get_time(epoch):
    return pd.to_datetime(epoch, unit='s')

# OTHER -----------------------------------------

def infer_map(model, features, iterator):
    for dataframe in iterator:
        dataframe["log_odds"] = model.predict(dataframe[features])
        yield dataframe

def infer(dataframe, model, features):
    schema = StructType(
        dataframe.schema.fields
        + [
            StructField(f"log_odds", FloatType())
        ]
    )
    dataframe = dataframe.mapInPandas(
        partial(infer_map,  model, features), schema=schema
    )
    return dataframe

# COMMON STEPS -----------------------------------------

def create_choices(dataframe, CONTEXT):
    choices = apply_to_create_columns(
        dataframe, partial(find_neighbors, CONTEXT["max_km"]),
        ['origin_h3_index'], [StructField("h3_index", ArrayType(StringType()))]
    )
    choices = choices.withColumn("h3_index", explode(choices.neighbors))
    choices = choices.withColumn("_choice", row_number().over(Window.partitionBy(["_individual", "_decision"]).orderBy("h3_index")))
    return choices

def join_environment(dataframe, physics, biochemistry, CONTEXT):
    environment = join(dataframe, physics, on='h3_index', how='inner')
    environment = join(environment, biochemistry, on='h3_index', how='inner')
    return environment

def derive_features(dataframe, CONTEXT):
    derived = apply_to_create_columns(
        dataframe, h3.h3_to_geo,
        ["h3_index"], [StructField("lat", FloatType()), StructField("lon", FloatType())],
    )
    derived = apply_to_create_columns(
        derived, h3.h3_to_geo,
        ["origin_h3_index"], [StructField("origin_lat", FloatType()), StructField("origin_lon", FloatType())],
    )
    derived = apply_to_create_columns(
        derived, add_distance, ["origin_lat", "origin_lon", "lat", "lon"],
        [StructField("distance", FloatType())]
    )
    derived = vectorized_create_column(
        derived, add_water_heading, ["velocity_north", "velocity_east"],
        "water_heading"
    )
    derived = vectorized_create_column(
        derived, add_movement_heading, ["origin_lat", "origin_lon", "lat", "lon"],
        "movement_heading"
    )
    return derived

# MAIN FUNCTIONS -----------------------------------------

def simulate(spark, model, CONTEXT):
    # Pull and Format Inputs
    # TODO Temporary for testing this should be an S3 bucket
    from datetime import datetime
    INPUT = pd.DataFrame([
        {'_quanta': 10.0, 'h3_index': '840c9ebffffffff', 'time': datetime(2020, 4, 17)},
        {'_quanta': 10.0, 'h3_index': '840c699ffffffff', 'time': datetime(2020, 4, 17)}
    ])
    distribution = spark.createDataFrame(INPUT).withColumnRenamed("h3_index", "origin_h3_index")

    # Add Individual and Decision Columns
    distribution = distribution.withColumn("_individual", dense_rank().over(Window.orderBy("h3_index").partitionBy()))
    distribution = distribution.withColumn("_decision", distribution["_individual"])

    # Create Choices
    choices = create_choices(distribution, CONTEXT)

    # Prepare for Join Against Environment Data
    choices = vectorized_create_column(choices, strftime, ["time"], "date")

    # Pull Environment Data
    date = choices.head(1)[0]["date"]
    physics = spark.read.parquet(f"s3a://haven-database/copernicus-physics/h3_resolution=4/region=chinook_study/date={date}/")
    physics = physics.filter(physics.depth_bin == 25.0).select("h3_index", "mixed_layer_thickness", "velocity_east", "velocity_north")
    biochemistry = spark.read.parquet(f"s3a://haven-database/copernicus-biochemistry/h3_resolution=4/region=chinook_study/date={date}/")
    biochemistry = biochemistry.filter(biochemistry.depth_bin == 25.0).select("h3_index", "net_primary_production")

    # Join to Choices
    environment = join_environment(choices, physics, biochemistry, CONTEXT)

    # Derive Features
    derived = derive_features(environment, CONTEXT)

    # Normalize
    normed = derived.withColumn("normed_distance", derived["distance"] / CONTEXT['max_km'])
    normed = normed.withColumn("normed_log_npp", log(normed["net_primary_production"]) - CONTEXT["mean_log_npp"])
    normed = normed.withColumn("normed_log_mlt", log(normed["mixed_layer_thickness"]) - CONTEXT["mean_log_mlt"])

    # Predict and Distribute Quanta
    predictions = infer(normed, model, CONTEXT["features"])
    predictions = predictions.withColumn("odds", exp(predictions["log_odds"]))
    predictions = predictions.withColumn("sum_odds", sum("odds").over(Window.partitionBy("origin_h3_index")))
    predictions = predictions.withColumn("probability", predictions["odds"] / predictions["sum_odds"])
    predictions = predictions.withColumn("_quanta", predictions["_quanta"] * predictions["probability"])

    # Group and Write
    grouped = predictions.groupby(CONTEXT["essential"]).agg(sum("_quanta").alias("_quanta"))

    db.write_partitions(
        grouped, 'spark_test_7', ['date']
    )

def build(spark, CONTEXT):
    # Pull and Format Inputs
    tag_tracks = spark.read.parquet("s3a://haven-database/mgietzmann_tag_tracks/").drop("upload_key")
    tag_tracks = apply_to_create_columns(
        tag_tracks, get_h3_index, ["latitude", "longitude"],
        [StructField("origin_h3_index", StringType())]
    ).drop(["latitude", "longitude"])
    tag_tracks = vectorized_create_column(
        tag_tracks, get_time, ["epoch"], "time"
    ).drop("epoch")
    tag_tracks = tag_tracks.withColumn("_decision", row_number().over(Window.orderBy()))

    # Add Individual and Decision Columns
    tag_tracks = tag_tracks.withColumn("_individual", dense_rank().over(Window.orderBy("tag_key").partitionBy()))
    tag_tracks = tag_tracks.withColumn("_decision", row_number().over(Window.partitionBy("_individual").orderBy("time")))

    # Create Choices
    choices = create_choices(tag_tracks, CONTEXT)

    # Prepare for Join Against Environment Data
    choices = vectorized_create_column(choices, strftime, ["time"], "date")

    # Pull Environment Data
    physics = spark.read.parquet("s3a://haven-database/copernicus-physics/h3_resolution=4/region=chinook_study/")
    physics = physics.filter(physics.depth_bin == 25.0).select("h3_index", "mixed_layer_thickness", "velocity_east", "velocity_north")
    biochemistry = spark.read.parquet("s3a://haven-database/copernicus-biochemistry/h3_resolution=4/region=chinook_study/")
    biochemistry = biochemistry.filter(biochemistry.depth_bin == 25.0).select("h3_index", "net_primary_production")

    # Join to Choices
    environment = join_environment(choices, physics, biochemistry, CONTEXT)

    # Derive Features
    derived = derive_features(environment, CONTEXT)

    # Determine Selected
    derived = derived.withColumn("next_h3_index", lead("origin_h3_index").over(Window.partitionBy("_individual").orderBy("time")))
    derived = derived.withColumn("_selected", derived["next_h3_index"] == derived["h3_index"])

    # Write it Back
    db.write_partitions(
        derived, "movement_model_raw_features_v4", ["tag_key"]
    )


if __name__ == '__main__':
    os.environ['AWS_REGION'] = 'us-east-1'
    os.environ['HAVEN_DATABASE'] = 'haven'

    CONTEXT = {
        "max_km": 100,
    }
    SIM_CONTEXT = {
        "mean_log_npp": 1.9670798, 
        "mean_log_mlt": 3.0952279761654187,
        "features": [
            "normed_log_mlt", "normed_log_npp", "normed_distance", 
            "water_heading", "movement_heading"
        ],
        "essential": [
            "date", "h3_index", "time"
        ],
    }

    spark = SparkSession.builder
    spark = db.configure(spark)
    spark = spark.getOrCreate()

    mode = sys.argv[1]
    if mode == "simulate":
        CONTEXT.update(SIM_CONTEXT)
        simulate(spark, CONTEXT)
    elif mode == "build":
        build(spark, CONTEXT)
    else:
        raise Exception("Invalid mode:", mode)
    
    spark.stop()
