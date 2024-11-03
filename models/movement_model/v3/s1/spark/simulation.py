import os
from datetime import datetime
from functools import partial

import pandas as pd
import numpy as np
import h3
import geopy.distance
import boto3
import tensorflow.keras as keras

import haven.spark as db 
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, pandas_udf, lit, avg, std, explode, row_number, log, exp, sum
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    FloatType,
    IntegerType,
    TimestampType,
    LongType,
    ArrayType,
    StringType,
)


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

@pandas_udf(StringType())
def strftime(time):
    return time.dt.strftime('%Y-%m-%d')

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

def load_model(space, experiment_name, run_id):
    bucket_name = f"{space}-models"
    model_key = f"{experiment_name}/{run_id}/model.keras"
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, model_key, "model.keras")

    return keras.models.load_model("model.keras")


if __name__ == "__main__":
    os.environ['AWS_REGION'] = 'us-east-1'
    os.environ['HAVEN_DATABASE'] = 'haven'

    INPUT = pd.DataFrame([
        {'_quanta': 10.0, 'h3_index': '840c9ebffffffff', 'time': datetime(2020, 4, 17)},
        {'_quanta': 10.0, 'h3_index': '840c699ffffffff', 'time': datetime(2020, 4, 17)}
    ])
    CONTEXT = {
        'max_km': 100.0, 
        'mean_log_npp': 1.9670798, 
        'mean_log_mlt': 3.0952279761654187,
        'features': [
            "normed_log_mlt", "normed_log_npp", "normed_distance", 
            "water_heading", "movement_heading"
        ],
        'essential': [
            'date', 'h3_index', 'time'
        ],
        'min_quanta': 0.01
    }
    model = load_model(
        'mimic-log-odds', 'movement-model-experiment-v3-s1', 
        'e864f08e675a8bd39b0764be4827adf827b49064ed473695c4509cf0cabda693'
    )

    spark = SparkSession.builder
    spark = db.configure(spark)
    spark = spark.getOrCreate()

    distribution = spark.createDataFrame(INPUT)

    choices = apply_to_create_columns(
        distribution, partial(find_neighbors, CONTEXT['max_km']),
        ['h3_index'], [StructField("neighbors", ArrayType(StringType()))]
    )
    # origin h3 index will be our "_decision" index
    choices = choices.withColumnRenamed("h3_index", "origin_h3_index")
    choices = choices.withColumn("h3_index", explode(choices.neighbors))
    choices = choices.drop("neighbors")
    choices = choices.withColumn("_choice", row_number().over(Window.partitionBy("origin_h3_index").orderBy("h3_index")))
    choices = vectorized_create_column(choices, strftime, ["time"], "date")

    physics = spark.read.parquet("s3a://haven-database/copernicus-physics/h3_resolution=4/region=chinook_study/date=2020-04-17/")
    physics = physics.filter(physics.depth_bin == 25.0).select("h3_index", "mixed_layer_thickness", "velocity_east", "velocity_north")
    biochemistry = spark.read.parquet("s3a://haven-database/copernicus-biochemistry/h3_resolution=4/region=chinook_study/date=2020-04-17/")
    biochemistry = biochemistry.filter(biochemistry.depth_bin == 25.0).select("h3_index", "net_primary_production")

    environment = join(choices, physics, on='h3_index', how='inner')
    environment = join(environment, biochemistry, on='h3_index', how='inner')

    derived = apply_to_create_columns(
        environment, h3.h3_to_geo,
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

    normed = derived.withColumn("normed_distance", derived["distance"] / CONTEXT['max_km'])
    normed = normed.withColumn("normed_log_npp", log(normed["net_primary_production"]) - CONTEXT["mean_log_npp"])
    normed = normed.withColumn("normed_log_mlt", log(normed["mixed_layer_thickness"]) - CONTEXT["mean_log_mlt"])

    predictions = infer(normed, model, CONTEXT["features"])
    predictions = predictions.withColumn("odds", exp(predictions["log_odds"]))
    predictions = predictions.withColumn("sum_odds", sum("odds").over(Window.partitionBy("origin_h3_index")))
    predictions = predictions.withColumn("probability", predictions["odds"] / predictions["sum_odds"])
    predictions = predictions.withColumn("_quanta", predictions["_quanta"] * predictions["probability"])

    grouped = predictions.groupby(CONTEXT["essential"]).agg(sum("_quanta").alias("_quanta"))

    grouped.write.mode("overwrite").partitionBy('date').parquet('output')

    spark.stop()
