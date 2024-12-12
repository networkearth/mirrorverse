import sys
import os
from functools import partial

import h3
import pandas as pd
import numpy as np
import geopy.distance

import haven.spark as db 
import boto3
import tensorflow.keras as keras

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf, pandas_udf, explode,
    row_number, lead, log, 
    exp, sum, lit,
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

def label_and_collect(header, dataframes):
    spark.sparkContext.setJobGroup(header, header)
    # headers are added to everything run since 
    # the last collection so we have to collect 
    # to add our header
    for dataframe in dataframes:
        # if we don't cache then each time we set a new 
        # label and collect, every preceding stage will 
        # have to be run again 
        dataframe.cache()
        dataframe.count()

def label(header):
    spark.sparkContext.setJobGroup(header, header)

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
    dataframe = dataframe.repartition(*on)
    other = other.repartition(*on)
    return dataframe.join(other, on=on, how=how)

def load_model(space, experiment_name, run_id):
    bucket_name = f"{space}-models"
    model_key = f"{experiment_name}/{run_id}/model.keras"
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, model_key, "model.keras")

    return keras.models.load_model("model.keras")

def read_from_partitioned_table(table_path, sub_path):
    assert table_path.endswith('/') and not sub_path.startswith('/')
    return (
        spark.read
        .option("basePath", table_path) # let's it recognize all of the partitions
        .parquet(''.join([table_path, sub_path]))
    )

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

# VECTORIZED UDFs -----------------------------------------

@pandas_udf(TimestampType())
def get_time(epoch):
    return pd.to_datetime(epoch, unit='s')

@pandas_udf(FloatType())
def add_water_heading(velocity_north, velocity_east):
    return np.arctan2(velocity_north, velocity_east)

@pandas_udf(FloatType())
def add_movement_heading(origin_lat, origin_lon, lat, lon):
    return np.arctan2(lat - origin_lat, lon - origin_lon)

@pandas_udf(TimestampType())
def step_forward(date):
    return date + pd.DateOffset(days=1)

# OTHER -----------------------------------------

def infer_map(model, features, iterator):
    for dataframe in iterator:
        dataframe["log_odds"] = model.predict(dataframe[features])
        yield dataframe

def infer(dataframe, model, features, prefix):
    schema = StructType(
        dataframe.schema.fields
        + [
            StructField(f"log_odds", FloatType())
        ]
    )
    dataframe = dataframe.mapInPandas(
        partial(infer_map,  model, features), schema=schema
    )
    dataframe.cache()
    label_and_collect(" ".join([prefix, "Running Inference"]), [dataframe])
    return dataframe

# COMMON STEPS -----------------------------------------

def create_choices(dataframe, CONTEXT, prefix):
    choices = apply_to_create_columns(
        dataframe, partial(find_neighbors, CONTEXT["max_km"]),
        ['origin_h3_index'], [StructField("h3_index", ArrayType(StringType()))]
    )
    label(" ".join([prefix, "Creating Choices"]))

    # somehow the following "limit" makes the explode work
    # remove it and the explode just blocks... 
    count = choices.count()
    choices = choices.limit(count)
    choices = choices.withColumn("h3_index", explode(choices.h3_index))

    choices = choices.withColumn("_choice", row_number().over(Window.partitionBy(["_individual", "_decision"]).orderBy("h3_index")))

    label_and_collect(" ".join([prefix, "Creating Choices"]), [choices])
    return choices

def derive_features(dataframe, CONTEXT, prefix):
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
    derived = derived.withColumn("normed_distance", derived["distance"] / CONTEXT["max_km"])
    label_and_collect(" ".join([prefix, "Deriving Features"]), [derived])
    return derived

# MAIN FUNCTIONS -----------------------------------------

def simulate(spark, model, CONTEXT):
    pass

def build(spark, CONTEXT):
    # Pull and Format Inputs
    tag_tracks = spark.read.parquet(f"s3a://haven-database/mgietzmann-tag-tracks-{CONTEXT['case']}/").drop("upload_key")
    label_and_collect("Pulling Tag Tracks", [tag_tracks])
    tag_tracks = apply_to_create_columns(
        tag_tracks, get_h3_index, ["latitude", "longitude"],
        [StructField("origin_h3_index", StringType())]
    ).drop(*["latitude", "longitude"])
    tag_tracks = vectorized_create_column(
        tag_tracks, get_time, ["epoch"], "time"
    ).drop("epoch")

    # Add Individual and Decision Columns
    tag_tracks = tag_tracks.withColumn("_individual", tag_tracks["tag_key"])
    tag_tracks = tag_tracks.withColumn("_decision", row_number().over(Window.partitionBy("_individual").orderBy("time")))

    # Determine Next
    tag_tracks = tag_tracks.withColumn("next_h3_index", lead("origin_h3_index").over(Window.partitionBy("_individual").orderBy("time")))
    label_and_collect("Prepping Tag Tracks", [tag_tracks])

    # Create Choices
    choices = create_choices(tag_tracks, CONTEXT, "")

    # Derive Features
    derived = derive_features(choices, CONTEXT, "")

    # Determine Selected
    derived = derived.withColumn("_selected", derived["next_h3_index"] == derived["h3_index"])

    # Filter Columns
    results = derived.select(
        "tag_key", "_decision", "_choice", "_selected", "_individual", "normed_distance", "origin_h3_index", 
        "next_h3_index", "h3_index", "time", "_train"
    )

    # Write it Back
    label("Writing")
    db.write_partitions(
        results, CONTEXT["build_table"], ["tag_key"]
    )


if __name__ == '__main__':
    os.environ['AWS_REGION'] = 'us-east-1'
    os.environ['HAVEN_DATABASE'] = 'haven'

    CONTEXT = {
        "max_km": 100,
        "simulate_table": "movement_model_simulation_m1_a2",
        "build_table": "movement_model_raw_features_m1_a2",
        "case": "train",
    }
    SIM_CONTEXT = {
        "features": ["distance"],
        "essential": ["h3_index", "time", "date"],
        "steps": 62,
    }

    spark = SparkSession.builder
    spark = db.configure(spark)
    spark = spark.getOrCreate()

    mode = sys.argv[1]
    if mode == "simulate":
        model = load_model(
            'mimic-log-odds', 'movement-model-experiment-v3-s1', 
            'e864f08e675a8bd39b0764be4827adf827b49064ed473695c4509cf0cabda693'
        )
        CONTEXT.update(SIM_CONTEXT)
        simulate(spark, model, CONTEXT)
    elif mode == "build":
        build(spark, CONTEXT)
    else:
        raise Exception("Invalid mode:", mode)
    
    spark.stop()
