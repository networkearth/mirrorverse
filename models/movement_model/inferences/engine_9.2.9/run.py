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
    exp, sum, lit, when, col
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
def month_in_radians(date):
    return (date.dt.month - 1) / 11 * 2 * np.pi

@pandas_udf(FloatType())
def apply_sin(input):
    return np.sin(input)

@pandas_udf(FloatType())
def apply_cos(input):
    return np.cos(input)

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

#def join_environment(dataframe, physics, biochemistry, CONTEXT, prefix):
def join_environment(dataframe, physics, CONTEXT, prefix):
    environment = join(dataframe, physics, on=['h3_index', 'time'], how='inner')
    #environment = join(environment, biochemistry, on=['h3_index', 'time'], how='inner')
    label_and_collect(" ".join([prefix, "Joining Environment"]), [environment])
    return environment

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
    #derived = vectorized_create_column(
    #    derived, add_water_heading, ["velocity_north", "velocity_east"],
    #    "water_heading"
    #)
    derived = vectorized_create_column(
        derived, add_movement_heading, ["origin_lat", "origin_lon", "lat", "lon"],
        "movement_heading"
    )
    derived = vectorized_create_column(
        derived, apply_cos, ["movement_heading"],
        "cos_mh"
    )
    derived = vectorized_create_column(
        derived, apply_sin, ["movement_heading"],
        "sin_mh"
    )


    derived = vectorized_create_column(
        derived, month_in_radians, ["time"],
        "month_radians"
    )
    derived = vectorized_create_column(
        derived, apply_cos, ["month_radians"],
        "cos_time"
    )
    derived = vectorized_create_column(
        derived, apply_sin, ["month_radians"],
        "sin_time"
    )

    derived = derived.withColumn("stay_put", when(col("distance") == 0.0, 1.0).otherwise(0.0))
    label_and_collect(" ".join([prefix, "Deriving Features"]), [derived])
    return derived

def pull_environment(sub_path, CONTEXT, prefix):
    physics = read_from_partitioned_table(
        "s3a://haven-database/copernicus-physics/",
        sub_path
    )
    physics = physics.filter(physics.depth_bin == 25.0).select("h3_index", "mixed_layer_thickness", "salinity", "date")
    physics = physics.withColumn("time", physics["date"])
    #biochemistry = read_from_partitioned_table(
    #    "s3a://haven-database/copernicus-biochemistry/",
    #    sub_path
    #)
    #biochemistry = biochemistry.withColumn("time", biochemistry["date"])
    #biochemistry = biochemistry.filter(biochemistry.depth_bin == 25.0).select("h3_index", "net_primary_production", "time")
    #label_and_collect(" ".join([prefix, "Pulling Environment"]), [physics, biochemistry])
    return physics#, biochemistry

# MAIN FUNCTIONS -----------------------------------------

def simulate(spark, model, CONTEXT):
    # Pull and Format Inputs
    # TODO Temporary for testing this should be an S3 bucket

    from datetime import datetime, timedelta
    date = datetime(2020, 2, 1)
   
    #INPUT = pd.DataFrame([
    #    {'_quanta': 10.0, 'h3_index': '840c9ebffffffff', 'time': date},
    #    {'_quanta': 10.0, 'h3_index': '840c699ffffffff', 'time': date}
    #])

    #grouped = spark.createDataFrame(INPUT)

    
    grouped = spark.read.parquet("s3a://haven-database/copernicus-physics/h3_resolution=4/region=chinook_study/date=2020-01-01/")
    grouped = grouped.select("h3_index").dropDuplicates()
    grouped = grouped.withColumn("time", lit(date))
    grouped = grouped.withColumn("_quanta", lit(10.0))
    grouped = grouped.withColumn("date", lit(date.strftime("%Y-%m-%d")))

    label_and_collect("Pulling Inputs", [grouped])

    label("Writing Inputs")
    db.write_partitions(
        grouped, CONTEXT["simulate_table"], ['date']
    )

    table_folder = CONTEXT["simulate_table"].replace('_', '-')

    for i in range(CONTEXT["steps"]):
        # read the last written output in order to ensure we don't end up
        # with an unordinately long chain of operations
        date_str = date.strftime("%Y-%m-%d")
        grouped = spark.read.parquet(f"s3a://haven-database/{table_folder}/date={date_str}/")

        # Add Individual and Decision Columns
        grouped = grouped.withColumnRenamed("h3_index", "origin_h3_index")
        grouped = grouped.withColumn("_individual", grouped["origin_h3_index"])
        grouped = grouped.withColumn("_decision", grouped["_individual"])

        # Create Choices
        choices = create_choices(grouped, CONTEXT, prefix=f"{i}")

        # Pull Environment Data
        #label(f"{i} Pulling Current Date")
        #date = choices.head(1)[0]["time"].strftime("%Y-%m-%d")
        #physics, biochemistry = pull_environment(
        physics = pull_environment(
            f"h3_resolution=4/region=chinook_study/date={date_str}/",
            CONTEXT, prefix=f"{i}"
        )

        # Join to Choices
        #environment = join_environment(choices, physics, biochemistry, CONTEXT, prefix=f"{i}")
        environment = join_environment(choices, physics, CONTEXT, prefix=f"{i}")

        # Derive Features
        derived = derive_features(environment, CONTEXT, prefix=f"{i}")

        # Normalize
        normed = derived.withColumn(
            "normed_salinity", 
            (derived["salinity"] - CONTEXT["mean_salinity"]) / CONTEXT["std_salinity"]
        )
        normed = normed.withColumn("normed_log_mlt", log(normed["mixed_layer_thickness"] + 0.01) / np.log(CONTEXT["max_mlt"] + 0.01))
        label_and_collect(f"{i} Normalizing Data", [normed])

        # Predict and Distribute Quanta
        predictions = infer(normed, model, CONTEXT["features"], prefix=f"{i}")
        predictions = predictions.withColumn("odds", exp(predictions["log_odds"]))
        predictions = predictions.withColumn("sum_odds", sum("odds").over(Window.partitionBy("origin_h3_index")))
        predictions = predictions.withColumn("probability", predictions["odds"] / predictions["sum_odds"])
        predictions = predictions.withColumn("_quanta", predictions["_quanta"] * predictions["probability"])
        label_and_collect(f"{i} Building Probabilities", [predictions])

        # Group, Update Timestamps, and Write
        grouped = predictions.groupby(CONTEXT["essential"]).agg(sum("_quanta").alias("_quanta"))
        grouped = vectorized_create_column(
            grouped, step_forward, ['time'], 'time'
        )
        date = date + timedelta(days=1)
        grouped = grouped.withColumn('date', lit(date.strftime("%Y-%m-%d")))
        label_and_collect(f"{i} Grouping by H3", [grouped])

        label(f"{i} Writing Step")
        db.write_partitions(
            grouped, CONTEXT["simulate_table"], ['date']
        )


def diffuse(spark, model, CONTEXT):
    # Pull and Format Inputs
    # TODO Temporary for testing this should be an S3 bucket

    from datetime import datetime, timedelta
    date = datetime(2020, 5, 1)
   
    #INPUT = pd.DataFrame([
    #    {'_quanta': 10.0, 'h3_index': '840c9ebffffffff', 'time': date},
    #    {'_quanta': 10.0, 'h3_index': '840c699ffffffff', 'time': date}
    #])

    #grouped = spark.createDataFrame(INPUT)

    

    #label("Writing Inputs")
    #db.write_partitions(
    #    grouped, CONTEXT["simulate_table"], ['date']
    #)

    #table_folder = CONTEXT["simulate_table"].replace('_', '-')

    for i in range(CONTEXT["steps"]):
        # read the last written output in order to ensure we don't end up
        # with an unordinately long chain of operations
        date_str = date.strftime("%Y-%m-%d")

        #INPUT = pd.DataFrame([
        #    {'_quanta': 1.0, 'h3_index': '840c9ebffffffff', 'time': date},
        #    {'_quanta': 1.0, 'h3_index': '840c699ffffffff', 'time': date}
        #])

        #grouped = spark.createDataFrame(INPUT)

        grouped = spark.read.parquet("s3a://haven-database/copernicus-physics/h3_resolution=4/region=chinook_study/date=2020-01-01/")
        grouped = grouped.select("h3_index").dropDuplicates()
        grouped = grouped.withColumn("time", lit(date))
        grouped = grouped.withColumn("_quanta", lit(1.0))
        #grouped = grouped.withColumn("date", lit(date.strftime("%Y-%m-%d")))

        label_and_collect("Pulling Inputs", [grouped])

        # Add Individual and Decision Columns
        grouped = grouped.withColumnRenamed("h3_index", "origin_h3_index")
        grouped = grouped.withColumn("_individual", grouped["origin_h3_index"])
        grouped = grouped.withColumn("_decision", grouped["_individual"])

        # Create Choices
        choices = create_choices(grouped, CONTEXT, prefix=f"{i}")

        # Pull Environment Data
        #label(f"{i} Pulling Current Date")
        #date = choices.head(1)[0]["time"].strftime("%Y-%m-%d")
        #physics, biochemistry = pull_environment(
        physics = pull_environment(
            f"h3_resolution=4/region=chinook_study/date={date_str}/",
            CONTEXT, prefix=f"{i}"
        )

        # Join to Choices
        #environment = join_environment(choices, physics, biochemistry, CONTEXT, prefix=f"{i}")
        environment = join_environment(choices, physics, CONTEXT, prefix=f"{i}")

        # Derive Features
        derived = derive_features(environment, CONTEXT, prefix=f"{i}")

        # Normalize
        normed = derived.withColumn(
            "normed_salinity", 
            (derived["salinity"] - CONTEXT["mean_salinity"]) / CONTEXT["std_salinity"]
        )
        normed = normed.withColumn("normed_log_mlt", log(normed["mixed_layer_thickness"] + 0.01) / np.log(CONTEXT["max_mlt"] + 0.01))
        label_and_collect(f"{i} Normalizing Data", [normed])

        # Predict and Distribute Quanta
        predictions = infer(normed, model, CONTEXT["features"], prefix=f"{i}")
        predictions = predictions.withColumn("odds", exp(predictions["log_odds"]))
        predictions = predictions.withColumn("sum_odds", sum("odds").over(Window.partitionBy("origin_h3_index")))
        predictions = predictions.withColumn("probability", predictions["odds"] / predictions["sum_odds"])
        predictions = predictions.withColumn("_quanta", predictions["_quanta"] * predictions["probability"])

        label(f"{i} Writing Step")
        predictions = predictions.select("h3_index", "origin_h3_index", "time", "date", "probability")
        db.write_partitions(
            predictions, CONTEXT["diffusion_table"], ['date']
        )

        date = date + timedelta(days=1)

        #label_and_collect(f"{i} Building Probabilities", [predictions])

        # Group, Update Timestamps, and Write
        #grouped = predictions.groupby(CONTEXT["essential"]).agg(sum("_quanta").alias("_quanta"))
        #grouped = vectorized_create_column(
        #    grouped, step_forward, ['time'], 'time'
        #)
        #date = date + timedelta(days=1)
        #grouped = grouped.withColumn('date', lit(date.strftime("%Y-%m-%d")))
        #label_and_collect(f"{i} Grouping by H3", [grouped])

        #label(f"{i} Writing Step")
        #db.write_partitions(
        #    grouped, CONTEXT["simulate_table"], ['date']
        #)

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

    # Pull Environment Data
    physics, biochemistry = pull_environment(
        "h3_resolution=4/region=chinook_study/",
        CONTEXT, ""
    )

    # Join to Choices
    environment = join_environment(choices, physics, biochemistry, CONTEXT, "")

    # Derive Features
    derived = derive_features(environment, CONTEXT, "")


    # Determine Selected
    derived = derived.withColumn("_selected", derived["next_h3_index"] == derived["h3_index"])

    # Filter Columns
    results = derived.select(
        "tag_key", "_decision", "_choice", "_selected", "_individual", "mixed_layer_thickness", "net_primary_production",
        "water_heading", "movement_heading", "distance", "origin_h3_index", "next_h3_index", "h3_index", "time", "_train"
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
        "max_km": 50,
        "simulate_table": "movement_model_simulation_m9_a2_v9_r3",
        "diffusion_table": "movement_model_diffusion_m9_a2_v9_t3",
        "build_table": "",
        "case": "train"
    }
    SIM_CONTEXT = {
        "mean_salinity": 31.769322063327927,
        "std_salinity": 0.7058350459612885, 
        "max_mlt": 153.96588768064976,
        "features": [
            "sin_mh", "cos_mh",
            "cos_time", "sin_time",
            "normed_salinity",
            "normed_log_mlt",
            "stay_put"
        ],
        "essential": [
            "h3_index", "time"
        ],
        "steps": 31,
    }

    spark = SparkSession.builder
    spark = db.configure(spark)
    spark = spark.getOrCreate()

    mode = sys.argv[1]
    if mode == "simulate":
        model = load_model(
            'mimic-log-odds', 'movement-model-m9-a2-v9', 
            '985acb97fdf84aaef5f7076a7c63bf93a7c6ee6703e71e7ffbf4b24743a0a773'
        )
        CONTEXT.update(SIM_CONTEXT)
        simulate(spark, model, CONTEXT)
    elif mode == "build":
        build(spark, CONTEXT)
    elif mode == "diffuse":
        model = load_model(
            'mimic-log-odds', 'movement-model-m9-a2-v9', 
            '985acb97fdf84aaef5f7076a7c63bf93a7c6ee6703e71e7ffbf4b24743a0a773'
        )
        CONTEXT.update(SIM_CONTEXT)
        diffuse(spark, model, CONTEXT)
    else:
        raise Exception("Invalid mode:", mode)
    
    spark.stop()
