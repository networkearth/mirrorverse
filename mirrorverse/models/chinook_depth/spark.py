import os
import uuid
import shutil
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import h3
from functools import partial
from scipy.stats import norm
from pyspark.sql.types import (
    StructType,
    StructField,
    FloatType,
    IntegerType,
    TimestampType,
    LongType,
)
from pyspark.sql.functions import udf, pandas_udf, lit, avg, std
from suntimes import SunTimes


def load_from_database(spark, connection, sql, schema):
    return (
        spark.read.format("jdbc")
        .option("url", connection)
        .option("query", sql)
        .option("customSchema", schema)
        .load()
    )


def update_schema(dataframe, new_fields):
    return StructType(dataframe.schema.fields + new_fields)


def apply_to_create_columns(dataframe, func, input_cols, new_fields):
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
    return dataframe.withColumn(
        field, pandas_udf(*[dataframe[col] for col in input_cols])
    )


def join(dataframe, other, on, how="inner"):
    dataframe = dataframe.repartition(on)
    other = other.repartition(on)
    return dataframe.join(other, on=on, how=how)


def load_training_states(spark, connection):
    sql = """
    select 

        d.epoch,
        t.h3_level_4_key,

        d.depth,
        "t" || t.tag_key as _individual,
        ROW_NUMBER() over () as _decision

    from 
        tag_depths d
        inner join tag_tracks t 
            on d.tag_key = t.tag_key
            and d.date_key = t.date_key
    where
        d.depth is not null
    """
    schema = "epoch INTEGER, h3_level_4_key BIGINT, depth FLOAT, _individual STRING, _decision INTEGER"
    return load_from_database(spark, connection, sql, schema)


def load_infer_states(spark, connection, sql):
    schema = "epoch INTEGER, h3_level_4_key BIGINT"
    return load_from_database(spark, connection, sql, schema)


def load_elevation(spark, connection):
    sql = """
    select 
        h3_level_4_key,
        elevation
    from
        elevation
    """
    schema = "h3_level_4_key BIGINT, elevation FLOAT"
    return load_from_database(spark, connection, sql, schema)


def get_depth_class(depth_classes, depth):
    sd = (
        depth * 0.08 / 1.96
    )  # ~two standard deviations gives our 95% confidence interval
    if sd == 0:
        division = np.zeros(len(depth_classes))
        division[0] = 1
    else:
        # we're going to assume the depth classes are sorted
        z = (depth_classes - depth) / sd
        division = norm.cdf(z)
        division[1:] = division[1:] - division[:-1]
    # if there aren't quite enough depth classes the
    # probabilities may not sum to 1, so we'll normalize
    division = division / division.sum()
    return float(np.random.choice(depth_classes, p=division))


def h3_to_geo(h3_key):
    h3_key = hex(h3_key)[2:]
    return h3.h3_to_geo(h3_key)


@pandas_udf(TimestampType())
def to_datetime(epoch):
    return pd.to_datetime(epoch, utc=True, unit="s")


@pandas_udf(IntegerType())
def get_month(date):
    return date.dt.month


def get_sunset(lat, lon, date):
    return SunTimes(longitude=lon, latitude=lat, altitude=0).setwhere(date, "UTC").hour


def get_sunrise(lat, lon, date):
    return SunTimes(longitude=lon, latitude=lat, altitude=0).risewhere(date, "UTC").hour


@pandas_udf(IntegerType())
def get_daytime(date, sunrise, sunset):
    hour = date.dt.hour
    return (
        (sunrise > sunset) * ((hour < sunset) + (hour >= sunrise))
        + (sunrise < sunset) * ((hour >= sunrise) * (hour < sunset))
    ).astype(int)


@pandas_udf(FloatType())
def get_period_progress(date, sunrise, sunset, daytime):
    hour = date.dt.hour
    hours_to_transition = (
        (
            (hour > sunrise) * (24 - hour + sunrise)
            + (hour <= sunrise) * (sunrise - hour)
        )
        * (1 - daytime)
    ).astype(float) + (
        ((hour > sunset) * (24 - hour + sunset) + (hour <= sunset) * (sunset - hour))
        * (daytime)
    ).astype(
        float
    )

    interval = (1 - daytime) * (
        (sunrise >= sunset) * (sunrise - sunset)
        + (sunrise < sunset) * (24 - sunset + sunrise)
    ) + (daytime) * (
        (sunset >= sunrise) * (sunset - sunrise)
        + (sunset < sunrise) * (24 - sunrise + sunset)
    )

    return (1 - hours_to_transition / interval).astype(float)


def create_common_context(states, elevation):
    states = apply_to_create_columns(
        states,
        h3_to_geo,
        ["h3_level_4_key"],
        [
            StructField("latitude", FloatType()),
            StructField("longitude", FloatType()),
        ],
    )
    states = vectorized_create_column(
        states,
        to_datetime,
        ["epoch"],
        "datetime",
    )
    states = vectorized_create_column(
        states,
        get_month,
        ["datetime"],
        "month",
    )
    states = apply_to_create_columns(
        states,
        get_sunrise,
        ["latitude", "longitude", "datetime"],
        [StructField("sunrise", IntegerType())],
    )
    states = apply_to_create_columns(
        states,
        get_sunset,
        ["latitude", "longitude", "datetime"],
        [StructField("sunset", IntegerType())],
    )
    states = vectorized_create_column(
        states,
        get_daytime,
        ["datetime", "sunrise", "sunset"],
        "daytime",
    )
    states = vectorized_create_column(
        states,
        get_period_progress,
        ["datetime", "sunrise", "sunset", "daytime"],
        "period_progress",
    )
    states = join(states, elevation, on="h3_level_4_key")
    return states


def create_choices(states, depth_classes):
    for i, depth_class in enumerate(depth_classes):
        states = states.withColumn(f"depth_class_{i}", lit(int(depth_class)))
    return states


def create_features(states, depth_classes, features):
    for i, _ in enumerate(depth_classes):
        for feature in features:
            if feature != "depth_class":
                states = states.withColumn(f"{feature}_{i}", states[feature])
    return states


def train_test_split(states, split):
    individuals = (
        states.select("_individual").distinct().rdd.map(lambda row: row[0]).collect()
    )
    train_individuals = set(
        np.random.choice(individuals, int(len(individuals) * split), replace=False)
    )
    test_individuals = set(individuals) - train_individuals
    train_states = states.filter(states["_individual"].isin(train_individuals))
    test_states = states.filter(states["_individual"].isin(test_individuals))
    return train_states, test_states


def explode_func(N, features, iterator):
    for dataframe in iterator:
        for i in range(N):
            sub_dataframe = dataframe[[f"{feature}_{i}" for feature in features]]
            sub_dataframe = sub_dataframe.rename(
                {f"{feature}_{i}": feature for feature in features}, axis=1
            )
            for feature in features:
                sub_dataframe[feature] = sub_dataframe[feature].astype(float)
            yield sub_dataframe


def explode(states, N, features):
    return states.mapInPandas(
        partial(explode_func, N, features),
        schema=StructType([StructField(feature, FloatType()) for feature in features]),
    )


def get_normalization_parameters(states, N, features):
    states = explode(states, N, features)
    avg_aggs = [avg(feature).alias(feature) for feature in features]
    std_aggs = [std(feature).alias(feature) for feature in features]
    return {
        "avg": states.agg(*avg_aggs).collect()[0].asDict(),
        "std": states.agg(*std_aggs).collect()[0].asDict(),
    }


def normalize(states, normalization_parameters, N, features):
    avg = normalization_parameters["avg"]
    std = normalization_parameters["std"]
    for i in range(N):
        for feature in features:
            states = states.withColumn(
                f"{feature}_{i}",
                (states[f"{feature}_{i}"] - avg[feature]) / std[feature],
            )
    return states


def serialize_example(depth_classes, features, row):
    # Create a dictionary with the features
    depth_classes = list(depth_classes)
    feature = {
        "_selected": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[depth_classes.index(row._selected)])
        ),
    }
    for i, _ in enumerate(depth_classes):
        current_feature = {
            f"{feature}_{i}": tf.train.Feature(
                float_list=tf.train.FloatList(value=[row[f"{feature}_{i}"]])
            )
            for feature in features
        }
        feature.update(current_feature)
    # Create an Example protocol buffer
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_partition(output_dir, iterator):
    partition_id = str(uuid.uuid4())
    file_path = os.path.join(output_dir, f"part-{partition_id}.tfrecord")
    with tf.io.TFRecordWriter(file_path) as writer:
        for record in iterator:
            writer.write(record)


def save_to_tfrecord(dataframe, features, output_dir, depth_classes, overwrite=False):
    if os.path.exists(output_dir):
        assert overwrite, f"{output_dir} already exists"
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    rdd = dataframe.rdd.map(partial(serialize_example, depth_classes, features))
    rdd.foreachPartition(partial(write_partition, output_dir))


def infer_map(depth_classes, features, model, iterator):
    for dataframe in iterator:
        inputs = {}
        for i, _ in enumerate(depth_classes):
            sub_features = [f"{feature}_{i}" for feature in features]
            inputs[f"input_{i}"] = dataframe[sub_features]
        predictions = model.predict(inputs)
        for i, _ in enumerate(depth_classes):
            dataframe[f"probability_{i}"] = predictions[:, i]
        yield dataframe


def infer(states, depth_classes, features, model):
    schema = StructType(
        states.schema.fields
        + [
            StructField(f"probability_{i}", FloatType())
            for i, _ in enumerate(depth_classes)
        ]
    )
    states = states.mapInPandas(
        partial(infer_map, depth_classes, features, model), schema=schema
    )
    return states


def explode_to_new_states_func(depth_classes, iterator):
    for dataframe in iterator:
        for i, depth_class in enumerate(depth_classes):
            columns = ["h3_level_4_key", "epoch", f"probability_{i}"]
            new_state = dataframe[columns]
            new_state = new_state.rename({f"probability_{i}": "probability"}, axis=1)
            new_state["depth_class"] = depth_class
            yield new_state


def explode_to_new_states(states, depth_classes):
    schema = StructType(
        [
            StructField("h3_level_4_key", LongType()),
            StructField("epoch", IntegerType()),
            StructField("depth_class", FloatType()),
            StructField("probability", FloatType()),
        ]
    )
    states = states.mapInPandas(
        partial(explode_to_new_states_func, depth_classes), schema=schema
    )
    return states


def write_to_database(states, connection, table, mode="overwrite"):
    states.write.jdbc(url=connection, table=table, mode=mode)


def build_training_data(
    spark,
    connection,
    depth_classes,
    features,
    train_dir,
    test_dir,
    split,
    overwrite=False,
):
    states = load_training_states(spark, connection)
    elevation = load_elevation(spark, connection)

    states = apply_to_create_columns(
        states,
        partial(get_depth_class, depth_classes),
        ["depth"],
        [StructField("_selected", FloatType())],
    )

    states = create_common_context(states, elevation)
    states = create_choices(states, depth_classes)
    states = create_features(states, depth_classes, features)

    train_states, test_states = train_test_split(states, split)

    normalization_parameters = get_normalization_parameters(
        train_states, len(depth_classes), features
    )
    train_states = normalize(
        train_states, normalization_parameters, len(depth_classes), features
    )
    test_states = normalize(
        test_states, normalization_parameters, len(depth_classes), features
    )

    save_to_tfrecord(train_states, features, train_dir, depth_classes, overwrite)
    save_to_tfrecord(test_states, features, test_dir, depth_classes, overwrite)
    with open(os.path.join(train_dir, "normalization_parameters.json"), "w") as f:
        json.dump(normalization_parameters, f, sort_keys=True, indent=4)


def run_inference(
    spark,
    connection,
    depth_classes,
    features,
    query,
    normalization_parameters_path,
    model_path,
    table,
):
    with open(normalization_parameters_path, "r") as f:
        normalization_parameters = json.load(f)

    model = tf.keras.models.load_model(model_path)

    states = load_infer_states(spark, connection, query)
    elevation = load_elevation(spark, connection)
    states = create_common_context(states, elevation)
    states = create_choices(states, depth_classes)
    states = create_features(states, depth_classes, features)
    states = normalize(states, normalization_parameters, len(depth_classes), features)
    states = infer(states, depth_classes, features, model)
    states = explode_to_new_states(states, depth_classes)
    write_to_database(states, connection, table)
