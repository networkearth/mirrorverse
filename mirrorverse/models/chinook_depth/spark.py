import os
import uuid
import shutil

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
)
from pyspark.sql.functions import udf, pandas_udf, lit
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


def create_common_context(states, elevation, depth_classes):
    states = apply_to_create_columns(
        states,
        partial(get_depth_class, depth_classes),
        ["depth"],
        [StructField("_selected", FloatType())],
    )
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
    # states = join(states, elevation, on="h3_level_4_key")
    return states


def create_choices(states, depth_classes):
    for i, depth_class in enumerate(depth_classes):
        states = states.withColumn(f"depth_class_{i}", lit(int(depth_class)))
    return states


def create_features(states, depth_classes, features):
    for i, _ in enumerate(depth_classes):
        for feature in features:
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
                float_list=tf.train.FloatList(value=[row.month])
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
    # Sanitize Inputs
    depth_classes = np.array(eval(depth_classes))
    features = eval(features)

    states = load_training_states(spark, connection)
    elevation = load_elevation(spark, connection)

    states = create_common_context(states, elevation, depth_classes)
    states = create_choices(states, depth_classes)
    states = create_features(states, depth_classes, features)

    train_states, test_states = train_test_split(states, split)

    save_to_tfrecord(train_states, features, train_dir, depth_classes, overwrite)
    save_to_tfrecord(test_states, features, test_dir, depth_classes, overwrite)


build_training_data(
    spark,
    ps_conn,
    depth_classes="[25, 50, 75, 100, 150, 200, 250, 300, 400, 500]",
    features='["month", "daytime", "period_progress"]',
    train_dir="train",
    test_dir="test",
    split=0.8,
    overwrite=True,
)
