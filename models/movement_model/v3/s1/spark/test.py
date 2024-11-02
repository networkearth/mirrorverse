import os

import haven.spark as db 
from pyspark.sql import SparkSession


if __name__ == "__main__":
    os.environ['AWS_REGION'] = 'us-east-1'
    os.environ['HAVEN_DATABASE'] = 'haven'

    spark = SparkSession.builder
    spark = db.configure(spark)
    spark = spark.getOrCreate()

    qrb = f"s3://aws-athena-query-results-575101084097-us-east-1"
    sql = """
        select 
            * 
        from 
            haven.copernicus_physics 
        where 
            depth_bin = 25 
            and date in ('2021-01-01', '2021-01-02')
    """
    df = db.read_data(
        sql, spark, qrb, public_internet_access=True
    )

    db.write_partitions(
        df, 'spark_test_1', ["date"]
    )
    db.register_partitions(
        'spark_test_1', public_internet_access=True
    )

    spark.stop()
