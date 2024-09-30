import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import add_h3
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as SqlFuncs

def sparkAggregate(glueContext, parentFrame, groups, aggs, transformation_ctx) -> DynamicFrame:
    aggsFuncs = []
    for column, func in aggs:
        aggsFuncs.append(getattr(SqlFuncs, func)(column))
    result = parentFrame.toDF().groupBy(*groups).agg(*aggsFuncs) if len(groups) > 0 else parentFrame.toDF().agg(*aggsFuncs)
    return DynamicFrame.fromDF(result, glueContext, transformation_ctx)

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Script generated for node Amazon S3
AmazonS3_node1725622907031 = glueContext.create_dynamic_frame.from_options(format_options={}, connection_type="s3", format="parquet", connection_options={"paths": ["s3://haven-database/elevation-uploads/"], "recurse": True}, transformation_ctx="AmazonS3_node1725622907031")

# Script generated for node Add H3
AddH3_node1725623086164 = AmazonS3_node1725622907031.add_h3(resolution="4", lat_column="lat", lon_column="lon")

# Script generated for node Drop Duplicates
DropDuplicates_node1725623805513 =  DynamicFrame.fromDF(AddH3_node1725623086164.toDF().dropDuplicates(), glueContext, "DropDuplicates_node1725623805513")

# Script generated for node Aggregate
Aggregate_node1725623866188 = sparkAggregate(glueContext, parentFrame = DropDuplicates_node1725623805513, groups = ["h3_resolution", "h3_index", "h3_lon_bin", "h3_lat_bin"], aggs = [["elevation", "avg"]], transformation_ctx = "Aggregate_node1725623866188")

# Script generated for node Rename Field
RenameField_node1725623972964 = RenameField.apply(frame=Aggregate_node1725623866188, old_name="`avg(elevation)`", new_name="elevation", transformation_ctx="RenameField_node1725623972964")

# Script generated for node Amazon S3
AmazonS3_node1725624065652 = glueContext.getSink(path="s3://haven-database/mean_elevation_by_h3/", connection_type="s3", updateBehavior="LOG", partitionKeys=["h3_resolution", "h3_lon_bin", "h3_lat_bin"], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1725624065652")
AmazonS3_node1725624065652.setCatalogInfo(catalogDatabase="haven",catalogTableName="mean_elevation_by_h3")
AmazonS3_node1725624065652.setFormat("glueparquet", compression="snappy")
AmazonS3_node1725624065652.writeFrame(RenameField_node1725623972964)
job.commit()