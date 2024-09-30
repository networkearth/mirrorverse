import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import add_h3
import add_suntimes

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Script generated for node AWS Glue Data Catalog
AWSGlueDataCatalog_node1725835278301 = glueContext.create_dynamic_frame.from_catalog(database="haven", table_name="mgietzmann_tag_tracks", transformation_ctx="AWSGlueDataCatalog_node1725835278301")

# Script generated for node Rename Field
RenameField_node1725835805896 = RenameField.apply(frame=AWSGlueDataCatalog_node1725835278301, old_name="longitude", new_name="lon", transformation_ctx="RenameField_node1725835805896")

# Script generated for node Rename Field
RenameField_node1725835814418 = RenameField.apply(frame=RenameField_node1725835805896, old_name="latitude", new_name="lat", transformation_ctx="RenameField_node1725835814418")

# Script generated for node Add Suntimes
AddSuntimes_node1725835890362 = RenameField_node1725835814418.add_suntimes()

# Script generated for node Add H3
AddH3_node1725835908317 = AddSuntimes_node1725835890362.add_h3(resolution="4", lat_column="lat", lon_column="lon")

# Script generated for node Select Fields
SelectFields_node1725836804813 = SelectFields.apply(frame=AddH3_node1725835908317, paths=["lat", "epoch", "sunset", "h3_resolution", "h3_index", "tag_key", "h3_lon_bin", "h3_lat_bin", "sunrise", "lon"], transformation_ctx="SelectFields_node1725836804813")

# Script generated for node Amazon S3
AmazonS3_node1725837064886 = glueContext.getSink(path="s3://haven-database/tag_position_features_mk1/", connection_type="s3", updateBehavior="LOG", partitionKeys=["tag_key"], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1725837064886")
AmazonS3_node1725837064886.setCatalogInfo(catalogDatabase="haven",catalogTableName="tag_position_features_mk1")
AmazonS3_node1725837064886.setFormat("glueparquet", compression="snappy")
AmazonS3_node1725837064886.writeFrame(SelectFields_node1725836804813)
job.commit()