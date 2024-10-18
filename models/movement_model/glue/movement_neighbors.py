import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import add_h3_options
from awsglue import DynamicFrame
import gs_explode

def sparkSqlQuery(glueContext, query, mapping, transformation_ctx) -> DynamicFrame:
    for alias, frame in mapping.items():
        frame.toDF().createOrReplaceTempView(alias)
    result = spark.sql(query)
    return DynamicFrame.fromDF(result, glueContext, transformation_ctx)
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Script generated for node AWS Glue Data Catalog
AWSGlueDataCatalog_node1728998939180 = glueContext.create_dynamic_frame.from_catalog(database="haven", table_name="movement_model_decisions", transformation_ctx="AWSGlueDataCatalog_node1728998939180")

# Script generated for node Add H3 Options
AddH3Options_node1728999213484 = AWSGlueDataCatalog_node1728998939180.add_h3_options(col="current_h3_index", max_km="100")

# Script generated for node Explode Array Or Map Into Rows
ExplodeArrayOrMapIntoRows_node1728999520802 = AddH3Options_node1728999213484.gs_explode(colName="neighbors", newCol="neighbor_h3_index")

# Script generated for node Drop Fields
DropFields_node1728999684570 = DropFields.apply(frame=ExplodeArrayOrMapIntoRows_node1728999520802, paths=["neighbors"], transformation_ctx="DropFields_node1728999684570")

# Script generated for node SQL Query
SqlQuery0 = '''
select 
    *,
    100 as max_km
from myDataSource
'''
SQLQuery_node1728999894404 = sparkSqlQuery(glueContext, query = SqlQuery0, mapping = {"myDataSource":DropFields_node1728999684570}, transformation_ctx = "SQLQuery_node1728999894404")

# Script generated for node Amazon S3
AmazonS3_node1729000076291 = glueContext.getSink(path="s3://haven-database/movement_model_neighbors/", connection_type="s3", updateBehavior="UPDATE_IN_DATABASE", partitionKeys=["max_km"], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1729000076291")
AmazonS3_node1729000076291.setCatalogInfo(catalogDatabase="haven",catalogTableName="movement_model_neighbors")
AmazonS3_node1729000076291.setFormat("glueparquet", compression="snappy")
AmazonS3_node1729000076291.writeFrame(SQLQuery_node1728999894404)
job.commit()