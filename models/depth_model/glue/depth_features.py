import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import add_depth_class
from awsglue import DynamicFrame

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
AWSGlueDataCatalog_node1725837638799 = glueContext.create_dynamic_frame.from_catalog(database="haven", table_name="mgietzmann_tag_depths", transformation_ctx="AWSGlueDataCatalog_node1725837638799")

# Script generated for node SQL Query
SqlQuery0 = '''
select 
    tag_key,
    epoch - epoch % 300 as epoch,
    avg(depth) as depth,
    avg(temperature) as temperature
from 
    myDataSource 
where 
    depth is not null 
    and temperature is not null
group by 
    1, 2
'''
SQLQuery_node1725838239173 = sparkSqlQuery(glueContext, query = SqlQuery0, mapping = {"myDataSource":AWSGlueDataCatalog_node1725837638799}, transformation_ctx = "SQLQuery_node1725838239173")

# Script generated for node Add Depth Class
AddDepthClass_node1725837653823 = SQLQuery_node1725838239173.add_depth_class(depth_classes="25,50,75,100,150,200,250,300,400,500", depth_column="depth")

# Script generated for node Amazon S3
AmazonS3_node1725837948039 = glueContext.getSink(path="s3://haven-database/tag_depth_features_mk1/", connection_type="s3", updateBehavior="LOG", partitionKeys=["tag_key"], enableUpdateCatalog=True, transformation_ctx="AmazonS3_node1725837948039")
AmazonS3_node1725837948039.setCatalogInfo(catalogDatabase="haven",catalogTableName="tag_depth_features_mk1")
AmazonS3_node1725837948039.setFormat("glueparquet", compression="snappy")
AmazonS3_node1725837948039.writeFrame(AddDepthClass_node1725837653823)
job.commit()