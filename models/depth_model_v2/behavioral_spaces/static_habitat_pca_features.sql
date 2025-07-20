CREATE TABLE "haven"."chinook_depth_features_3_8" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/chinook_depth_features_3_8/',
  write_compression = 'SNAPPY',
  partitioned_by = array['_train']
) AS 
select
    h.component_0,
    h.component_1,
    h.component_2,
    c.*
from 
    chinook_depth_features_3 c 
    inner join static_pca_habitat h 
        on h.version = 1
        and h.h3_index = c.h3_index