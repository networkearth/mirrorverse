CREATE TABLE "haven"."chinook_depth_features_3_7" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/chinook_depth_features_3_7/',
  write_compression = 'SNAPPY',
  partitioned_by = array['_train']
) AS 
select
    h.lat,
    h.lon,
    c.*
from 
    chinook_depth_features_3 c
    inner join h3_to_position h 
        on h.version = 1
        and h.h3_index = c.h3_index