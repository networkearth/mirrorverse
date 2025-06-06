CREATE TABLE "haven"."chinook_depth_contrasts_3_5" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/chinook_depth_contrasts_3_5/',
  write_compression = 'SNAPPY',
  partitioned_by = array['_train', 'partition']
) AS 
select
    h.habitat,
    c.*
from 
    chinook_depth_contrasts_3_1 c
    inner join chlorophyll_habitat_index_1 h 
        on h.year = 2016 and h.month = 6 
        and h.h3_index = c.h3_index