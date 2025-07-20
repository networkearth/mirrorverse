CREATE TABLE "haven"."chinook_depth_features_3_6" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/chinook_depth_features_3_6/',
  write_compression = 'SNAPPY',
  partitioned_by = array['_train']
) AS 
select
    h.component_0,
    h.component_1,
    h.component_2,
    h.component_3,
    h.component_4,
    h.component_5,
    h.component_6,
    h.component_7,
    h.component_8,
    h.component_9,
    h.component_10,
    h.component_11,
    c.*
from 
    chinook_depth_features_3 c 
    inner join chlorophyll_phenology_pca h 
        on h.year = extract(year from c.time)
        and h.month = extract(month from c.time)
        and h.h3_index = c.h3_index