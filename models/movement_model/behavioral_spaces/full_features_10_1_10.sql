CREATE TABLE "haven"."movement_model_full_features_10_1_10" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/movement_model_full_features_10_1_10/',
  write_compression = 'SNAPPY',
  partitioned_by = array['_train']
) as with data as (
    select 
        row_number() over (partition by o.h3_index, p.time) as _choice,
        o.h3_index as origin_h3_index,
        o.neighbor as next_h3_index,
        o.h3_index,
        case 
            when o.h3_index = o.neighbor
                then 1.0 
            else 0.0 
        end as stay_put,
        cos(s.season_radians) as cos_season_radians,
        sin(s.season_radians) as sin_season_radians,
        o.toward_coast,
        c.coast_distance,
        sin(movement_heading) as sin_mh,
        cos(movement_heading) as cos_mh,
        ln(p.mixed_layer_thickness + 0.01) / ln(153.96588768064976 + 0.01) as normed_log_mlt,
        (p.salinity - 31.770250889748862) / 0.7063574934846404 as normed_salinity,
        p.time
    from 
        options_w_angles o 
        inner join coastal_info c 
            on o.h3_index = c.h3_index
        inner join movement_headings h 
            on o.h3_index = h.h3_index 
            and o.neighbor = h.neighbor
        inner join copernicus_physics p 
            on p.time between TIMESTAMP '2022-01-01 00:00:00' and TIMESTAMP '2023-01-01 00:00:00'
            and p.region = 'chinook_study'
            and p.h3_index = o.h3_index 
            and p.depth_bin = 25.0
        inner join season_radians s
            on (to_unixtime(p.time) - 12 * 3600) = s.epoch
    where 
        o.distance < 50 
        and o.resolution = 4
), decisions as (
    select distinct
        origin_h3_index,
        time 
    from 
        data  
    order by 
        origin_h3_index, time asc
), meta_data as (
    select
        row_number() over () as _individual,
        row_number() over () as _decision,
        False as _train,
        origin_h3_index,
        time 
    from 
        decisions
)
select 
    m._individual,
    m._decision,
    d.*,
    m._train
from 
    data d 
    inner join meta_data m
        on m.time = d.time 
        and m.origin_h3_index = d.origin_h3_index
order by 
    d.origin_h3_index, d.time asc