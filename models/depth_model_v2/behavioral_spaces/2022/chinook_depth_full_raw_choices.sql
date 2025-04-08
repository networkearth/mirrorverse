CREATE TABLE "haven"."chinook_depth_full_raw_choices_3" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/chinook_depth_full_raw_choices_3/',
  write_compression = 'SNAPPY',
  partitioned_by = array['region']
) as with unexploded_depth_data as (
    select 
        false as _train,
        row_number() over () as _individual,
        row_number() over () as _decision,
        d.*,
        e.elevation,
        array[25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0] as depth_bins,
        format_datetime(date(from_unixtime(d.epoch)), 'yyyy-MM-dd') AS date
    from 
        chinook_depth_time_features d
        inner join mean_elevation_by_h3 e 
            on e.h3_index = d.h3_index
), exploded_depth_data as (
    select 
        _individual,
        _decision,
        epoch,
        time,
        date,
        h3_index,
        cos_sun,
        sin_sun,
        cos_moon,
        sin_moon,
        cos_orbit,
        sin_orbit,
        elevation,
        depth_bin,
        row_number() over (partition by _decision) as _choice,
        _train,
        region
    from 
        unexploded_depth_data 
        cross join unnest(depth_bins) as t(depth_bin)
), joined_features as (
    select 
        b.chlorophyll,
        b.nitrate,
        b.phosphate,
        b.silicate,
        b.oxygen,
        b.net_primary_production,
        p.mixed_layer_thickness,
        p.temperature,
        p.velocity_east,
        p.velocity_north,
        p.salinity,
        e.*
    from 
        exploded_depth_data e
        inner join copernicus_biochemistry b 
            on b.h3_index = e.h3_index 
            and b.depth_bin = e.depth_bin
            and b.date = e.date
        inner join copernicus_physics p 
            on p.h3_index = e.h3_index 
            and p.depth_bin = e.depth_bin
            and p.date = e.date
)
select 
    * 
from 
    joined_features
