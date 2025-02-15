CREATE TABLE "haven"."chinook_depth_raw_choices" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/chinook_depth_raw_choices/',
  write_compression = 'SNAPPY',
  partitioned_by = array['_train']
) AS with unexploded_depth_data as (
    select 
        d._train,
        d.tag_key,
        d._individual,
        d._decision,
        d.epoch,
        d.time,
        d.depth,
        d.depth_bin as selected_depth_bin,
        array[25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0] as depth_bins,
        format_datetime(date(from_unixtime(d.epoch)), 'yyyy-MM-dd') AS date,
        d.h3_index,
        d.cos_sun,
        d.sin_sun,
        d.cos_moon,
        d.sin_moon,
        d.cos_orbit,
        d.sin_orbit,
        e.elevation 
    from 
        chinook_depth_decisions_w_time d
        inner join mean_elevation_by_h3 e 
            on e.h3_index = d.h3_index
), exploded_depth_data as (
    select 
        tag_key,
        _individual,
        _decision,
        epoch,
        time,
        date,
        depth,
        h3_index,
        cos_sun,
        sin_sun,
        cos_moon,
        sin_moon,
        cos_orbit,
        sin_orbit,
        elevation,
        selected_depth_bin,
        depth_bin,
        row_number() over (partition by _decision) as _choice,
        _train
    from 
        unexploded_depth_data 
        cross join unnest(depth_bins) as t(depth_bin)
), joined_features as (
    select 
        e.selected_depth_bin = e.depth_bin as _selected,
        max(e.selected_depth_bin = e.depth_bin) over (partition by _individual, _decision) as has_selected,
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
where 
    has_selected