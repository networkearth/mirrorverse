CREATE TABLE "haven"."depth_model_features_mk3" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/depth_model_features_mk3/',
  write_compression = 'SNAPPY'
) AS with elevation as (
  select h3_index,
    elevation
  from mean_elevation_by_h3
  where h3_resolution = 4
),
tag_features as (
  select p.tag_key,
    p.h3_index,
    p.sunrise,
    p.sunset,
    d.epoch,
    p.lon,
    p.lat,
    extract(
      HOUR
      from from_unixtime(d.epoch)
    ) as hour,
    extract(
      MONTH
      from from_unixtime(d.epoch)
    ) as month,
    d.selected_depth_class
  from tag_position_features_mk1 as p
    inner join tag_depth_features_mk1 as d on p.tag_key = d.tag_key
    and p.epoch = (d.epoch - d.epoch % (24 * 3600))
    and p.tag_key = d.tag_key
),
joined as (
  select t.tag_key,
    t.selected_depth_class,
    t.epoch,
    t.lon,
    t.lat,
    (
      (
        hour < sunset
        and hour >= sunrise
      )
      or (
        hour >= sunrise
        and sunrise > sunset
      )
      or (
        hour < sunset
        and sunrise > sunset
      )
    ) as daytime,
    case
      when sunrise < sunset then sunset - sunrise else 24 - sunrise + sunset
    end as daytime_length,
    t.hour,
    t.month,
    t.sunrise as sunrise,
    t.sunset as sunset,
    e.elevation
  from tag_features as t
    inner join elevation as e on t.h3_index = e.h3_index
),
features as (
  select tag_key,
    epoch,
    lon,
    lat,
    selected_depth_class,
    row_number() over () as _decision,
    case
      when daytime
      and hour >= sunrise then cast((hour - sunrise) as double) / daytime_length
      when daytime
      and hour < sunrise then cast((hour + 24 - sunrise) as double) / daytime_length
      when not daytime
      and hour >= sunset then cast((hour - sunset) as double) / (24 - daytime_length) else cast((hour + 24 - sunset) as double) / (24 - daytime_length)
    end as period_progress,
    daytime,
    elevation,
    month
  from joined
),
expanded_features as (
  select *
  from unnest (
      ARRAY [ 25.0,
      50.0,
      75.0,
      100.0,
      150.0,
      200.0,
      250.0,
      300.0,
      400.0,
      500.0 ]
    ) as t(depth_class)
    cross join (
      select *
      from features
    )
  order by _decision
),
tags as (
  select distinct tag_key
  from tag_position_features_mk1
),
individual_ids as (
  select tag_key,
    row_number() over () as _individual
  from tags
)
select 
    f.tag_key,
    f.epoch,
    f.lon,
    f.lat,
  _individual,
  _decision,
  depth_class = selected_depth_class as _selected,
  row_number() over () as _choice,
  _individual % 3 != 0 as _train,
  (depth_class - (avg(depth_class) over ())) / (stddev(depth_class) over ()) as depth_class,
  avg(depth_class) over () as avg_depth_class,
  stddev(depth_class) over () as std_depth_class,
  (period_progress - (avg(period_progress) over ())) / (stddev(period_progress) over ()) as period_progress,
  avg(period_progress) over () as avg_period_progress,
  stddev(period_progress) over () as std_period_progress,
  case
    when daytime then 1.0 else 0.0
  end as daytime,
  (elevation - (avg(elevation) over ())) / (stddev(elevation) over ()) as elevation,
  avg(elevation) over () as avg_elevation,
  stddev(elevation) over () as std_elevation,
  (month - (avg(month) over ())) / (stddev(month) over ()) as month,
  avg(month) over () as avg_month,
  stddev(month) over () as std_month
from expanded_features f
  inner join individual_ids i on f.tag_key = i.tag_key
order by _decision asc
