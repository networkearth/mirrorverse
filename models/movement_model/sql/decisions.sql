CREATE TABLE "haven"."movement_model_decisions" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/movement_model_decisions/',
  write_compression = 'SNAPPY'
) AS with tags as (
    select distinct 
        tag_key
    from 
        tag_position_features_mk1
), individuals as (
    select 
        row_number() over () as _individual,
        tag_key
    from tags
), decisions as (
    select 
        row_number() over () as _decision,
        tag_key,
        epoch,
        h3_index as current_h3_index,
        lead(h3_index, 1) over (partition by tag_key order by epoch asc) as selected_h3_index
    from 
        tag_position_features_mk1
    order by 
        tag_key, epoch asc
)
select 
    i.tag_key,
    i._individual,
    d._decision, 
    d.current_h3_index,
    d.selected_h3_index,
    date_format(from_unixtime(d.epoch), '%Y-%m-%d') as date
from 
    decisions d 
    inner join individuals i 
        on d.tag_key = i.tag_key 
where 
    d.selected_h3_index is not null