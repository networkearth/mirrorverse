CREATE TABLE "haven"."chinook_depth_features_3" WITH (
  format = 'parquet',
  external_location = 's3://haven-database/chinook_depth_features_3/',
  write_compression = 'SNAPPY',
  partitioned_by = array['_train']
) as with features as (
    select 
        _individual,
        _decision,
        _choice,
        _selected,
        tag_key,
        h3_index,
        time,
        depth_bin,

        ------------------

        case 
            when depth_bin = 25
                then 0.1
            when depth_bin = 50
                then 0.2
            when depth_bin = 75
                then 0.3
            when depth_bin = 100
                then 0.4
            when depth_bin = 150
                then 0.5
            when depth_bin = 200
                then 0.6
            when depth_bin = 250
                then 0.7
            when depth_bin = 300
                then 0.8
            when depth_bin = 400
                then 0.9
            else 1.0
        end as n_depth_bin,
        
        ------------------
        
        cos_moon,
        sin_moon,
        cos_orbit,
        sin_orbit,
        cos_sun,
        sin_sun,
        
        ------------------
        
        chlorophyll,
        net_primary_production,
        nitrate,
        oxygen,
        phosphate,
        silicate,
        
        ------------------
        
        (ln(chlorophyll + 0.001) - (-5.504930988474634)) / (1.6989573938383091 - (-5.504930988474634)) as n_chlorophyll,
        (ln(net_primary_production + 0.001) - (-6.907755278982137)) / (4.727879629770012 - (-6.907755278982137)) as n_net_primary_production,
        (nitrate - 0.0032857046462595463) / (40.51780700683594 - 0.0032857046462595463) as n_nitrate,
        (oxygen - 9.647261619567871) / (381.22198486328125 - 9.647261619567871) as n_oxygen,
        (phosphate - 0.2412072867155075) / (3.0300753116607666 - 0.2412072867155075) as n_phosphate,
        (silicate - 1.902060866355896) / (104.22795867919922 - 1.902060866355896) as n_silicate,
        
        ------------------
        
        elevation,
        mixed_layer_thickness,
        salinity,
        temperature,
        
        ------------------
        
        case 
            when elevation >= 0
                then 0
            else -elevation / 4983.178342862019
        end as n_elevation,
        (ln(mixed_layer_thickness + 0.001) - 2.184872639789435) / (4.879886400631615 - 2.184872639789435) as n_mixed_layer_thickness,
        (salinity - 22.382081968826242) / (34.35277867829427 - 22.382081968826242) as n_salinity,
        (temperature - (-0.37473675794899464)) / (15.986583786194142 - (-0.37473675794899464)) as n_temperature,
        
        ------------------
        
        velocity_east,
        velocity_north,
        
        ------------------
        
        _train
        
    from 
        chinook_depth_raw_choices
), num_choices as (
    select 
        _individual,
        _decision,
        count(*) as num_choices 
    from 
        features
    group by 
        1, 2
)
select 
    f.*
from 
    features f 
    inner join num_choices n 
        on n._individual = f._individual 
        and n._decision = f._decision
        and n.num_choices > 1