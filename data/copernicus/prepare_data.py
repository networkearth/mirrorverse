import os
import json
from multiprocessing import Process, Queue


import click 
import h3
import xarray as xr
import pandas as pd
from tqdm import tqdm

REGIONS = {
    "chinook_study": [
        {
            'lon': slice(-180, -100),
            'lat': slice(30, 75),
        },
        {
            'lon': slice(130, 180),
            'lat': slice(30, 75),
        }
    ]
}

def prepare_data(data_path, h3_mapping, vars_to_drop, col_map, region):
    depth_bins = [0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0]
    h3_resolution = 4
    data = xr.open_dataset(data_path)
    # start by filtering down to the region, depths and variables of interest
    dfs = []
    for slices in REGIONS[region]:
        # filter to the slice in question
        filtered = data.sel(
            longitude=slices['lon'],
            latitude=slices['lat'],
            depth=slice(0, depth_bins[-1])
        ).drop_vars(vars_to_drop)
        # group by depth bins
        filtered = filtered.groupby_bins("depth", depth_bins, labels=depth_bins[1:]).mean()
        filtered = filtered.rename({'depth_bins': 'depth_bin'})
        # convert to a pandas dataframe
        dfs.append(filtered.to_dataframe().reset_index())
    df = pd.concat(dfs)
    # merge the h3 index back into the dataframe
    df = df.merge(h3_mapping)
    df = df.dropna()
    df = df.drop(columns=['latitude', 'longitude'])
    # group by time, h3 index and depth bin
    df = df.groupby(['depth_bin', 'time', 'h3_index']).mean().reset_index()
    # rename columns
    df = df.rename(columns=col_map)
    # add partition columns
    df['h3_resolution'] = h3_resolution
    df['region'] = region
    df['date'] = df['time'].dt.date.values[0].strftime('%Y-%m-%d')
    return df

def get_h3_mapping(data_path, region):
    h3_resolution = 4
    data = xr.open_dataset(data_path)
    # start by filtering down to the region, depths and variables of interest
    dfs = []
    for slices in REGIONS[region]:
        # filter to the slice in question
        filtered = data.sel(
            longitude=slices['lon'],
            latitude=slices['lat'],
            depth=slice(data['depth'].values[0], data['depth'].values[0])
        )
        dfs.append(filtered.to_dataframe().reset_index())
    df = pd.concat(dfs)
    # get the h3 index for each lat/lon pair
    lonlats = df[['latitude', 'longitude']].drop_duplicates()
    lonlats['h3_index'] = lonlats.apply(lambda r: h3.geo_to_h3(r['latitude'], r['longitude'], h3_resolution), axis=1)
    return lonlats

@click.command()
@click.option('--input-directory', type=str, required=True)
@click.option('--output-directory', type=str, required=True)
@click.option('--config', type=str, required=True)
def main(input_directory, output_directory, config):
    with open(config, 'r') as f:
        config = json.load(f)

    file_paths = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.nc'):
                file_paths.append(os.path.join(root, file))

    file_paths = sorted(file_paths)

    h3_mapping = get_h3_mapping(file_paths[0], config['region'])

    for file_path in tqdm(file_paths):
        df = prepare_data(file_path, h3_mapping, config['vars_to_drop'], config['col_map'], config['region'])
        date = df['date'].values[0]
        df.to_parquet(f'{output_directory}/{date}.snappy.parquet', index=False)


if __name__ == '__main__':
    main()