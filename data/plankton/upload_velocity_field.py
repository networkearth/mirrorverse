import os
import click 
import netCDF4 as nc
import numpy as np
import pandas as pd
import h3

import haven.db as db

os.environ['AWS_PROFILE'] = 'admin'
os.environ['HAVEN_DATABASE'] = 'haven'

def upload(file, table, resolution):
    print(f'Reading {file}...')
    dataset = nc.Dataset(file)

    print(f'u: Getting Shapes...')
    space_shape = dataset['lat_u'][:].shape
    time_shape = dataset['ocean_time'][:].shape[0]
    depth_shape = dataset['depth_bnds'][:].shape[0]

    print(f'u: Creating DataFrame...')
    latitude = np.tile(dataset['lat_u'][:], (time_shape, depth_shape, 1, 1))
    longitude = np.tile(dataset['lon_u'][:], (time_shape, depth_shape, 1, 1))
    epoch = np.tile(dataset['ocean_time'][:].reshape(time_shape, 1, 1, 1), (1, depth_shape, *space_shape))
    depth = np.tile(dataset['depth_bnds'][:][:,0][:].reshape(1, depth_shape, 1, 1), (time_shape, 1, *space_shape))

    data_u = pd.DataFrame({
        'latitude': latitude.flatten(), 
        'longitude': longitude.flatten(),
        'epoch': epoch.flatten(),
        'depth': depth.flatten(),
        'u': dataset['u'][:].flatten(),
    }).dropna()

    data_u.rename(columns={col: col.lower() for col in data_u.columns}, inplace=True)

    print(f'u: Formatting Time...')
    data_u['epoch'] = pd.to_datetime(data_u['epoch'], unit='s', origin='1900-01-01')
    data_u['year'] = data_u['epoch'].dt.year
    data_u['month'] = data_u['epoch'].dt.month

    print(f'u: Determining H3 Indices...')
    h3_df = data_u[['latitude', 'longitude']].drop_duplicates()
    h3_df['h3_index'] = h3_df.apply(
        lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], resolution), axis=1
    )
    h3_df['resolution'] = resolution
    data_u = data_u.merge(h3_df, how='inner')

    print(f'u: Grouping by H3 Index...')
    data_u = data_u.groupby(['epoch', 'depth', 'h3_index']).mean().reset_index()
    data_u = data_u.drop(columns=['latitude', 'longitude'])

    print(f'v: Getting Shapes...')
    space_shape = dataset['lat_v'][:].shape
    time_shape = dataset['ocean_time'][:].shape[0]
    depth_shape = dataset['depth_bnds'][:].shape[0]

    print(f'v: Creating DataFrame...')
    latitude = np.tile(dataset['lat_v'][:], (time_shape, depth_shape, 1, 1))
    longitude = np.tile(dataset['lon_v'][:], (time_shape, depth_shape, 1, 1))
    epoch = np.tile(dataset['ocean_time'][:].reshape(time_shape, 1, 1, 1), (1, depth_shape, *space_shape))
    depth = np.tile(dataset['depth_bnds'][:][:,0][:].reshape(1, depth_shape, 1, 1), (time_shape, 1, *space_shape))

    data_v = pd.DataFrame({
        'latitude': latitude.flatten(), 
        'longitude': longitude.flatten(),
        'epoch': epoch.flatten(),
        'depth': depth.flatten(),
        'v': dataset['v'][:].flatten(),
    }).dropna()

    data_v.rename(columns={col: col.lower() for col in data_v.columns}, inplace=True)

    print(f'v: Formatting Time...')
    data_v['epoch'] = pd.to_datetime(data_v['epoch'], unit='s', origin='1900-01-01')
    data_v['year'] = data_v['epoch'].dt.year
    data_v['month'] = data_v['epoch'].dt.month

    print(f'v: Determining H3 Indices...')
    h3_df = data_v[['latitude', 'longitude']].drop_duplicates()
    h3_df['h3_index'] = h3_df.apply(
        lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], resolution), axis=1
    )
    h3_df['resolution'] = resolution
    data_v = data_v.merge(h3_df, how='inner')

    print(f'v: Grouping by H3 Index...')
    data_v = data_v.groupby(['epoch', 'depth', 'h3_index']).mean().reset_index()
    data_v = data_v.drop(columns=['latitude', 'longitude'])

    print('Merging u and v...')
    data = data_u.merge(data_v, how='inner')

    print(f'Uploading to {table}...')
    db.write_data(data, table, ['year', 'resolution'])

@click.command()
@click.option('--folder', '-f', type=click.Path(exists=True), help='Path to the folder to upload')
@click.option('--table', '-t', type=str, help='Table name to upload to')
@click.option('--resolution', '-r', type=int, default=6, help='H3 resolution to use')
def main(folder, table, resolution):
    print(f'Uploading files from {folder} to {table}...')
    for file in os.listdir(folder):
        if file.endswith('.nc'):
            upload(os.path.join(folder, file), table, resolution)

if __name__ == '__main__':
    main()