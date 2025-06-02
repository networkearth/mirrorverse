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

    print(f'Getting Shapes...')
    space_shape = dataset['lat_rho'][:].shape
    time_shape = dataset['ocean_time'][:].shape[0]
    depth_shape = dataset['depth_bnds'][:].shape[0]

    print(f'Creating DataFrame...')
    latitude = np.tile(dataset['lat_rho'][:], (time_shape, depth_shape, 1, 1))
    longitude = np.tile(dataset['lon_rho'][:], (time_shape, depth_shape, 1, 1))
    epoch = np.tile(dataset['ocean_time'][:].reshape(time_shape, 1, 1, 1), (1, depth_shape, *space_shape))
    depth = np.tile(dataset['depth_bnds'][:][:,0][:].reshape(1, depth_shape, 1, 1), (time_shape, 1, *space_shape))
    zeta = np.tile(dataset['zeta'][:].reshape(time_shape, 1, *space_shape), (1, depth_shape, 1, 1))

    data = pd.DataFrame({
        'latitude': latitude.flatten(), 
        'longitude': longitude.flatten(),
        'zeta': zeta.flatten(),
        'epoch': epoch.flatten(),
        'depth': depth.flatten(),
        'NO3': dataset['NO3'][:].flatten(),
        'SiOH4': dataset['SiOH4'][:].flatten(),
        'NH4': dataset['NH4'][:].flatten(),
        'nanophytoplankton': dataset['nanophytoplankton'][:].flatten(),
        'diatom': dataset['diatom'][:].flatten(),
        'microzoo1': dataset['microzoo1'][:].flatten(),
        'microzoo2': dataset['microzoo2'][:].flatten(),
        'mesozoo1': dataset['mesozoo1'][:].flatten(),
        'mesozoo2': dataset['mesozoo2'][:].flatten(),
        'mesozoo3': dataset['mesozoo3'][:].flatten(),
        'Pzooplankton': dataset['Pzooplankton'][:].flatten(),
        'PON': dataset['PON'][:].flatten(),
        'DON': dataset['DON'][:].flatten(),
        'opal': dataset['opal'][:].flatten(),
        'FeD': dataset['FeD'][:].flatten(),
        'FeL': dataset['FeL'][:].flatten(),
        'temp': dataset['temp'][:].flatten(),
        'salt': dataset['salt'][:].flatten(),
    }).dropna()

    data.rename(columns={col: col.lower() for col in data.columns}, inplace=True)

    print(f'Formatting Time...')
    data['epoch'] = pd.to_datetime(data['epoch'], unit='s', origin='1900-01-01')
    data['year'] = data['epoch'].dt.year
    data['month'] = data['epoch'].dt.month

    print(f'Determining H3 Indices...')
    h3_df = data[['latitude', 'longitude']].drop_duplicates()
    h3_df['h3_index'] = h3_df.apply(
        lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], resolution), axis=1
    )
    h3_df['resolution'] = resolution
    data = data.merge(h3_df, how='inner')

    print(f'Grouping by H3 Index...')
    data = data.groupby(['epoch', 'depth', 'h3_index']).mean().reset_index()
    data = data.drop(columns=['latitude', 'longitude'])

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