"""
For Loading Elevation Data
"""

import os
import multiprocessing as mp
from collections import defaultdict

import pandas as pd
import numpy as np
import h3
import netCDF4 as nc


# pylint: disable=consider-using-enumerate
def get_grouped_info(args):
    """
    Inputs:
    - args (tuple): (elevation, lats, lons)

    Returns:
    - (totals, counts) (tuple): (dict, dict) where totals is the sum of elevation
        across all data points in the h3 index and counts is the number of data points
        (allows you to calculate the mean elevation for each h3 index)
    """
    elevation, lats, lons = args
    totals = defaultdict(int)
    counts = defaultdict(int)
    for i in range(len(lats)):
        for j in range(len(lons)):
            h3_index = h3.geo_to_h3(lats[i], lons[j], 4)
            totals[h3_index] += elevation[i, j]
            counts[h3_index] += 1
    return totals, counts


def get_split_data(elevation, lats, lons, n):
    """
    Inputs:
    - elevation (array): array of elevation data
    - lats (array): array of latitudes
    - lons (array): array of longitudes
    - n (int): number of splits

    Returns:
    - splits (list): list of tuples of (elevation, lats, lons) for each split
    """
    split_size = int(np.ceil(len(lats) / n))
    splits = [
        (elevation[i : i + split_size], lats[i : i + split_size], lons)
        for i in range(0, len(lats), split_size)
    ]
    return splits


# pylint: disable=no-member
def pull_elevation_data(input_path, output_path):
    """
    Inputs:
    - input_path(str): path to .nc file
    - output_path(str): path to output csv file

    Computes the average elevation for each h3 index
    and writes the results to a csv file
    """

    print("Reading Data...")
    dataset = nc.Dataset(input_path)

    elevation = dataset["elevation"][:]
    lats = dataset["lat"][:]
    lons = dataset["lon"][:]

    print("Grouping Data Within Splits...")
    num_processes = os.cpu_count() - 1
    with mp.Pool(num_processes) as p:
        results = p.map(
            get_grouped_info, get_split_data(elevation, lats, lons, num_processes)
        )

    print("Accumulating Data Across Splits...")
    totals = results[0][0]
    counts = results[0][1]
    for new_totals, new_counts in results[1:]:
        for key in new_totals:
            totals[key] += new_totals[key]
            counts[key] += new_counts[key]

    print("Converting to a DataFrame...")
    dataframe = pd.DataFrame(
        [
            {
                "latitude": h3.h3_to_geo(key)[0],
                "longitude": h3.h3_to_geo(key)[1],
                "elevation": totals[key] / counts[key],
            }
            for key in totals
        ]
    )

    print("Writing Data...")
    dataframe.to_csv(output_path, index=False)
