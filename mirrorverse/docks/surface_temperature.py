"""
For Loading Surface Temperature Data
"""

import os
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

import ee
from tqdm import tqdm
import pandas as pd
import numpy as np


def pull_surface_temperature_data(input_path, output_path):
    """
    Inputs:
    - input_path(str): path to params file
    - output_path(str): path to save output csv

    Params should have a start and end date in the format "YYYY-MM-DD"
    and a Region of Interest (ROI) in the format [lon_min, lat_min, lon_max, lat_max].

    This function will pull surface temperature data from
    the NOAA/CDR/SST_PATHFINDER/V53 dataset and save it to a csv.
    """
    ee.Authenticate()
    ee.Initialize(project=os.environ["EE_PROJECT"])

    with open(input_path, "r") as f:
        params = json.load(f)

    start = datetime.strptime(params["start"], "%Y-%m-%d")
    end = datetime.strptime(params["end"], "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += relativedelta(months=1)

    roi = ee.Geometry.BBox(*params["roi"])

    dfs = []
    for date in tqdm(dates):
        # we want to get at the middle of the month
        window_start = date + relativedelta(days=12)
        window_end = date + relativedelta(days=18)
        dataset = (
            ee.ImageCollection("NOAA/CDR/SST_PATHFINDER/V53")
            .filterDate(
                window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
            )
            .select(["sea_surface_temperature"])
        )
        pixel_info = dataset.getRegion(roi, 50000).getInfo()
        df = pd.DataFrame(pixel_info[1:], columns=pixel_info[0])
        df = df[~np.isnan(df["sea_surface_temperature"])]
        df["temperature_c"] = 0.01 * (df["sea_surface_temperature"] + 273.15)
        del df["sea_surface_temperature"]
        # average over time
        df = (
            df.groupby(["longitude", "latitude"])
            .agg({"temperature_c": "mean"})
            .reset_index()
        )
        df["date"] = date
        dfs.append(df)

    dataframe = pd.concat(dfs)
    dataframe.to_csv(output_path, index=False)
