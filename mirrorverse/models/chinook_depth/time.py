"""
Time Features for the Chinook Depth Model
"""

import pandas as pd
import numpy as np
from suntimes import SunTimes


def get_sunrise(lat, lon, date):
    """
    Inputs:
    - lat: float, latitude
    - lon: float, longitude
    - date: str, date

    Outputs:
    - int, hour of sunrise
    """
    return SunTimes(longitude=lon, latitude=lat, altitude=0).risewhere(date, "UTC").hour


def get_sunset(lat, lon, date):
    """
    Inputs:
    - lat: float, latitude
    - lon: float, longitude
    - date: str, date

    Outputs:
    - int, hour of sunset
    """
    return SunTimes(longitude=lon, latitude=lat, altitude=0).setwhere(date, "UTC").hour


def add_time_features(input_file, output_file):
    """
    Inputs:
    - input_file: str, path to the input file
    - output_file: str, path to save the output file

    Adds time features to the data and saves it to a csv file.
    """
    data = pd.read_csv(input_file)
    data["datetime"] = pd.to_datetime(data["epoch"], utc=True, unit="s")
    data["date"] = data["datetime"].dt.date
    data = data[np.abs(data["longitude"]) <= 180]
    data["sunrise"] = data.apply(
        lambda r: get_sunrise(r["latitude"], r["longitude"], r["date"]), axis=1
    )
    data["sunset"] = data.apply(
        lambda r: get_sunset(r["latitude"], r["longitude"], r["date"]), axis=1
    )

    data["hour"] = data["datetime"].dt.hour

    data.loc[data["sunrise"] > data["sunset"], "daytime"] = (
        data["hour"] < data["sunset"]
    ) | (data["hour"] >= data["sunrise"])
    data.loc[data["sunrise"] < data["sunset"], "daytime"] = (
        data["hour"] >= data["sunrise"]
    ) & (data["hour"] < data["sunset"])

    data["month"] = data["datetime"].dt.month
    data["daytime"] = data["daytime"].astype(float)

    data["hours_to_transition"] = (
        (
            (data["hour"] > data["sunrise"]) * (24 - data["hour"] + data["sunrise"])
            + (data["hour"] <= data["sunrise"]) * (data["sunrise"] - data["hour"])
        )
        * (1 - data["daytime"])
    ).astype(float) + (
        (
            (data["hour"] > data["sunset"]) * (24 - data["hour"] + data["sunset"])
            + (data["hour"] <= data["sunset"]) * (data["sunset"] - data["hour"])
        )
        * (data["daytime"])
    ).astype(
        float
    )

    data["interval"] = (1 - data["daytime"]) * (
        (data["sunrise"] >= data["sunset"]) * (data["sunrise"] - data["sunset"])
        + (data["sunrise"] < data["sunset"]) * (24 - data["sunset"] + data["sunrise"])
    ) + (data["daytime"]) * (
        (data["sunset"] >= data["sunrise"]) * (data["sunset"] - data["sunrise"])
        + (data["sunset"] < data["sunrise"]) * (24 - data["sunrise"] + data["sunset"])
    )

    data["period_progress"] = (
        1 - data["hours_to_transition"] / data["interval"]
    ).astype(float)

    data.to_csv(output_file, index=False)
