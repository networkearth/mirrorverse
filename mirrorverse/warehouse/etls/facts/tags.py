"""
ETLs for Tags Fact Data
"""

import pandas as pd

from mirrorverse.warehouse.models import TagTracks, TagDepths
from mirrorverse.warehouse.etls.dimensions.dates import add_date_keys_to_facts
from mirrorverse.warehouse.etls.dimensions.spatial import add_spatial_keys_to_facts


def format_tag_tracks(raw_data):
    """
    Input:
    - raw_data (pd.DataFrame): Raw PSAT data

    Returns a pd.DataFrame
    """
    dataframe = raw_data.rename(
        {
            "Ptt": "tag_key",
            "Date": "date",
            "Most.Likely.Longitude": "longitude",
            "Most.Likely.Latitude": "latitude",
        },
        axis=1,
    )
    dataframe["tag_key"] = dataframe["tag_key"].astype(str)
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe = add_date_keys_to_facts(dataframe, "date", "date_key")

    dataframe = add_spatial_keys_to_facts(dataframe)

    columns = [column.key for column in TagTracks.__table__.columns]
    return dataframe[columns]


def format_home_regions(raw_data):
    """
    Input:
    - raw_data (pd.DataFrame): Raw mappings to home regions

    Returns a pd.DataFrame
    """
    dataframe = raw_data.rename(
        {
            "Ptt": "tag_key",
            "MBC_Broadscaleregion": "home_region",
        },
        axis=1,
    )
    dataframe["tag_key"] = dataframe["tag_key"].astype(str)
    dataframe = dataframe[~dataframe["home_region"].isnull()]
    return dataframe


def format_tag_depths(raw_data):
    """
    Input:
    - raw_data (pd.DataFrame): Raw PSAT data

    Returns a pd.DataFrame
    """
    dataframe = raw_data.rename(
        {
            "Ptt": "tag_key",
            "date.time.GMT": "date",
            "depth.m": "depth",
            "temp.c": "temperature",
        },
        axis=1,
    )
    dataframe["tag_key"] = dataframe["tag_key"].astype(str)
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["epoch"] = dataframe["date"].astype(int) // 10**9
    dataframe = add_date_keys_to_facts(dataframe, "date", "date_key")

    columns = [column.key for column in TagDepths.__table__.columns]
    return dataframe[columns]
