"""
Tags Dimensions
"""

import pandas as pd

from mirrorverse.warehouse.etls.dimensions.spatial import add_spatial_keys_to_facts
from mirrorverse.warehouse.etls.dimensions.dates import add_date_keys_to_facts
from mirrorverse.warehouse.models import Tags


def build_tags(missing_keys, raw_data):
    """
    Input:
    - missing_keys (list): A list of missing tag keys
    - raw_data (pd.DataFrame): Raw PSAT data

    Returns a pd.DataFrame
    """

    dataframe = raw_data.rename(
        {
            "Ptt": "tag_key",
            "tag.model": "tag_model",
            "time.series.resolution.min": "time_resolution_min",
            "fork.length.cm": "fork_length_cm",
            "deploy.latitude": "deploy_latitude",
            "deploy.longitude": "deploy_longitude",
            "End.Latitude": "end_latitude",
            "End.Longitude": "end_longitude",
            "deploy.date.GMT": "deploy_date",
            "end.date.time.GMT": "end_date",
        },
        axis=1,
    )
    dataframe["tag_key"] = dataframe["tag_key"].astype(str)
    dataframe = dataframe[dataframe["tag_key"].isin(missing_keys)]

    dataframe = add_spatial_keys_to_facts(
        dataframe,
        lon_col="deploy_longitude",
        lat_col="deploy_latitude",
        prefix="deploy_",
    )
    dataframe = add_spatial_keys_to_facts(
        dataframe,
        lon_col="end_longitude",
        lat_col="end_latitude",
        prefix="end_",
    )

    dataframe["deploy_date"] = pd.to_datetime(dataframe["deploy_date"])
    dataframe["end_date"] = pd.to_datetime(dataframe["end_date"])
    dataframe = add_date_keys_to_facts(dataframe, "deploy_date", "deploy_date_key")
    dataframe = add_date_keys_to_facts(dataframe, "end_date", "end_date_key")

    columns = [column.key for column in Tags.__table__.columns]
    return dataframe[columns]
