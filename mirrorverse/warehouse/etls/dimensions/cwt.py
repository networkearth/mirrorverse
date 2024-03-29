"""
CWT Dimensions ETLs
"""

import numpy as np

from mirrorverse.warehouse.etls.dimensions.spatial import add_spatial_keys_to_facts
from mirrorverse.warehouse.models import CWTLocations, CWTTags


def build_cwt_locations(missing_keys, raw_data):
    """
    Input:
    - missing_keys (list): A list of missing location keys
    - raw_data (pd.DataFrame): Raw data from the CWT Website

    Returns a pd.DataFrame
    """

    dataframe = raw_data.rename(
        {
            "location_code": "cwt_location_key",
            "name": "cwt_location_name",
            "rmis_latitude": "latitude",
            "rmis_longitude": "longitude",
        },
        axis=1,
    )

    assert dataframe.shape[0] == dataframe.drop_duplicates("cwt_location_key").shape[0]

    dataframe = dataframe[dataframe["cwt_location_key"].isin(missing_keys)]

    dataframe = dataframe[
        ~np.isnan(dataframe["latitude"]) & ~np.isnan(dataframe["longitude"])
    ]

    dataframe = add_spatial_keys_to_facts(dataframe)

    columns = [column.key for column in CWTLocations.__table__.columns]
    return dataframe[columns]


def build_cwt_tags(missing_keys, raw_data):
    """
    Input:
    - missing_keys (list): A list of missing tag keys
    - raw_data (pd.DataFrame): Raw data from the CWT Website

    Returns a pd.DataFrame
    """

    dataframe = raw_data.rename(
        {
            "tag_code_or_release_id": "cwt_tag_key",
            "release_location_code": "cwt_release_location_key",
        },
        axis=1,
    )

    assert dataframe.shape[0] == dataframe.drop_duplicates("cwt_tag_key").shape[0]

    dataframe = dataframe[dataframe["cwt_tag_key"].isin(missing_keys)]

    columns = [column.key for column in CWTTags.__table__.columns]
    return dataframe[columns]
