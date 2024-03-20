"""
Elevation ETLs
"""

from mirrorverse.warehouse.etls.dimensions.spatial import add_spatial_keys_to_facts
from mirrorverse.warehouse.models import Elevation


def format_elevation(raw_data):
    """
    Inputs:
    - raw_data (pd.DataFrame): raw elevation data

    Returns a formatted DataFrame
    """

    dataframe = raw_data.copy()
    dataframe = add_spatial_keys_to_facts(
        dataframe, lat_col="latitude", lon_col="longitude"
    )
    dataframe = dataframe.groupby("h3_level_4_key").mean().reset_index()

    columns = [column.key for column in Elevation.__table__.columns]
    return dataframe[columns]
