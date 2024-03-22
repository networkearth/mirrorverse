"""
Surface Temperature ETLs
"""

from mirrorverse.warehouse.etls.dimensions.spatial import add_spatial_keys_to_facts
from mirrorverse.warehouse.etls.dimensions.dates import add_date_keys_to_facts
from mirrorverse.warehouse.models import SurfaceTemperature


def format_surface_temperature(raw_data):
    """
    Inputs:
    - raw_data (pd.DataFrame): raw surface temperature data

    Returns a formatted DataFrame
    """

    dataframe = raw_data.copy()
    dataframe = add_spatial_keys_to_facts(
        dataframe, lat_col="latitude", lon_col="longitude"
    )
    dataframe = add_date_keys_to_facts(dataframe, "date", date_key_column="date_key")
    dataframe = (
        dataframe[["h3_level_4_key", "date_key", "temperature_c"]]
        .groupby(["h3_level_4_key", "date_key"])
        .mean()
        .reset_index()
    )

    columns = [column.key for column in SurfaceTemperature.__table__.columns]
    return dataframe[columns]
