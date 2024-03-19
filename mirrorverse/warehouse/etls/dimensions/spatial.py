"""
Spatial Dimension ETLs
"""

import h3
import geopandas as gpd
from shapely.geometry import Polygon

from mirrorverse.warehouse.models import H3Level4


def spatial_index_to_key(spatial_index):
    """
    Input:
    - spatial_index (str): A hex string representing the h3 index

    Returns the 64 bit int representation
    """
    return int(spatial_index, 16)


def spatial_key_to_index(spatial_key):
    """
    Input:
    - spatial_key (int): A 64 bit int representing the h3 index

    Returns the hex string representation
    """
    return hex(spatial_key)[2:]


def add_spatial_keys_to_facts(dataframe, lon_col="lon", lat_col="lat"):
    """
    Input:
    - dataframe (pd.DataFrame): The dataframe to transform
    - lon_col (str): The name of the longitude column (default: "lon")
    - lat_col (str): The name of the latitude column (default: "lat")

    Returns a dataframe with the h3 keys for the given resolutions
    """

    # pylint: disable=cell-var-from-loop
    for resolution in [4]:
        dataframe[f"h3_level_{resolution}_key"] = dataframe.apply(
            lambda row: spatial_index_to_key(
                h3.geo_to_h3(row[lat_col], row[lon_col], resolution)
            ),
            axis=1,
        )
    return dataframe


def get_coords(h3_index):
    """
    Input:
    - h3_index (str): A hex string representing the h3 index

    Returns a tuple of (lon, lat) coordinates for the h3 cell
        geometry
    """
    coords = h3.h3_to_geo_boundary(h3_index, True)
    coords = tuple((lon, lat) for lon, lat in coords)
    lons = [lon for lon, _ in coords]
    if max(lons) - min(lons) > 180:
        coords = tuple(
            (lon if lon > 0 else 180 + (180 + lon), lat) for lon, lat in coords
        )
    return coords


def build_spatial(resolution, missing_keys):
    """
    Input:
    - resolution (int): The resolution of the h3 index
    - missing_keys (list): A list of missing h3 keys

    Returns a pd.DataFrame
    """

    dataframe = gpd.GeoDataFrame(
        [
            {
                f"h3_level_{resolution}_key": key,
                "geometry": Polygon(get_coords(spatial_key_to_index(key))),
            }
            for key in missing_keys
        ]
    ).to_wkt()

    model = {
        4: H3Level4,
    }[resolution]

    columns = [column.key for column in model.__table__.columns]
    return dataframe[columns]
