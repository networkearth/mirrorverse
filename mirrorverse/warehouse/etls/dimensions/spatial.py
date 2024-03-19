"""
Spatial Dimension ETLs
"""

import h3


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
