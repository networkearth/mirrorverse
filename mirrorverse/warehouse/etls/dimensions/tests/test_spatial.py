"""
Spatial ETL Tests
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, duplicate-code

import h3
import pandas as pd

from mirrorverse.warehouse.etls.dimensions.spatial import (
    spatial_index_to_key,
    spatial_key_to_index,
    add_spatial_keys_to_facts,
)


def test_spatial_key_funcs():
    h3_index = h3.geo_to_h3(0, 0, 2)
    spatial_key = spatial_index_to_key(h3_index)
    assert isinstance(spatial_key, int)
    assert spatial_key_to_index(spatial_key) == h3_index


def test_add_spatial_keys_to_facts():
    dataframe = pd.DataFrame(
        [
            {"lon": 0, "lat": 0, "fact": "a cool fact"},
            {"lon": 0, "lat": 25, "fact": "a less cool fact"},
            {"lon": 25, "lat": 0, "fact": "a boring fact"},
        ]
    )
    add_spatial_keys_to_facts(dataframe)
    assert set(dataframe.columns) == set(
        ["lon", "lat", "fact"] + [f"h3_level_{resolution}_key" for resolution in [4]]
    )
    assert dataframe.shape[0] == 3
