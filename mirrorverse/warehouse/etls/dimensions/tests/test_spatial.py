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
    get_coords,
    build_spatial,
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


def test_get_coords_cross_meridian():
    # this example normally causes problems
    # by crossing the meridian
    h3_index = h3.geo_to_h3(0, -180, 2)
    # normally h3 will return a list of coordinates
    # where one of the longitudes is extremely far away
    # thus ruining our polygons
    # this function is supposed to fix that
    coords = get_coords(h3_index)
    lons = [lon for lon, _ in coords]
    assert max(lons) - min(lons) < 180


def test_get_coords_no_cross_meridian():
    h3_index = h3.geo_to_h3(0, 0, 2)
    coords = get_coords(h3_index)
    assert coords == h3.h3_to_geo_boundary(h3_index, True)


def test_build_spatial():
    missing_keys = [
        594804128127909887,
        595193999489236991,
    ]
    results = build_spatial(4, missing_keys)
    assert results.shape[0] == 2
    assert set(results.columns) == set(["h3_level_4_key", "geometry"])
    assert results["h3_level_4_key"].dtype == int
    assert results["geometry"].dtype == object
