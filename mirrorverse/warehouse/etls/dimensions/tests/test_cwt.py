"""
CWT ETLs Tests
"""

import pandas as pd
from pandas.testing import assert_frame_equal

from mirrorverse.warehouse.etls.dimensions.cwt import (
    build_cwt_locations,
    build_cwt_tags,
)
from mirrorverse.warehouse.data.cwt import CWT_LOCATIONS_DATA, CWT_TAGS_DATA


def test_build_cwt_locations():
    """
    Test the build_cwt_locations function
    """

    missing_keys = ["ASK", "BON", "BON3"]
    results = build_cwt_locations(missing_keys, CWT_LOCATIONS_DATA)
    expected = pd.DataFrame(
        [
            {
                "cwt_location_key": "ASK",
                "cwt_location_name": "Astoria, OR",
                "lon": -123.83,
                "lat": 46.19,
                "h3_level_4_key": 595195434008313855,
            },
            {
                "cwt_location_key": "BON",
                "cwt_location_name": "Bonners Ferry, ID",
                "lon": -116.32,
                "lat": 48.69,
                "h3_level_4_key": 594806782417698815,
            },
        ]
    )
    assert_frame_equal(results, expected)


def test_build_cwt_tags():
    """
    Test the build_cwt_tags function
    """

    missing_keys = ["091485", "091488"]
    results = build_cwt_tags(missing_keys, CWT_TAGS_DATA)
    expected = pd.DataFrame(
        [
            {
                "cwt_tag_key": "091485",
                "cwt_release_location_key": "ASK",
                "run": 1,
            },
            {
                "cwt_tag_key": "091488",
                "cwt_release_location_key": "BON3",
                "run": 4,
            },
        ]
    )
    assert_frame_equal(results.reset_index(drop=True), expected)
