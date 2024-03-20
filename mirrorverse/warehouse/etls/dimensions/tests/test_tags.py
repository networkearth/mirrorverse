"""
Tests for Tags ETLs
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, duplicate-code

import pandas as pd
from pandas.testing import assert_frame_equal

from mirrorverse.warehouse.etls.dimensions.tags import build_tags
from mirrorverse.warehouse.data.tags import TAGS_DATA


def test_build_tags():
    missing_keys = ["205415", "239204"]
    results = build_tags(missing_keys, TAGS_DATA)
    expected = pd.DataFrame(
        [
            {
                "tag_key": "205415",
                "tag_model": "MiniPAT",
                "time_resolution_min": 10.0,
                "fork_length_cm": 80.0,
                "deploy_date_key": 1602547200,
                "deploy_latitude": 55.3,
                "deploy_longitude": -151.2,
                "deploy_h3_level_4_key": 594997496145510399,
                "end_date_key": 1618790400,
                "end_latitude": 45.7,
                "end_longitude": -124.6,
                "end_h3_level_4_key": 595195451188183039,
            },
            {
                "tag_key": "239204",
                "tag_model": "MiniPAT",
                "time_resolution_min": 5.0,
                "fork_length_cm": 75.0,
                "deploy_date_key": 1653955200,
                "deploy_latitude": 55.6,
                "deploy_longitude": -134.3,
                "deploy_h3_level_4_key": 594804342876274687,
                "end_date_key": 1659225600,
                "end_latitude": 51.6,
                "end_longitude": -130.9,
                "end_h3_level_4_key": 594801946284523519,
            },
        ]
    )
    assert set(results.columns) == set(expected.columns)
    assert_frame_equal(results[expected.columns].reset_index(drop=True), expected)
