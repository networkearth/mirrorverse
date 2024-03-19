"""
Test Dates ETLs
"""

# pylint: disable=missing-function-docstring

from datetime import datetime

import pandas as pd

from mirrorverse.warehouse.etls.dimensions.dates import (
    add_date_keys_to_facts,
    build_dates,
)


def test_date_to_key():
    dataframe = pd.DataFrame(
        [
            {"my_date": datetime(1970, 1, 1, 0, 0, 0), "fact": "a cool fact"},
            {"my_date": datetime(1970, 1, 1, 3, 0, 0), "fact": "a less cool fact"},
            {"my_date": datetime(1970, 1, 2, 3, 0, 0), "fact": "a boring fact"},
        ]
    )
    add_date_keys_to_facts(dataframe, "my_date", "date_key")
    assert set(dataframe.columns) == set(["my_date", "fact", "date_key"])
    assert dataframe.shape[0] == 3
    assert dataframe["date_key"].dtype == int
    assert (dataframe["date_key"].values == [0, 0, 24 * 3600]).all()


def test_build_dates():
    missing_keys = [0, 24 * 3600]
    results = build_dates(missing_keys)
    assert results.shape[0] == 2
    assert set(results["year"]) == set([1970])
    assert set(results["month"]) == set([1])
    assert set(results["day"]) == set([1, 2])
