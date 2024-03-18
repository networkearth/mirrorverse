"""
Test Dates ETLs
"""

from datetime import datetime

import pandas as pd

from mirrorverse.warehouse.etls.dimensions.dates import add_date_keys_to_facts


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
