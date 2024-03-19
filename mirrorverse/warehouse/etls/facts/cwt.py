"""
ETLs for CWT Fact Data
"""

import pandas as pd

from mirrorverse.warehouse.etls.dimensions.dates import add_date_keys_to_facts
from mirrorverse.warehouse.models.facts import CWTRecoveries


def format_cwt_recoveries_data(raw_data):
    """
    Input:
    - raw_data (pd.DataFrame): Raw data from the CWT Website
    """

    # some basic renaming
    dataframe = raw_data.rename(
        {
            "recovery_location_code": "cwt_recovery_location_key",
            "species": "species_key",
            "reporting_agency": "cwt_reporting_agency_key",
            "number_cwt_estimated": "number_estimated",
            "tag_code": "cwt_tag_key",
        },
        axis=1,
    )

    # deal with their very weird date column
    dataframe["valid_date"] = dataframe["recovery_date"].apply(lambda x: x > 10000000)
    dataframe = dataframe[dataframe["valid_date"]]
    dataframe["recovery_date"] = pd.to_datetime(
        dataframe["recovery_date"].astype(str), format="%Y%m%d"
    )
    dataframe = add_date_keys_to_facts(dataframe, "recovery_date", "recovery_date_key")

    # filter down to the columns in our table
    columns = [column.key for column in CWTRecoveries.__table__.columns]
    return dataframe[columns]
