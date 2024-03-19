"""
Dates ETLing
"""

import pandas as pd

from mirrorverse.warehouse.models import Dates


def add_date_keys_to_facts(dataframe, date_column, date_key_column):
    """
    Input:
    - dataframe (pd.DataFrame): The dataframe to transform
    - date_column (str): The name of the date column to transform
    - date_key_column (str): The name of the date key column to create

    Returns a pd.DataFrame
    """

    dataframe[date_key_column] = (
        dataframe[date_column].astype("datetime64[s]").astype(int)
    )
    dataframe[date_key_column] = (
        dataframe[date_key_column] - dataframe[date_key_column] % 86400
    )
    return dataframe


def build_dates(missing_keys):
    """
    Input:
    - missing_keys: A list of missing epochs

    Returns a pd.DataFrame
    """

    dataframe = pd.DataFrame(missing_keys, columns=["date_key"])
    dataframe["date"] = pd.to_datetime(dataframe["date_key"], unit="s")
    dataframe["year"] = dataframe["date"].dt.year
    dataframe["month"] = dataframe["date"].dt.month
    dataframe["day"] = dataframe["date"].dt.day

    columns = [column.key for column in Dates.__table__.columns]
    return dataframe[columns]
