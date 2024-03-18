"""
Dates ETLing
"""


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
