"""
Click Commands
"""

import click
import pandas as pd

from mirrorverse.warehouse.etls.facts.cwt import format_cwt_recoveries_data


@click.command()
@click.option("--table", "-t", help="The table to upload to", required=True)
@click.option("--file_path", "-f", help="Path to the raw data", required=True)
@click.option("--output_path", "-o", help="Path to the output data", required=True)
def format_data(table, file_path, output_path):
    """
    Format the raw data into a format that can be uploaded to the warehouse.
    """
    dataframe = pd.read_csv(file_path)
    formatted = {
        "cwt_recoveries": format_cwt_recoveries_data,
    }[
        table
    ](dataframe)
    formatted.to_csv(output_path, index=False)
