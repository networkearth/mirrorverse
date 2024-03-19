"""
Click Commands
"""

import json
from time import time

import click
import pandas as pd
from sqlalchemy.orm import Session

from mirrorverse.warehouse.etls.facts.cwt import format_cwt_recoveries_data
from mirrorverse.warehouse.models import CWTRecoveries
from mirrorverse.warehouse.utils import upload_dataframe, get_engine


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


@click.command()
@click.option("--table", "-t", help="The table to upload to", required=True)
@click.option("--file_path", "-f", help="Path to the formatted data", required=True)
@click.option("--output_path", "-o", help="Path to the output data", required=True)
def upload_data(table, file_path, output_path):
    """
    Upload the formatted data to the warehouse.
    """
    dataframe = pd.read_csv(file_path)
    model = {
        "cwt_recoveries": CWTRecoveries,
    }[table]
    session = Session(get_engine())
    upload_dataframe(session, model, dataframe)
    session.close()

    status = {
        "status": "success",
        "timestamp": time(),
    }
    # pylint: disable=unspecified-encoding
    with open(output_path, "w") as fh:
        json.dump(status, fh)
