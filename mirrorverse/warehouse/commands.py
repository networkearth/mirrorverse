"""
Click Commands
"""

import json
from time import time

import click
import pandas as pd
from sqlalchemy.orm import Session
from eralchemy2 import render_er

from mirrorverse.warehouse.etls.missing_dimensions import get_primary_key
from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import upload_dataframe, get_engine
from mirrorverse.warehouse.api import FACT_FORMATTERS, DIMENSION_FORMATTERS


MODEL_KEY = {model.__tablename__: model for model in ModelBase.__subclasses__()}


@click.command()
@click.option("--table", "-t", help="The table to upload to", required=True)
@click.option("--file_path", "-f", help="Path to the raw data", required=True)
@click.option("--output_path", "-o", help="Path to the output data", required=True)
def upload_facts(table, file_path, output_path):
    """
    Format and upload the data.
    """
    dataframe = pd.read_csv(file_path)
    formatted = FACT_FORMATTERS[table](dataframe)

    model = MODEL_KEY[table]
    session = Session(get_engine())
    upload_dataframe(session, model, formatted)
    session.close()

    status = {
        "status": "success",
        "timestamp": time(),
    }
    # pylint: disable=unspecified-encoding
    with open(output_path, "w") as fh:
        json.dump(status, fh)


@click.command()
@click.option("--table", "-t", help="The table to upload to", required=True)
@click.option(
    "--missing_dimensions_path",
    "-m",
    help="Path to the missing dimensions data",
    required=True,
)
@click.option("--file_path", "-f", help="Path to the raw data", required=False)
@click.option("--output_path", "-o", help="Path to the output data", required=True)
def upload_dimensions(table, missing_dimensions_path, file_path, output_path):
    """
    Build the missing dimensions for a given fact table.
    """
    model = MODEL_KEY[table]
    primary_key = get_primary_key(model)

    # pylint: disable=unspecified-encoding
    with open(missing_dimensions_path, "r") as fh:
        missing_dimensions = json.load(fh)

    missing_keys = missing_dimensions[primary_key]
    build_func = DIMENSION_FORMATTERS[table]

    if file_path:
        dataframe = pd.read_csv(file_path)
        formatted = build_func(missing_keys, dataframe)
    else:
        formatted = build_func(missing_keys)

    if formatted.shape[0] > 0:
        session = Session(get_engine())
        upload_dataframe(session, model, formatted)
        session.close()
    else:
        print("Nothing to upload...")

    status = {
        "status": "success",
        "timestamp": time(),
    }
    # pylint: disable=unspecified-encoding
    with open(output_path, "w") as fh:
        json.dump(status, fh)


@click.command()
@click.option("--output_path", "-o", help="Path to the output data", required=True)
def build_erd(output_path):
    """
    Build the ERD for the warehouse.
    """
    render_er(ModelBase.metadata, output_path)
