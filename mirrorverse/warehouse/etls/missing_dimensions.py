"""
Queries to Help Identify Missing Dimensions
"""

import json

import click
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from mirrorverse.warehouse.models import ModelBase, CWTRecoveries
from mirrorverse.warehouse.utils import get_engine

MODEL_KEY = {model.__tablename__: model for model in ModelBase.__subclasses__()}


def get_primary_key(dimenion_model):
    """
    Input:
    - dimenion_model (sqlalchemy.orm.DeclarativeBase): A model class for the
        dimension table you're querying

    Output:
    - primary_key (str): The name of the primary key column
    """

    return list(dimenion_model.__table__.primary_key.columns)[0].name


def get_associated_dimensions(fact_model):
    """
    Input:
    - fact_model (sqlalchemy.orm.DeclarativeBase): A model class for the fact
        table you're querying

    Output:
    - keys (list): column names that are foreign keys to dimension tables
    - dimension_models (list): model classes for the dimension tables
    """

    keys = []
    dimension_models = []
    for column in list(fact_model.__table__.columns):
        if column.foreign_keys:
            keys.append(column.name)
            table_name = list(column.foreign_keys)[0].column.table.name
            dimension_models.append(MODEL_KEY[table_name])
    return keys, dimension_models


def get_missing_dimension_keys(dimension_model, fact_model, key, session):
    """
    Input:
    - dimenion_model (sqlalchemy.orm.DeclarativeBase): A model class for the
        dimension table you're querying
    - fact_model (sqlalchemy.orm.DeclarativeBase): A model class for the fact
        table you're querying
    - key (str): The name of the foreign key on the fact table
    - session (sqlalchemy.orm.Session): A session object

    Output:
    - missing_keys (dict): single key (the dimension key)
        with a list of the key values that are in the fact table
        but not in the dimension table
    """
    dimension_key = get_primary_key(dimension_model)

    # pylint: disable=singleton-comparison
    query = (
        select(fact_model.__dict__[key])
        .join(dimension_model, isouter=True)
        .where(dimension_model.__dict__[dimension_key] == None)
        .distinct()
    )
    results = pd.read_sql_query(query, session.bind)
    return {dimension_key: results[key].tolist()}


@click.command()
@click.option("--table", "-t", help="The table to upload to", required=True)
@click.option("--output_path", "-o", help="Path to the output data", required=True)
def enumerate_missing_dimensions(table, output_path):
    """
    Enumerate the missing dimensions for a given fact table.
    """
    fact_model = {
        "cwt_recoveries": CWTRecoveries,
    }[table]

    keys, dimension_models = get_associated_dimensions(fact_model)
    session = Session(get_engine())
    missing_keys = {}
    for key, dimension_model in zip(keys, dimension_models):
        missing_keys.update(
            get_missing_dimension_keys(dimension_model, fact_model, key, session)
        )

    # pylint: disable=unspecified-encoding
    with open(output_path, "w") as fh:
        json.dump(missing_keys, fh, indent=2, sort_keys=True)
