"""
Queries to Help Identify Missing Dimensions
"""

import pandas as pd
from sqlalchemy import select

from mirrorverse.warehouse.models import ModelBase

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
            table_name = dimension_models.append(
                list(column.foreign_keys)[0].column.table.name
            )
            dimension_models.append(MODEL_KEY[table_name])
    return keys, dimension_models


def get_dimension_keys(model):
    """
    Input:
    - model (sqlalchemy.orm.DeclarativeBase): A model class for the fact
        table you're querying

    Output:
    - keys (list): pairs of (column_name, dimension_table_key)
        that can then be used to join to the dimension tables
    """

    keys = []
    for column in list(model.__table__.columns):
        if column.foreign_keys:
            keys.append((column.name, list(column.foreign_keys)[0].column.name))


def get_missing_dimension_keys(dimension_model, fact_model, key, engine):
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
    query = (
        select(fact_model.__dict__[key])
        .join(dimension_model, isouter=True)
        .where(dimension_model.__dict__[dimension_key] == None)
        .distinct()
    )
    results = pd.read_sql_query(query, engine)
    return {dimension_key: results[key].tolist()}
