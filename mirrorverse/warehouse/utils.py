"""
Utility functions for the mirrorverse warehouse.
"""

import os

from sqlalchemy import create_engine, insert

from mirrorverse.warehouse.models import ModelBase


def get_engine(db_url=None):
    """
    Simply returns a sqlite engine.
    """
    if not db_url:
        db_url = os.environ.get("DATABASE_URL", "sqlite://")
    return create_engine(db_url, echo=True)


def upload_dataframe(session, model, dataframe):
    """
    Input:
    - session (sqlalchemy.orm.Session): A session object
    - model (sqlalchemy.orm.DeclarativeBase): A model class for the table
        you're inserting to
    - dataframe (pd.DataFrame): The data you want to insert
    """
    ModelBase.metadata.create_all(session.bind)
    session.execute(insert(model), dataframe.to_dict(orient="records"))
    session.commit()
