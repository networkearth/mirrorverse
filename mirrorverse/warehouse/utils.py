"""
Utility functions for the mirrorverse warehouse.
"""

from sqlalchemy import create_engine, insert


def get_engine():
    """
    Simply returns a sqlite engine.
    """
    return create_engine("sqlite://")


def upload_dataframe(session, model, dataframe):
    """
    Input:
    - session (sqlalchemy.orm.Session): A session object
    - model (sqlalchemy.orm.DeclarativeBase): A model class for the table
        you're inserting to
    - dataframe (pd.DataFrame): The data you want to insert
    """
    session.execute(insert(model), dataframe.to_dict(orient="records"))
