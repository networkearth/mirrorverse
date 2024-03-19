"""
Just tests we can build the models
"""

# pylint: disable=missing-function-docstring

from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine


def test_database_set():
    engine = get_engine(db_url="sqlite://")
    ModelBase.metadata.create_all(engine)
