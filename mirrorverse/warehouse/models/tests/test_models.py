from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine


def test_database_set():
    engine = get_engine()
    ModelBase.metadata.create_all(engine)
