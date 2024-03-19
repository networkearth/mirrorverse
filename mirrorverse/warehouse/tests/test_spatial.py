"""
Tests of Integrated Dates Data
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, duplicate-code

import unittest

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine, upload_dataframe

from mirrorverse.warehouse.models import H3Level4
from mirrorverse.warehouse.etls.dimensions.spatial import build_spatial


class TestDates(unittest.TestCase):

    def setUp(self):
        engine = get_engine(db_url="sqlite://")
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.rollback()
        self.session.close()

    def test_upload(self):
        formatted = build_spatial(
            4,
            [
                594804128127909887,
                595193999489236991,
            ],
        )
        upload_dataframe(self.session, H3Level4, formatted)
        stmt = select(H3Level4)
        results = pd.read_sql_query(stmt, self.session.bind)
        assert results.shape[0] == 2
        assert set(results.columns) == set(["h3_level_4_key", "geometry"])
        assert results["h3_level_4_key"].dtype == int
        assert results["geometry"].dtype == object
