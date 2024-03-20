"""
Tests of Integrated CWT Data
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, duplicate-code

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from sqlalchemy import select
from sqlalchemy.orm import Session

from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine, upload_dataframe

from mirrorverse.warehouse.data.elevation import ELEVATION_DATA
from mirrorverse.warehouse.models import Elevation
from mirrorverse.warehouse.etls.facts.elevation import format_elevation


class TestElevation(unittest.TestCase):

    def setUp(self):
        engine = get_engine(db_url="sqlite://")
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.rollback()
        self.session.close()

    def test_upload_elevation(self):
        formatted = format_elevation(ELEVATION_DATA)
        upload_dataframe(self.session, Elevation, formatted)
        stmt = select(Elevation)
        results = pd.read_sql_query(stmt, self.session.bind)
        expected = pd.DataFrame(
            [
                {
                    "h3_level_4_key": 594804239797059583,
                    "elevation": 100.0,
                },
                {
                    "h3_level_4_key": 594992694372073471,
                    "elevation": 150.0,
                },
            ]
        )

        assert set(results.columns) == set(expected.columns)
        assert_frame_equal(expected, results[expected.columns])
