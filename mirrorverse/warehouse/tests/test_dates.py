"""
Tests of Integrated Dates Data
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, duplicate-code

import unittest
from datetime import date

import pandas as pd
from pandas.testing import assert_frame_equal
from sqlalchemy import select
from sqlalchemy.orm import Session

from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine, upload_dataframe

from mirrorverse.warehouse.models import Dates
from mirrorverse.warehouse.etls.dimensions.dates import build_dates


class TestDates(unittest.TestCase):

    def setUp(self):
        engine = get_engine(db_url="sqlite://")
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.rollback()
        self.session.close()

    def test_upload(self):
        formatted = build_dates([0, 24 * 3600])
        upload_dataframe(self.session, Dates, formatted)
        stmt = select(Dates)
        results = pd.read_sql_query(stmt, self.session.bind)
        expected = pd.DataFrame(
            [
                {
                    "date_key": 0,
                    "date": date(1970, 1, 1),
                    "day": 1,
                    "month": 1,
                    "year": 1970,
                },
                {
                    "date_key": 24 * 3600,
                    "date": date(1970, 1, 2),
                    "day": 2,
                    "month": 1,
                    "year": 1970,
                },
            ]
        )
        print(results)
        assert_frame_equal(expected, results)
