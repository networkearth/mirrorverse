"""
Tests of Integrated Surface Temperature Data
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, duplicate-code

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from sqlalchemy import select
from sqlalchemy.orm import Session

from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine, upload_dataframe

from mirrorverse.warehouse.data.surface_temperature import SURFACE_TEMPERATURE_DATA
from mirrorverse.warehouse.models import SurfaceTemperature
from mirrorverse.warehouse.etls.facts.surface_temperature import (
    format_surface_temperature,
)


class TestSurfaceTemperature(unittest.TestCase):

    def setUp(self):
        engine = get_engine(db_url="sqlite://")
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.rollback()
        self.session.close()

    def test_upload_elevation(self):
        formatted = format_surface_temperature(SURFACE_TEMPERATURE_DATA)
        upload_dataframe(self.session, SurfaceTemperature, formatted)
        stmt = select(SurfaceTemperature)
        results = pd.read_sql_query(stmt, self.session.bind)
        expected = pd.DataFrame(
            [
                {
                    "h3_level_4_key": 594804239797059583,
                    "temperature_c": 5.0,
                    "date_key": 1609459200,
                },
                {
                    "h3_level_4_key": 594804239797059583,
                    "temperature_c": 10.0,
                    "date_key": 1612137600,
                },
                {
                    "h3_level_4_key": 594992694372073471,
                    "temperature_c": 7.5,
                    "date_key": 1609459200,
                },
            ]
        )

        assert set(results.columns) == set(expected.columns)
        assert_frame_equal(expected, results[expected.columns])
