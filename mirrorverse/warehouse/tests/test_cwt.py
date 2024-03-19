"""
Tests of Integrated CWT Data
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, duplicate-code

import unittest

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sqlalchemy import select
from sqlalchemy.orm import Session

from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine, upload_dataframe

from mirrorverse.warehouse.data.cwt import CWT_RETRIEVAL_DATA, CWT_LOCATIONS_DATA
from mirrorverse.warehouse.models import CWTRecoveries, CWTLocations
from mirrorverse.warehouse.etls.facts.cwt import format_cwt_recoveries_data
from mirrorverse.warehouse.etls.dimensions.cwt import build_cwt_locations


class TestCWT(unittest.TestCase):

    def setUp(self):
        engine = get_engine(db_url="sqlite://")
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.rollback()
        self.session.close()

    def test_upload_retrievals(self):
        formatted = format_cwt_recoveries_data(CWT_RETRIEVAL_DATA)
        upload_dataframe(self.session, CWTRecoveries, formatted)
        stmt = select(CWTRecoveries)
        results = pd.read_sql_query(stmt, self.session.bind)
        expected = pd.DataFrame(
            [
                {
                    "run_year": 2019,
                    "recovery_id": "C1295149",
                    "cwt_reporting_agency_key": "USFWS",
                    "weight": 10.0,
                    "length": 5.0,
                    "sex": None,
                    "number_estimated": 1.02,
                    "species_key": 1,
                    "cwt_tag_key": "091485",
                    "recovery_date_key": 1609459200,
                    "cwt_recovery_location_key": "ASK",
                },
                {
                    "run_year": 2018,
                    "recovery_id": "C1295149",
                    "cwt_reporting_agency_key": "USFWS",
                    "weight": np.nan,
                    "length": np.nan,
                    "sex": "M",
                    "number_estimated": 1.02,
                    "species_key": 1,
                    "cwt_tag_key": "091485",
                    "recovery_date_key": 1609459200,
                    "cwt_recovery_location_key": "ASK",
                },
            ]
        )
        assert_frame_equal(expected, results)

    def test_upload_locations(self):
        missing_keys = ["ASK", "BON"]
        formatted = build_cwt_locations(missing_keys, CWT_LOCATIONS_DATA)
        upload_dataframe(self.session, CWTLocations, formatted)
        stmt = select(CWTLocations)
        results = pd.read_sql_query(stmt, self.session.bind)
        expected = pd.DataFrame(
            [
                {
                    "cwt_location_key": "ASK",
                    "cwt_location_name": "Astoria, OR",
                    "lon": -123.83,
                    "lat": 46.19,
                    "h3_level_4_key": 595195434008313855,
                },
                {
                    "cwt_location_key": "BON",
                    "cwt_location_name": "Bonners Ferry, ID",
                    "lon": -116.32,
                    "lat": 48.69,
                    "h3_level_4_key": 594806782417698815,
                },
            ]
        )
        assert_frame_equal(expected, results)
