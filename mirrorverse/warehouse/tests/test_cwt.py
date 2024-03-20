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

from mirrorverse.warehouse.data.cwt import (
    CWT_RETRIEVAL_DATA,
    CWT_LOCATIONS_DATA,
    CWT_TAGS_DATA,
)
from mirrorverse.warehouse.models import CWTRecoveries, CWTLocations, CWTTags
from mirrorverse.warehouse.etls.facts.cwt import format_cwt_recoveries_data
from mirrorverse.warehouse.etls.dimensions.cwt import (
    build_cwt_locations,
    build_cwt_tags,
)


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
                    "longitude": -123.83,
                    "latitude": 46.19,
                    "h3_level_4_key": 595195434008313855,
                },
                {
                    "cwt_location_key": "BON",
                    "cwt_location_name": "Bonners Ferry, ID",
                    "longitude": -116.32,
                    "latitude": 48.69,
                    "h3_level_4_key": 594806782417698815,
                },
            ]
        )
        assert_frame_equal(expected, results)

    def test_upload_tags(self):
        missing_keys = ["091485", "091488"]
        formatted = build_cwt_tags(missing_keys, CWT_TAGS_DATA)
        upload_dataframe(self.session, CWTTags, formatted)
        stmt = select(CWTTags)
        results = pd.read_sql_query(stmt, self.session.bind)
        expected = pd.DataFrame(
            [
                {
                    "cwt_tag_key": "091485",
                    "cwt_release_location_key": "ASK",
                    "run": 1,
                },
                {
                    "cwt_tag_key": "091488",
                    "cwt_release_location_key": "BON3",
                    "run": 4,
                },
            ]
        )
        assert_frame_equal(expected, results)
