"""
Tests of Integrated CWT Data
"""

# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sqlalchemy import select
from sqlalchemy.orm import Session

from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine, upload_dataframe

from mirrorverse.warehouse.data.cwt import CWT_RETRIEVAL_DATA
from mirrorverse.warehouse.models import CWTRecoveries
from mirrorverse.warehouse.etls.facts.cwt import format_cwt_recoveries_data


class TestCWTRetrievals(unittest.TestCase):

    def setUp(self):
        engine = get_engine()
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def test_upload(self):
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

    def tearDown(self):
        self.session.rollback()
        self.session.close()
