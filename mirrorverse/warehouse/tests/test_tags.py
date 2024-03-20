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

from mirrorverse.warehouse.data.tags import (
    TAGS_DATA,
)
from mirrorverse.warehouse.models import Tags
from mirrorverse.warehouse.etls.dimensions.tags import (
    build_tags,
)


class TestTags(unittest.TestCase):

    def setUp(self):
        engine = get_engine(db_url="sqlite://")
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.rollback()
        self.session.close()

    def test_upload_tags(self):
        missing_keys = ["205415", "239204"]
        formatted = build_tags(missing_keys, TAGS_DATA)
        upload_dataframe(self.session, Tags, formatted)
        stmt = select(Tags)
        results = pd.read_sql_query(stmt, self.session.bind)
        expected = pd.DataFrame(
            [
                {
                    "tag_key": "205415",
                    "tag_model": "MiniPAT",
                    "time_resolution_min": 10,
                    "fork_length_cm": 80.0,
                    "deploy_date_key": 1602547200,
                    "deploy_latitude": 55.3,
                    "deploy_longitude": -151.2,
                    "deploy_h3_level_4_key": 594997496145510399,
                    "end_date_key": 1618790400,
                    "end_latitude": 45.7,
                    "end_longitude": -124.6,
                    "end_h3_level_4_key": 595195451188183039,
                },
                {
                    "tag_key": "239204",
                    "tag_model": "MiniPAT",
                    "time_resolution_min": 5,
                    "fork_length_cm": 75.0,
                    "deploy_date_key": 1653955200,
                    "deploy_latitude": 55.6,
                    "deploy_longitude": -134.3,
                    "deploy_h3_level_4_key": 594804342876274687,
                    "end_date_key": 1659225600,
                    "end_latitude": 51.6,
                    "end_longitude": -130.9,
                    "end_h3_level_4_key": 594801946284523519,
                },
            ]
        )
        assert set(results.columns) == set(expected.columns)
        assert_frame_equal(expected, results[expected.columns])
