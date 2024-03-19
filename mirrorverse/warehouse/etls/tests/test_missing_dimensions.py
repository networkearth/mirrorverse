"""
Test Missing Dimensions ETLs
"""

# pylint: disable=missing-function-docstring, duplicate-code, missing-class-docstring

import unittest

from sqlalchemy.orm import Session

from mirrorverse.warehouse.etls.missing_dimensions import (
    get_primary_key,
    get_associated_dimensions,
    get_missing_dimension_keys,
)
from mirrorverse.warehouse.etls.facts.cwt import format_cwt_recoveries_data
from mirrorverse.warehouse.models import (
    ModelBase,
    CWTRecoveries,
    Dates,
    CWTLocations,
    CWTReportingAgencies,
    CWTTags,
    Species,
)
from mirrorverse.warehouse.data.cwt import CWT_RETRIEVAL_DATA
from mirrorverse.warehouse.utils import get_engine, upload_dataframe


def test_get_primary_key():
    assert get_primary_key(Dates) == "date_key"


def test_get_associated_dimensions():
    keys, models = get_associated_dimensions(CWTRecoveries)
    assert set(keys) == set(
        [
            "species_key",
            "cwt_tag_key",
            "cwt_reporting_agency_key",
            "recovery_date_key",
            "cwt_recovery_location_key",
        ]
    )
    assert set(models) == set(
        [
            Species,
            CWTTags,
            CWTReportingAgencies,
            Dates,
            CWTLocations,
        ]
    )


class TestMissingDimensions(unittest.TestCase):

    def setUp(self):
        engine = get_engine(db_url="sqlite://")
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.rollback()
        self.session.close()

    def test_get_missing_dimension_keys(self):
        formatted = format_cwt_recoveries_data(CWT_RETRIEVAL_DATA)
        upload_dataframe(self.session, CWTRecoveries, formatted)
        missing_keys = get_missing_dimension_keys(
            Dates, CWTRecoveries, "recovery_date_key", self.session
        )
        assert missing_keys == {"date_key": [1609459200]}
