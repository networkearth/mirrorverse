"""
Tests of Integrated CWT Data
"""

import unittest

from sqlalchemy.orm import Session

from mirrorverse.warehouse.models import ModelBase
from mirrorverse.warehouse.utils import get_engine


class TestCWTRetrievals(unittest.TestCase):

    def setUp(self):
        engine = get_engine()
        ModelBase.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.rollback()
        self.session.close()
