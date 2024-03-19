"""
Models!
"""

# pylint: disable=wrong-import-position, cyclic-import

from sqlalchemy.orm import DeclarativeBase


class ModelBase(DeclarativeBase):
    """
    The base to which all other models are attached
    """


from .dimensions import Dates, CWTLocations, CWTReportingAgencies, CWTTags, Species
from .facts import CWTRecoveries
