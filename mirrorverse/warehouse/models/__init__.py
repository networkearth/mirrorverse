"""
Models!
"""

# pylint: disable=wrong-import-position, cyclic-import

from sqlalchemy.orm import DeclarativeBase


class ModelBase(DeclarativeBase):
    """
    The base to which all other models are attached
    """


from .dimensions import Dates, CWTLocations, CWTTags, H3Level4, Tags
from .facts import (
    CWTRecoveries,
    TagTracks,
    TagDepths,
    Elevation,
    SurfaceTemperature,
    HomeRegions,
)
