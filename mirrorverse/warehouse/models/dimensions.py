"""
Dimension Tables
"""

from datetime import date

from typing import Optional
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from mirrorverse.warehouse.models import ModelBase


class Dates(ModelBase):
    """
    Dates Dimension Table
    """

    __tablename__ = "dates"

    date_key: Mapped[int] = mapped_column(primary_key=True)

    date: Mapped[date]
    day: Mapped[int]
    month: Mapped[int]
    year: Mapped[int]


class CWTLocations(ModelBase):
    """
    CWT Locations Dimension Table
    """

    __tablename__ = "cwt_locations"

    cwt_location_key: Mapped[str] = mapped_column(String(19), primary_key=True)

    cwt_location_name: Mapped[str]
    lon: Mapped[float]
    lat: Mapped[float]
    h3_level_4_key: Mapped[int] = mapped_column(ForeignKey("h3_level_4.h3_level_4_key"))


class CWTTags(ModelBase):
    """
    CWT Tags Dimension Table
    """

    __tablename__ = "cwt_tags"

    cwt_tag_key: Mapped[str] = mapped_column(String(10), primary_key=True)

    cwt_release_location_key: Mapped[str] = mapped_column(
        ForeignKey("cwt_locations.cwt_location_key")
    )
    run: Mapped[Optional[int]]


class H3Level4(ModelBase):
    """
    H3 Level 4 Dimension Table
    """

    __tablename__ = "h3_level_4"

    h3_level_4_key: Mapped[int] = mapped_column(primary_key=True)

    geometry: Mapped[str]
