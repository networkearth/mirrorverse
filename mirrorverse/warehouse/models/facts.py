"""
Fact Tables
"""

from typing import Optional
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from mirrorverse.warehouse.models import ModelBase


class CWTRecoveries(ModelBase):
    """
    CWT Recovieres Fact Table
    """

    __tablename__ = "cwt_recoveries"

    run_year: Mapped[int] = mapped_column(primary_key=True)
    recovery_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    cwt_reporting_agency_key: Mapped[str]

    weight: Mapped[Optional[float]]
    length: Mapped[Optional[float]]
    sex: Mapped[Optional[str]] = mapped_column(String(1))
    number_estimated: Mapped[Optional[float]]
    species_key: Mapped[int]

    cwt_tag_key: Mapped[str] = mapped_column(ForeignKey("cwt_tags.cwt_tag_key"))
    recovery_date_key: Mapped[int] = mapped_column(ForeignKey("dates.date_key"))
    cwt_recovery_location_key: Mapped[str] = mapped_column(
        ForeignKey("cwt_locations.cwt_location_key")
    )


class TagTracks(ModelBase):
    """
    Tag Tracks Fact Table
    """

    __tablename__ = "tag_tracks"

    tag_key: Mapped[str] = mapped_column(ForeignKey("tags.tag_key"), primary_key=True)
    date_key: Mapped[int] = mapped_column(
        ForeignKey("dates.date_key"), primary_key=True
    )

    longitude: Mapped[float]
    latitude: Mapped[float]
    h3_level_4_key: Mapped[int] = mapped_column(ForeignKey("h3_level_4.h3_level_4_key"))


class Elevation(ModelBase):
    """
    Elevation Facts Table
    """

    __tablename__ = "elevation"

    h3_level_4_key: Mapped[int] = mapped_column(
        ForeignKey("h3_level_4.h3_level_4_key"), primary_key=True
    )
    elevation: Mapped[float]


class SurfaceTemperature(ModelBase):
    """
    Surface Temperature Facts Table
    """

    __tablename__ = "surface_temperature"

    h3_level_4_key: Mapped[int] = mapped_column(
        ForeignKey("h3_level_4.h3_level_4_key"), primary_key=True
    )
    date_key: Mapped[int] = mapped_column(
        ForeignKey("dates.date_key"), primary_key=True
    )
    temperature_c: Mapped[float]


class HomeRegions(ModelBase):
    """
    Home Region Facts Table
    """

    __tablename__ = "home_regions"

    tag_key: Mapped[str] = mapped_column(ForeignKey("tags.tag_key"), primary_key=True)
    home_region: Mapped[str] = mapped_column(primary_key=True)
