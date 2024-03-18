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
    cwt_reporting_agency_key: Mapped[str] = mapped_column(
        ForeignKey("cwt_reporting_agencies.cwt_reporting_agency_key"), primary_key=True
    )

    weight: Mapped[Optional[float]]
    length: Mapped[Optional[float]]
    sex: Mapped[Optional[str]] = mapped_column(String(1))
    number_estimated: Mapped[Optional[float]]

    species_key: Mapped[int] = mapped_column(ForeignKey("species.species_key"))
    cwt_tag_key: Mapped[str] = mapped_column(ForeignKey("cwt_tags.cwt_tag_key"))
    recovery_date_key: Mapped[int] = mapped_column(ForeignKey("dates.date_key"))
    cwt_recovery_location_key: Mapped[str] = mapped_column(
        ForeignKey("cwt_locations.cwt_location_key")
    )
