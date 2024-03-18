from sqlalchemy.orm import DeclarativeBase


class ModelBase(DeclarativeBase):
    pass


from .dimensions import Dates, CWTLocations, CWTReportingAgencies, CWTTags, Species
from .facts import CWTRecoveries
