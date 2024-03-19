"""
To keep commands.py from changing everytime we add some new table.
"""

from mirrorverse.warehouse.etls.dimensions.dates import build_dates
from mirrorverse.warehouse.etls.facts.cwt import format_cwt_recoveries_data
from mirrorverse.warehouse.etls.dimensions.cwt import (
    build_cwt_locations,
    build_cwt_tags,
)

DIMENSION_FORMATTERS = {
    "dates": build_dates,
    "cwt_locations": build_cwt_locations,
    "cwt_tags": build_cwt_tags,
}
FACT_FORMATTERS = {"cwt_recoveries": format_cwt_recoveries_data}
