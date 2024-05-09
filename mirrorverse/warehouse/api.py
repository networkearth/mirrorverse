"""
To keep commands.py from changing everytime we add some new table.
"""

from functools import partial

from mirrorverse.warehouse.etls.dimensions.dates import build_dates
from mirrorverse.warehouse.etls.dimensions.spatial import build_spatial
from mirrorverse.warehouse.etls.facts.cwt import format_cwt_recoveries_data
from mirrorverse.warehouse.etls.dimensions.cwt import (
    build_cwt_locations,
    build_cwt_tags,
)
from mirrorverse.warehouse.etls.dimensions.tags import build_tags
from mirrorverse.warehouse.etls.facts.tags import (
    format_tag_tracks,
    format_home_regions,
    format_tag_depths,
)
from mirrorverse.warehouse.etls.facts.elevation import format_elevation
from mirrorverse.warehouse.etls.facts.surface_temperature import (
    format_surface_temperature,
)

DIMENSION_FORMATTERS = {
    "dates": build_dates,
    "cwt_locations": build_cwt_locations,
    "cwt_tags": build_cwt_tags,
    "h3_level_4": partial(build_spatial, 4),
    "tags": build_tags,
}
FACT_FORMATTERS = {
    "cwt_recoveries": format_cwt_recoveries_data,
    "tag_tracks": format_tag_tracks,
    "elevation": format_elevation,
    "surface_temperature": format_surface_temperature,
    "home_regions": format_home_regions,
    "tag_depths": format_tag_depths,
}
