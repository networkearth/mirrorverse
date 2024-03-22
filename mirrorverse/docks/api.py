"""
To keep commands.py from changing everytime we add some new importer.
"""

from mirrorverse.docks.elevation import pull_elevation_data
from mirrorverse.docks.surface_temperature import pull_surface_temperature_data

FILE_IMPORTERS = {
    "elevation": pull_elevation_data,
    "surface_temperature": pull_surface_temperature_data,
}
