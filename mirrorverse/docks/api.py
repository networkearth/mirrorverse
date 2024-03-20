"""
To keep commands.py from changing everytime we add some new importer.
"""

from mirrorverse.docks.elevation import pull_elevation_data

FILE_IMPORTERS = {
    "elevation": pull_elevation_data,
}
