"""
Tests for Elevation Importer
"""

# pylint: disable=missing-function-docstring

import os
from unittest import mock

import pandas as pd
import numpy as np

from mirrorverse.docks.elevation import pull_elevation_data


FAKE_ELEVATION_DATA = {
    "elevation": np.array([[2000, 1000, 500], [500, -1500, 750]]),
    "lat": np.array([0, 50]),
    "lon": np.array([0, 0, 100]),
}


def test_pull_elevation_data():
    output_path = "tmp_elevation.csv"
    try:
        with mock.patch("netCDF4.Dataset", return_value=FAKE_ELEVATION_DATA):
            pull_elevation_data("fake.nc", output_path)
            results = pd.read_csv(output_path)
            assert results.shape == (4, 3)
            assert list(results["elevation"]) == [1500, 500, -500, 750]
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e
