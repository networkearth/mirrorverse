"""
Fake Surface Temperature Data for Testing
"""

from datetime import datetime

import pandas as pd

SURFACE_TEMPERATURE_DATA = pd.DataFrame(
    [
        {
            "latitude": 46.6,
            "longitude": -133.2,
            "temperature_c": 5,
            "date": datetime(2021, 1, 1),
        },
        {
            "latitude": 46.6,
            "longitude": -133.2,
            "temperature_c": 10,
            "date": datetime(2021, 1, 1),
        },
        {
            "latitude": 56.6,
            "longitude": -133.2,
            "temperature_c": 5,
            "date": datetime(2021, 1, 1),
        },
        {
            "latitude": 56.6,
            "longitude": -133.2,
            "temperature_c": 10,
            "date": datetime(2021, 2, 1),
        },
    ]
)
