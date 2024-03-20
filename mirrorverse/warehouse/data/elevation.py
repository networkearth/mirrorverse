"""
Fake Elevation Data for Testing
"""

import pandas as pd

ELEVATION_DATA = pd.DataFrame(
    [
        {
            "latitude": 46.6,
            "longitude": -133.2,
            "elevation": 100,
        },
        {
            "latitude": 46.6,
            "longitude": -133.2,
            "elevation": 200,
        },
        {
            "latitude": 56.6,
            "longitude": -133.2,
            "elevation": 100,
        },
    ]
)
