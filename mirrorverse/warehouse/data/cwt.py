"""
Fake CWT Data for Testing
"""

import pandas as pd
import numpy as np

CWT_RETRIEVAL_DATA = pd.DataFrame(
    [
        {
            "reporting_agency": "USFWS",
            "recovery_date": 20210101,
            "recovery_id": "C1295149",
            "species": 1,
            "run_year": 2019,
            "recovery_location_code": "ASK",
            "length": 5,
            "weight": 10,
            "number_cwt_estimated": 1.02,
            "tag_code": "091485",
            "sex": np.nan,
        },
        {
            "reporting_agency": "USFWS",
            "recovery_date": 20210101,
            "recovery_id": "C1295149",
            "species": 1,
            "run_year": 2018,
            "recovery_location_code": "ASK",
            "length": np.nan,
            "weight": np.nan,
            "number_cwt_estimated": 1.02,
            "tag_code": "091485",
            "sex": "M",
        },
        {
            "reporting_agency": "USFWS",
            "recovery_date": 202101,
            "recovery_id": "C1295149",
            "species": 1,
            "run_year": 2017,
            "recovery_location_code": "ASK",
            "length": 5,
            "weight": 10,
            "number_cwt_estimated": 1.02,
            "tag_code": "091485",
            "sex": "F",
        },
        {
            "reporting_agency": "USFWS",
            "recovery_date": 202101,
            "recovery_id": "C1295149",
            "species": 1,
            "run_year": 2016,
            "recovery_location_code": "ASK",
            "length": 5,
            "weight": 10,
            "number_cwt_estimated": 1.02,
            "tag_code": "091485",
            "sex": "F",
        },
    ]
)

CWT_LOCATIONS_DATA = pd.DataFrame(
    [
        {
            "location_code": "ASK",
            "name": "Astoria, OR",
            "rmis_longitude": -123.83,
            "rmis_latitude": 46.19,
        },
        {
            "location_code": "BON",
            "name": "Bonners Ferry, ID",
            "rmis_longitude": -116.32,
            "rmis_latitude": 48.69,
        },
        {
            "location_code": "BON2",
            "name": "Bonners Ferry 2, ID",
            "rmis_longitude": -116.32,
            "rmis_latitude": 48.69,
        },
        {
            "location_code": "BON3",
            "name": "Bonners Ferry 3, ID",
            "rmis_longitude": np.nan,
            "rmis_latitude": np.nan,
        },
    ]
)
