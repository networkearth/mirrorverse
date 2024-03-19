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
            "recovery_location_name": "ASK",
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
            "recovery_location_name": "ASK",
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
            "recovery_location_name": "ASK",
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
            "recovery_location_name": "ASK",
            "length": 5,
            "weight": 10,
            "number_cwt_estimated": 1.02,
            "tag_code": "091485",
            "sex": "F",
        },
    ]
)
