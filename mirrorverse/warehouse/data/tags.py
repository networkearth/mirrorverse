"""
Fake Tag Data for Testing
"""

import pandas as pd

TAGS_DATA = pd.DataFrame(
    [
        {
            "Ptt": "205415",
            "tag.model": "MiniPAT",
            "time.series.resolution.min": 10.0,
            "Region": "Kodiak",
            "fork.length.cm": 80.0,
            "deploy.date.GMT": "2020-10-13 20:17",
            "deploy.latitude": 55.3,
            "deploy.longitude": -151.2,
            "end.date.time.GMT": "2021-04-19 03:00",
            "hypothetical.data.retrieved": 0.658,
            "data.type": "Transmitted",
            "End.Latitude": 45.7,
            "End.Longitude": -124.6,
        },
        {
            "Ptt": "142100",
            "tag.model": "Xtag",
            "time.series.resolution.min": 15.0,
            "Region": "Dutch Harbor",
            "fork.length.cm": 75.0,
            "deploy.date.GMT": "2015-12-02 19:15",
            "deploy.latitude": 59.3,
            "deploy.longitude": -166.6,
            "end.date.time.GMT": "2016-01-22 03:45",
            "hypothetical.data.retrieved": 0.827,
            "data.type": "Transmitted",
            "End.Latitude": 59.5,
            "End.Longitude": -168.4,
        },
        {
            "Ptt": "239204",
            "tag.model": "MiniPAT",
            "time.series.resolution.min": 5.0,
            "Region": "Craig",
            "fork.length.cm": 75.0,
            "deploy.date.GMT": "2022-05-31 16:00",
            "deploy.latitude": 55.6,
            "deploy.longitude": -134.3,
            "end.date.time.GMT": "2022-07-31 01:35",
            "hypothetical.data.retrieved": 0.836,
            "data.type": "Transmitted",
            "End.Latitude": 51.6,
            "End.Longitude": -130.9,
        },
    ]
)

TAG_TRACKS_DATA = pd.DataFrame(
    [
        {
            "Ptt": "211761",
            "Date": "2021-05-01",
            "Most.Likely.Latitude": 46.6,
            "Most.Likely.Longitude": -133.2,
        },
        {
            "Ptt": "212586",
            "Date": "2020-09-15",
            "Most.Likely.Latitude": 58.8,
            "Most.Likely.Longitude": -161.5,
        },
        {
            "Ptt": "205564",
            "Date": "2020-11-26",
            "Most.Likely.Latitude": 67.9,
            "Most.Likely.Longitude": -153.1,
        },
    ]
)
