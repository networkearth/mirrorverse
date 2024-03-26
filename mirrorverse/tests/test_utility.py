"""
Tests for utility functions.
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, invalid-name

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from mirrorverse.utility import get_proposed_utility, train_utility_model


def test_get_proposed_utility():
    dataframe = pd.DataFrame(
        {
            "_decision": [0, 0, 1, 2, 2, 3, 3, 4],
            "selected": [0, 1, 1, 1, 0, 1, 0, 1],
            "utility": [3, 1, 3, 3, 1, 3, 1, 1],
        }
    )
    results = get_proposed_utility(dataframe)
    expected = pd.DataFrame(
        {
            "_decision": [0, 0, 1, 2, 2, 3, 3, 4],
            "selected": [0, 1, 1, 1, 0, 1, 0, 1],
            "utility": [3, 1, 3, 3, 1, 3, 1, 1],
            "sum_utility": [4, 4, 3, 4, 4, 4, 4, 1],
            "probability": [0.75, 0.25, 1.0, 0.75, 0.25, 0.75, 0.25, 1.0],
            "score": [-0.75, 0.75, 0.0, 0.25, -0.25, 0.25, -0.25, 0.0],
            "proposed": [0.75, 1.75, 3.0, 3.75, 0.75, 3.75, 0.75, 1.0],
        }
    )
    assert_frame_equal(results, expected)


class FakeModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        X = X.copy()
        X["target"] = y
        self.model = X.groupby("feature").agg({"target": "mean"}).reset_index()

    def predict(self, X):
        return X.merge(self.model, on="feature", how="left")["target"]


def test_train_utility_model_1():
    dataframe = pd.DataFrame(
        {
            "_decision": [0, 0, 1, 2, 2, 3, 3, 4],
            "selected": [0, 1, 1, 1, 0, 1, 0, 1],
            "utility": [3, 1, 3, 3, 1, 3, 1, 1],
            "feature": [
                "pizza",
                "sandwich",
                "pizza",
                "pizza",
                "sandwich",
                "pizza",
                "sandwich",
                "sandwich",
            ],
        }
    )
    model = FakeModel()
    trained_model, _ = train_utility_model(model, dataframe, ["feature"], N=1)
    results = trained_model.predict(dataframe[["feature"]])
    expected = pd.Series([1.125, 0.875, 1.125, 1.125, 0.875, 1.125, 0.875, 0.875])
    assert (results == expected).all()


def test_train_utility_model_1_w_diagnostics():
    dataframe = pd.DataFrame(
        {
            "_decision": [0, 0, 1, 2, 2, 3, 3, 4],
            "selected": [0, 1, 1, 1, 0, 1, 0, 1],
            "utility": [3, 1, 3, 3, 1, 3, 1, 1],
            "feature": [
                "pizza",
                "sandwich",
                "pizza",
                "pizza",
                "sandwich",
                "pizza",
                "sandwich",
                "sandwich",
            ],
        }
    )
    model = FakeModel()
    trained_model, diagnostics_results = train_utility_model(
        model, dataframe, ["feature"], N=1, diagnostics=[mean_absolute_error]
    )
    results = trained_model.predict(dataframe[["feature"]])
    expected = pd.Series([1.125, 0.875, 1.125, 1.125, 0.875, 1.125, 0.875, 0.875])
    assert (results == expected).all()
    assert diagnostics_results["mean_absolute_error"] == [0.375, 0.359375]


def test_train_utility_model_100_w_diagnostics():
    dataframe = pd.DataFrame(
        {
            "_decision": [0, 0, 1, 2, 2, 3, 3, 4],
            "selected": [0, 1, 1, 1, 0, 1, 0, 1],
            "utility": [3, 1, 3, 3, 1, 3, 1, 1],
            "feature": [
                "pizza",
                "sandwich",
                "pizza",
                "pizza",
                "sandwich",
                "pizza",
                "sandwich",
                "sandwich",
            ],
        }
    )
    model = FakeModel()
    trained_model, _ = train_utility_model(
        model, dataframe, ["feature"], N=100, diagnostics=[mean_absolute_error]
    )
    results = list(trained_model.predict(dataframe[["feature"]]))
    assert abs(results[0] / results[1] - 2) < 0.05


def test_train_utility_model_1_random_forest():
    dataframe = pd.DataFrame(
        {
            "_decision": [0, 0, 1, 2, 2, 3, 3, 4],
            "selected": [0, 1, 1, 1, 0, 1, 0, 1],
            "utility": [3, 1, 3, 3, 1, 3, 1, 1],
            "feature": [
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
            ],
        }
    )
    model = RandomForestRegressor(bootstrap=False)
    trained_model, _ = train_utility_model(model, dataframe, ["feature"], N=1)
    results = trained_model.predict(dataframe[["feature"]])
    expected = pd.Series([1.125, 0.875, 1.125, 1.125, 0.875, 1.125, 0.875, 0.875])
    assert (results == expected).all()
