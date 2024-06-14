"""
Odds Model Tests
"""

# pylint: disable=missing-function-docstring, invalid-name, missing-class-docstring

import unittest

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from mirrorverse.log_odds_model.log_odds_model import (
    get_central_likelihood,
    get_probability,
    get_proposed_log_odds,
    LogOddsModel,
)


def test_get_central_likelihood():
    X = pd.DataFrame(
        {
            "_selected": [False, True, True, False],
            "probability": [0.1, 0.2, 0.3, 0.4],
        }
    )
    likelihood = get_central_likelihood(X)
    assert likelihood == np.exp((np.log(0.2) + np.log(0.3)) / 2)


def test_get_probability():
    X = pd.DataFrame(
        {
            "_decision": [0, 1, 1, 2],
            "log_odds": [np.log(1), np.log(2), np.log(3), np.log(4)],
        }
    )
    get_probability(X)
    print(X["probability"] - [1, 2 / 5, 3 / 5, 1])
    assert np.abs(X["probability"] - [1.0, 2 / 5, 3 / 5, 1.0]).max() < 10**-10
    assert np.abs(X["sum_odds"] - [1, 5, 5, 4]).max() < 10**-10


def test_get_proposed_log_odds():
    X = pd.DataFrame(
        {
            "_decision": [0, 1, 1, 2],
            "_selected": [True, False, True, True],
            "log_odds": [0, 1, 2, 3],
            "odds": [1, 2, 3, 4],
            "sum_odds": [1, 5, 5, 4],
            "probability": [1, 2 / 5, 3 / 5, 1],
        }
    )
    learning_rate = 0.9
    get_proposed_log_odds(X, learning_rate)
    assert (
        abs(X["partial"] - [0, -1 / 5 * 2, 1 / 5 * 5 / 3 * (1 - 3 / 5) * 3, 0]).max()
        < 10**-10
    )
    assert (X["step"] == X["partial"] * learning_rate).all()
    assert (X["proposed"] == X["log_odds"] + X["step"]).all()


class TestOddsModel(unittest.TestCase):

    def setUp(self):
        self.X_train = pd.concat(
            [
                pd.DataFrame(  # 75% of the time 0 is selected
                    {
                        "_decision": list(range(75)) + list(range(75)),
                        "_selected": [True] * 75 + [False] * 75,
                        "feature": [0] * 75 + [1] * 75,
                    }
                ),
                pd.DataFrame(  # 25% of the time 1 is selected
                    {
                        "_decision": list(range(75, 100)) + list(range(75, 100)),
                        "_selected": [False] * 25 + [True] * 25,
                        "feature": [0] * 25 + [1] * 25,
                    }
                ),
            ]
        )
        self.X_test = pd.concat(
            [
                pd.DataFrame(  # 70% of the time 0 is selected
                    {
                        "_decision": list(range(70)) + list(range(70)),
                        "_selected": [True] * 70 + [False] * 70,
                        "feature": [0] * 70 + [1] * 70,
                    }
                ),
                pd.DataFrame(  # 30% of the time 1 is selected
                    {
                        "_decision": list(range(70, 100)) + list(range(70, 100)),
                        "_selected": [False] * 30 + [True] * 30,
                        "feature": [0] * 30 + [1] * 30,
                    }
                ),
            ]
        )

        model = RandomForestRegressor()
        self.odds_model = LogOddsModel(model)
        self.odds_model.fit(self.X_train, self.X_test, 0.8, 10)

    def test_odds_model_fit(self):
        train_expectation = np.exp(np.log(0.75) * 0.75 + np.log(0.25) * 0.25)
        test_expectation = np.exp(np.log(0.75) * 0.7 + np.log(0.25) * 0.3)

        diagnostics = self.odds_model.diagnostics

        assert (
            diagnostics[diagnostics["iteration"] == 0]["train_likelihood"].values[0]
            == 0.5
        )
        assert (
            diagnostics[diagnostics["iteration"] == 0]["test_likelihood"].values[0]
            == 0.5
        )
        assert (
            abs(
                diagnostics[diagnostics["iteration"] == 10]["train_likelihood"].values[
                    0
                ]
                - train_expectation
            )
            < 0.001
        )
        assert (
            abs(
                diagnostics[diagnostics["iteration"] == 10]["test_likelihood"].values[0]
                - test_expectation
            )
            < 0.001
        )

        assert set(self.X_train.columns) == set(self.X_test.columns)
        assert set(self.X_train.columns) == set(["feature", "_decision", "_selected"])

    def test_odds_model_predict(self):
        self.odds_model.predict(self.X_test)
        assert set(self.X_test.columns) == set(
            ["feature", "_decision", "_selected", "odds", "log_odds", "probability"]
        )
        assert self.X_test[self.X_test["feature"] == 0]["probability"].mean() > 0.7
        assert self.X_test[self.X_test["feature"] == 1]["probability"].mean() < 0.7
