"""
Tests for odds model hyperparameter search.
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, duplicate-code, invalid-name

import unittest

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mirrorverse.log_odds_model.search import (
    build_randomized_param_sets,
    randomized_odds_model_search,
)


def test_build_randomized_param_sets():
    param_grids = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }
    M = 5
    max_attempts = 10

    param_sets = build_randomized_param_sets(param_grids, M, max_attempts)

    assert len(param_sets) == M
    for param_set in param_sets:
        assert param_set["a"] in param_grids["a"]
        assert param_set["b"] in param_grids["b"]


def test_build_randomized_param_sets_too_few_for_M():
    param_grids = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }
    M = 10
    max_attempts = 10

    unittest.TestCase().assertRaises(
        AssertionError, build_randomized_param_sets, param_grids, M, max_attempts
    )


class TestRandomizedOddsModelSearch(unittest.TestCase):

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

    def test_randomized_odds_model_search(self):
        param_grid = {"min_weight_fraction_leaf": [0.5, 0.1], "learning_rate": [0.8]}
        M = 2
        max_attempts = 10
        model_class = RandomForestRegressor
        iterations = 10

        param_sets, diagnostics = randomized_odds_model_search(
            self.X_train,
            self.X_test,
            iterations,
            model_class,
            param_grid,
            M,
            max_attempts,
        )

        assert len(param_sets) == M
        assert diagnostics.shape[0] == M * (iterations + 1)
        assert set(param_grid.keys()) - set(diagnostics.columns) == set()
        assert (
            diagnostics[diagnostics["best"]]["min_weight_fraction_leaf"].values[0]
            == 0.1
        )
