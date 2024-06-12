"""
Hyperparameter search for the odds model.
"""

# pylint: disable=invalid-name

import numpy as np
import pandas as pd

from mirrorverse.log_odds_model.log_odds_model import LogOddsModel


def build_randomized_param_sets(param_grids, M, max_attempts):
    """
    Inputs:
    - param_grids: dict, parameter names to grid of values
    - M: int, number of parameter sets to generate
    - max_attempts: int, maximum number of attempts to generate a unique parameter set

    Outputs:
    - list of dicts, parameter sets
    """
    param_sets = []
    attempts = 0
    while len(param_sets) < M:
        assert attempts < max_attempts

        param_set = {}
        for param, grid in param_grids.items():
            param_set[param] = np.random.choice(grid)
        if param_set in param_sets:
            attempts += 1
        else:
            attempts = 0
            param_sets.append(param_set)

    return param_sets


def randomized_odds_model_search(
    train_data,
    test_data,
    iterations,
    model_class,
    param_grids,
    M,
    max_attempts,
):
    """
    Inputs:
    - train_data: pd.DataFrame, training data
    - test_data: pd.DataFrame, testing data
    - iterations: int, number of iterations to run
    - model_class: class, model class to use
    - param_grids: dict, parameter names to grid of values
    - M: int, number of parameter sets to generate
    - max_attempts: int, maximum number of attempts to generate a unique parameter set

    Outputs:
    - list of dicts, parameter sets
    - pd.DataFrame, diagnostics
    """
    param_sets = build_randomized_param_sets(param_grids, M, max_attempts)
    diagnostics_array = []
    for i, param_set in enumerate(param_sets):
        print(f"Running parameter set {i + 1}/{M}")
        to_pass = {k: v for k, v in param_set.items() if k != "learning_rate"}
        model = LogOddsModel(model_class(**to_pass))
        model.fit(train_data, test_data, param_set["learning_rate"], iterations)
        diagnostics = model.diagnostics
        for param, val in param_set.items():
            diagnostics[param] = val
        diagnostics["_param_set"] = i
        diagnostics_array.append(diagnostics)

    diagnostics = pd.concat(diagnostics_array)

    # determine the best parameter set
    diagnostics["best_score"] = diagnostics.groupby("_param_set")[
        "test_likelihood"
    ].transform("max")
    diagnostics["best"] = diagnostics["best_score"] == diagnostics["best_score"].max()
    del diagnostics["best_score"]

    return param_sets, diagnostics
