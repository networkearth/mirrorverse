"""
Building the Model Itself
"""

# pylint: disable=eval-used

import json
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from mirrorverse.odds_model.odds_model import OddsModel
from mirrorverse.odds_model.search import randomized_odds_model_search


pd.options.mode.chained_assignment = None


def split_data(input_file, output_file, train_fraction):
    """
    Inputs:
    - input_file: str, path to the input file
    - output_file: str, path to save the output file
    - train_fraction: float, fraction of data to use for training

    Splits the data into training and testing sets and saves them to csv files.
    """
    data = pd.read_csv(input_file)
    ids = data["_identifier"].unique()
    train_ids = np.random.choice(ids, int(train_fraction * len(ids)), replace=False)
    test_ids = np.array(list(set(ids) - set(train_ids)))

    print(f"Training set size: {len(train_ids)}")
    print(f"Testing set size: {len(test_ids)}")

    train_data = data[data["_identifier"].isin(train_ids)]
    test_data = data[data["_identifier"].isin(test_ids)]
    train_data.to_csv(f"train_{output_file}", index=False)
    test_data.to_csv(f"test_{output_file}", index=False)


def clean(param_set):
    """
    Inputs:
    - param_set: dict, parameter set

    Outputs:
    - dict, cleaned parameter set
    """
    _type_map = {
        np.bool_: bool,
        np.int64: int,
    }
    return {
        k: _type_map[type(v)](v) if type(v) in _type_map else v
        for k, v in param_set.items()
    }


def do_search(input_files, output_files, features, iterations, num_param_sets):
    """
    Inputs:
    - input_files: str, paths to the input files
    - output_files: str, paths to save the output files
    - features: str, list of feature names
    - iterations: int, number of iterations to run per fit
    - num_param_sets: int, number of parameter sets to generate

    Searches for the best model parameters and saves
    the diagnostics and parameters to csv and json files.
    """
    input_files = input_files.split(",")
    train_data = pd.read_csv(input_files[0])
    test_data = pd.read_csv(input_files[1])
    with open(input_files[2], "r") as f:
        param_grid = json.load(f)

    features = eval(features)
    assert "_decision" not in features
    assert "_selected" not in features
    features += ["_decision", "_selected"]

    param_sets, diagnostics = randomized_odds_model_search(
        train_data[features],
        test_data[features],
        iterations,
        RandomForestRegressor,
        param_grid,
        num_param_sets,
        100,
    )

    output_files = output_files.split(",")
    diagnostics.to_csv(output_files[0], index=False)
    param_sets = [clean(param_set) for param_set in param_sets]
    with open(output_files[1], "w") as f:
        json.dump(param_sets, f, indent=4, sort_keys=True)


def train_model(input_files, output_files, features, iterations):
    """
    Inputs:
    - input_files: str, paths to the input files
    - output_files: str, paths to save the output files
    - features: str, list of feature names
    - iterations: int, number of iterations to run

    Trains the model and saves the diagnostics and model to csv and pickle files.
    """
    input_files = input_files.split(",")
    train_data = pd.read_csv(input_files[0])
    test_data = pd.read_csv(input_files[1])
    with open(input_files[2], "r") as f:
        param_sets = json.load(f)
    diagnostics = pd.read_csv(input_files[3])

    model_params = param_sets[diagnostics[diagnostics["best"]]["_param_set"].values[0]]
    learning_rate = model_params.pop("learning_rate")

    features = eval(features)
    assert "_decision" not in features
    assert "_selected" not in features
    features += ["_decision", "_selected"]

    model = OddsModel(RandomForestRegressor(**model_params))
    model.fit(train_data[features], test_data[features], learning_rate, iterations)

    output_data = test_data[features].copy()
    model.predict(output_data)

    output_files = output_files.split(",")
    model.diagnostics.to_csv(output_files[0], index=False)
    with open(output_files[1], "wb") as f:
        pickle.dump(model, f)
    output_data.to_csv(output_files[2], index=False)
