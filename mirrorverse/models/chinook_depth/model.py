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


def train_model(input_files, output_files, features, learning_rate, iterations):
    """
    Inputs:
    - input_files: str, paths to the input files
    - output_files: str, paths to save the output files
    - features: str, list of feature names
    - learning_rate: float, learning rate for the model
    - iterations: int, number of iterations to run

    Trains the model and saves the diagnostics and model to csv and pickle files.
    """
    input_files = input_files.split(",")
    train_data = pd.read_csv(input_files[0])
    test_data = pd.read_csv(input_files[1])
    with open(input_files[2], "r") as f:
        model_params = json.load(f)

    features = eval(features)
    assert "_decision" not in features
    assert "_selected" not in features
    features += ["_decision", "_selected"]

    model = OddsModel(RandomForestRegressor(**model_params))
    model.fit(train_data[features], test_data[features], learning_rate, iterations)

    output_files = output_files.split(",")
    model.diagnostics.to_csv(output_files[0], index=False)
    with open(output_files[1], "wb") as f:
        pickle.dump(model, f)
