"""
Code for training the models for the Chinook decision tree.
"""

# pylint: disable=duplicate-code

import pickle

import click
import pandas as pd

from mirrorverse.chinook.tree.drift_movement import train_drift_movement_model
from mirrorverse.chinook.tree.run_movement import train_run_movement_model
from mirrorverse.chinook.tree.run_heading import train_run_heading_model
from mirrorverse.chinook.tree.run_or_drift import train_run_or_drift_model
from mirrorverse.chinook.tree.run_stay_or_go import train_run_stay_or_go_model
from mirrorverse.chinook.states import get_elevation, get_surface_temps


@click.command()
@click.option("--node", "-n", help="node to train on", required=True)
@click.option("--train_data_path", "-tr", help="path to training data", required=True)
@click.option("--test_data_path", "-te", help="path to testing data", required=True)
@click.option("--temps_path", "-t", help="path to surface temps file", required=True)
@click.option("--elevation_path", "-e", help="path to elevation file", required=True)
@click.option("--model_dir", "-m", help="directory to save models in", required=True)
def main(node, train_data_path, test_data_path, temps_path, elevation_path, model_dir):
    """
    Main function for training the models for the Chinook decision tree.
    """
    pd.options.mode.chained_assignment = None

    print("Pulling Enrichment...")
    enrichment = {
        "elevation": get_elevation(elevation_path),
        "surface_temps": get_surface_temps(temps_path),
        "neighbors": {},
    }

    print("Loading Data...")
    training_data = pd.read_csv(train_data_path)
    testing_data = pd.read_csv(test_data_path)

    model_export = {
        "DriftMovementLeaf": train_drift_movement_model,
        "RunMovementLeaf": train_run_movement_model,
        "RunHeadingBranch": train_run_heading_model,
        "RunOrDriftBranch": train_run_or_drift_model,
        "RunStayOrGoBranch": train_run_stay_or_go_model,
    }[node](training_data, testing_data, enrichment)

    print("Exporting Models...")
    model_export_path = f"{model_dir}/{node}.pkl"
    with open(model_export_path, "wb") as fh:
        pickle.dump(model_export, fh)
