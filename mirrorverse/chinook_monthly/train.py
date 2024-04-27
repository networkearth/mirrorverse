"""
Code for training the models for the Chinook decision tree.
"""

# pylint: disable=duplicate-code

import pickle

import click
import pandas as pd
import h3

from mirrorverse.chinook_monthly.tree.drift_movement import train_drift_movement_model
from mirrorverse.chinook.states import get_elevation, get_surface_temps


def regroup(data, resolution, extra_keys=[]):
    """
    Regroup data into a resolution.
    """
    data["lat"] = data.apply(lambda r: h3.h3_to_geo(r["h3_index"])[0], axis=1)
    data["lon"] = data.apply(lambda r: h3.h3_to_geo(r["h3_index"])[1], axis=1)
    data["h3_index"] = data.apply(
        lambda r: h3.geo_to_h3(r["lat"], r["lon"], resolution), axis=1
    )
    data.drop(["lat", "lon"], axis=1, inplace=True)
    data = data.groupby(extra_keys + ["h3_index"]).mean().reset_index()
    return data


@click.command()
@click.option("--node", "-n", help="node to train on", required=True)
@click.option("--train_data_path", "-tr", help="path to training data", required=True)
@click.option("--test_data_path", "-te", help="path to testing data", required=True)
@click.option("--temps_path", "-t", help="path to surface temps file", required=True)
@click.option("--elevation_path", "-e", help="path to elevation file", required=True)
@click.option("--model_dir", "-m", help="directory to save models in", required=True)
@click.option("--resolution", "-r", help="resolution", type=int, required=True)
def main(
    node,
    train_data_path,
    test_data_path,
    temps_path,
    elevation_path,
    model_dir,
    resolution,
):
    """
    Main function for training the models for the Chinook decision tree.
    """
    pd.options.mode.chained_assignment = None

    print("Pulling Enrichment...")
    enrichment = {
        "elevation": regroup(get_elevation(elevation_path), resolution),
        "surface_temps": regroup(
            get_surface_temps(temps_path), resolution, extra_keys=["month"]
        ),
        "neighbors": {},
    }

    print(enrichment)

    print("Loading Data...")
    training_data = pd.read_csv(train_data_path)
    testing_data = pd.read_csv(test_data_path)

    model_export = {
        "DriftMovementLeaf": train_drift_movement_model,
    }[
        node
    ](training_data, testing_data, enrichment)

    print("Exporting Models...")
    model_export_path = f"{model_dir}/{node}.pkl"
    with open(model_export_path, "wb") as fh:
        pickle.dump(model_export, fh)
