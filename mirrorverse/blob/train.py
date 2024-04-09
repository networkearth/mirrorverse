"""
Code for training the models for the Blob decision tree.
"""

# pylint: disable=duplicate-code

import pickle

import click
import pandas as pd

from mirrorverse.blob.tree.blob_root import train_blob_model


@click.command()
@click.option("--node", "-n", help="node to train on", required=True)
@click.option("--train_data_path", "-tr", help="path to training data", required=True)
@click.option("--test_data_path", "-te", help="path to testing data", required=True)
@click.option("--model_dir", "-m", help="directory to save models in", required=True)
def main(node, train_data_path, test_data_path, model_dir):
    """
    Main function for training the models for the Blob decision tree.
    """
    pd.options.mode.chained_assignment = None

    print("Pulling Enrichment...")
    enrichment = {}

    print("Loading Data...")
    training_data = pd.read_csv(train_data_path)
    testing_data = pd.read_csv(test_data_path)

    model_export = {
        "BlobRoot": train_blob_model,
    }[
        node
    ](training_data, testing_data, enrichment)

    print("Exporting Models...")
    model_export_path = f"{model_dir}/{node}.pkl"
    with open(model_export_path, "wb") as fh:
        pickle.dump(model_export, fh)
