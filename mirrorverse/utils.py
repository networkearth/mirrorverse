"""
Mirrorverse utility functions
"""

import os
import pickle
import click


@click.command()
@click.option("--model_dir", "-m", help="Directory to save models to")
def bundle_models(model_dir):
    """
    Input:
    - model_dir (str): directory to pull models from and
        save models to
    """
    model_export = {}
    for file_path in os.listdir(model_dir):
        if file_path.endswith(".pkl"):
            with open(f"{model_dir}/{file_path}", "rb") as fh:
                sub_model_export = pickle.load(fh)
                model_export.update(sub_model_export)
    model_export_path = f"{model_dir}/models.pkl"
    with open(model_export_path, "wb") as fh:
        pickle.dump(model_export, fh)
