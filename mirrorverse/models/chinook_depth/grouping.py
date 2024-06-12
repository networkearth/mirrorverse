"""
Grouping of the Chinook Depth model.
"""

import pandas as pd


def merge_back(input_files, output_file):
    """
    Inputs:
    - input_files (str): comma-separated list of input files
    - output_file (str): output file

    Merges the predictions with the choices.
    """
    input_files = input_files.split(",")
    preds = pd.read_csv(input_files[0])[
        ["_decision", "depth_class", "probability", "utility"]
    ]
    choices = pd.read_csv(input_files[1])
    result = choices.merge(preds, on=["_decision", "depth_class"], how="inner")
    result.to_csv(output_file, index=False)
