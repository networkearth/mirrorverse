"""
Creates Full Array of Choices
"""

# pylint: disable=eval-used

import pandas as pd
import numpy as np


def fill_out_choices(input_file, output_file, depth_classes):
    """
    Inputs:
    - input_file: str, path to the input file
    - output_file: str, path to save the output file
    - depth_classes: str, string of depth classes as an array

    Fills out the choices with all possible depth classes and saves it to a csv file.
    """
    depth_classes = np.array(eval(depth_classes))
    choices = pd.read_csv(input_file)
    choices = (
        choices.reset_index(drop=True)
        .reset_index()
        .rename(
            {
                "index": "_decision",
                "depth_class": "selected_class",
                "tag_key": "_identifier",
            },
            axis=1,
        )
    )
    all_choices = choices[["_decision"]].merge(
        pd.DataFrame({"depth_class": depth_classes}), how="cross"
    )
    choices = choices.merge(all_choices, how="outer", on="_decision")
    choices["_selected"] = choices["depth_class"] == choices["selected_class"]
    del choices["selected_class"]
    choices.to_csv(output_file, index=False)
