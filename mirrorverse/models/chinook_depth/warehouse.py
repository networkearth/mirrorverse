"""
Loading and Prepping Data for the Chinook Depth Model
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

from mirrorverse.warehouse.utils import get_engine


def load_depth_data(output_file):
    """
    Inputs:
    - output_file: str, path to save the output file

    Loads the depth data from the warehouse and saves it to a csv file.
    """
    sql = """
    select 
        tag_key,
        date_key,
        depth,
        epoch
    from 
        tag_depths
    """
    depth = pd.read_sql(sql, get_engine())
    depth = depth[~np.isnan(depth["depth"])]
    depth.to_csv(output_file, index=False)


def select_a_class(depth, depth_classes):
    """
    Inputs:
    - depth: float, the depth of the fish as recorded
    - depth_classes: np.array, the depth classes to choose from

    Outputs:
    - int, the selected depth class

    Selects a depth class based on the depth of the fish.

    It turns out that PSAT summary data bins the depth into
    intervals so the actual depth is not known. However
    given the recorded depth we can estimate the depth classes
    it could belong to and the likelihoods of each.
    """
    sd = (
        depth * 0.08 / 1.96
    )  # ~two standard deviations gives our 95% confidence interval
    if sd == 0:
        division = np.zeros(len(depth_classes))
        division[0] = 1
    else:
        # we're going to assume the depth classes are sorted
        z = (depth_classes - depth) / sd
        division = norm.cdf(z)
        division[1:] = division[1:] - division[:-1]
    # if there aren't quite enough depth classes the
    # probabilities may not sum to 1, so we'll normalize
    division = division / division.sum()
    return np.random.choice(depth_classes, p=division)


def add_depth_classes(input_file, depth_classes, output_file):
    """
    Inputs:
    - input_file: str, path to the input file
    - depth_classes: str, string of depth classes as an array
    - output_file: str, path to save the output file

    Adds a depth class to the depth data and saves it to a csv file.
    """
    depth_data = pd.read_csv(input_file)
    depth_classes = np.array(eval(depth_classes))
    depth_data["depth_class"] = depth_data["depth"].apply(
        lambda x: select_a_class(x, depth_classes)
    )
    depth_data.to_csv(output_file, index=False)


def load_context_data(output_file):
    """
    Inputs:
    - output_file: str, path to save the output file

    Loads the context data from the warehouse and saves it to a csv file.
    """
    sql = """
    select 
        tt.*,
        h.home_region,
        e.elevation
    from 
        tag_tracks tt 
        left join home_regions h
            on tt.tag_key = h.tag_key
        left join elevation e 
            on tt.h3_level_4_key = e.h3_level_4_key
    """
    context = pd.read_sql(sql, get_engine())
    context.to_csv(output_file, index=False)


def join_in_context_data(input_files, output_file):
    """
    Inputs:
    - input_files: list of str, paths to the input files
    - output_file: str, path to save the output file

    Joins the context data to the depth data and saves it to a csv file.
    """
    input_files = input_files.split(",")
    depth_data = pd.read_csv(input_files[0])
    context_data = pd.read_csv(input_files[1])
    depth_data["tag_key"] = depth_data["tag_key"].astype(str)
    context_data["tag_key"] = context_data["tag_key"].astype(str)
    data = depth_data.merge(
        context_data[
            ["tag_key", "date_key", "longitude", "latitude", "home_region", "elevation"]
        ]
    )
    data.to_csv(output_file, index=False)
