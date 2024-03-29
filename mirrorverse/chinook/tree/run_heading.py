"""
Run Heading Model for Chinook salmon
"""

# pylint: disable=duplicate-code, protected-access

from time import time

import pandas as pd
import numpy as np
import h3
from tqdm import tqdm
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.chinook import utils
from mirrorverse.chinook.tree.run_movement import RunMovementLeaf


class RunHeadingChoiceBuilder:
    """
    Run Heading choice builder for Chinook salmon.
    """

    STATE = ["h3_index", "month", "mean_heading", "drifting"]
    CHOICE_STATE = []
    COLUMNS = [
        "mean_heading",
        "elevation",
        "diff_elevation",
        "temp",
        "last_mean_heading",
        "was_drifting",
        "month",
        # REMOVE
        "start_lat",
        "start_lon",
    ]

    def __init__(self, enrichment):
        self.surface_temps = enrichment["surface_temps"]
        self.elevation = enrichment["elevation"]

    def __call__(self, state, choice_state):
        slices = 24

        choices = pd.DataFrame(
            {"mean_heading": np.linspace(2 * np.pi / slices, 2 * np.pi, slices)}
        )

        # we only want immediate neighbors in this case
        neighbors = pd.DataFrame({"h3_index": list(h3.k_ring(state["h3_index"], 1))})
        start_lat, start_lon = h3.h3_to_geo(state["h3_index"])
        neighbors["h3_heading"] = neighbors.apply(
            lambda row: utils.get_heading(
                start_lat, start_lon, *h3.h3_to_geo(row["h3_index"])
            ),
            axis=1,
        )
        choices = choices.merge(neighbors, how="cross")
        choices["diff_heading"] = choices.apply(
            lambda row: utils.diff_heading(row["mean_heading"], row["h3_heading"]),
            axis=1,
        )
        choices = choices.sort_values("diff_heading", ascending=True).drop_duplicates(
            "mean_heading", keep="first"
        )
        del choices["h3_heading"]
        del choices["diff_heading"]

        # REMOVE
        choices["start_lat"] = start_lat
        choices["start_lon"] = start_lon

        choices["month"] = state["month"]
        choices = choices.merge(
            self.surface_temps, on=["h3_index", "month"], how="inner"
        )
        choices = choices.merge(self.elevation, on="h3_index", how="inner")

        original_elevation = self.elevation[
            self.elevation["h3_index"] == state["h3_index"]
        ]["elevation"].values[0]
        choices["diff_elevation"] = choices["elevation"] - original_elevation

        choices["last_mean_heading"] = (
            state["mean_heading"] if state["mean_heading"] is not np.nan else 0.0
        )
        choices["was_drifting"] = state["drifting"]

        return choices


class RunHeadingBranch(DecisionTree):
    """
    Run Heading model for Chinook salmon.
    """

    BUILDERS = [RunHeadingChoiceBuilder]
    FEATURE_COLUMNS = [
        "mean_heading",
        "elevation",
        "temp",
        "last_mean_heading",
        "was_drifting",
        "diff_elevation",
        "month",
    ]
    OUTCOMES = ["mean_heading"]
    BRANCHES = {"run_movement": RunMovementLeaf}
    PARAM_GRID = {"n_estimators": [10, 20, 50, 100], "min_samples_leaf": [50, 100, 200]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    # pylint: disable=unused-argument
    @staticmethod
    def get_identifier(choice):
        """
        Input:
        - choice (dict): the choice made

        Always returns "run_movement".
        """
        return "run_movement"

    @staticmethod
    def update_branch(choice, choice_state):
        """
        Input:
        - choice (dict): the choice made
        - choice_state (dict): the state of the choice
            thus far

        Updates the choice with a "mean_heading" key.
        """
        choice_state["mean_heading"] = choice["mean_heading"]

    @staticmethod
    def _stitch_selection(choices, selection):
        """
        Input:
        - choices (pd.DataFrame): the choices possible
        - selection (dict): the selection made

        Returns the choices with a "selected" column.
        Selected is given to be the closest mean_heading
        to the given selection.
        """
        df = choices[["mean_heading"]]
        df["selected_heading"] = selection
        df["diff"] = df.apply(
            lambda row: utils.diff_heading(
                row["mean_heading"], row["selected_heading"]
            ),
            axis=1,
        )
        best_choice = df.sort_values("diff", ascending=True)["mean_heading"].values[0]

        choices["selected"] = choices["mean_heading"] == best_choice
        return choices


def train_run_heading_model(training_data, testing_data, enrichment):
    """
    Trains a Run Heading model.
    """
    print("Training Run Heading Model...")
    start_time = time()
    heading_states_train = []
    heading_choice_states_train = []
    heading_selections_train = []
    heading_states_test = []
    heading_choice_states_test = []
    heading_selections_test = []

    data = pd.concat([training_data, testing_data])
    training_ptt = set(training_data["ptt"].unique())

    for ptt in tqdm(data["ptt"].unique()):
        ptt_data = data[data["ptt"] == ptt].sort_values("date", ascending=True)
        rows = [row for _, row in ptt_data.iterrows()]
        for start, end in zip(rows[:-1], rows[1:]):
            if end["drifting"]:
                continue
            state = {
                "h3_index": start["h3_index"],
                "month": start["month"],
                "mean_heading": start["mean_heading"],
                "drifting": start["drifting"],
            }
            choice_state = {}
            selection = end["mean_heading"]
            if ptt in training_ptt:
                heading_states_train.append(state)
                heading_choice_states_train.append(choice_state)
                heading_selections_train.append(selection)
            else:
                heading_states_test.append(state)
                heading_choice_states_test.append(choice_state)
                heading_selections_test.append(selection)

    decision_tree = RunHeadingBranch(enrichment)
    model_data = decision_tree._build_model_data(
        heading_states_train, heading_choice_states_train, heading_selections_train
    )
    model_data.to_csv("RunHeadingBranch.csv")
    decision_tree.train_model(
        heading_states_train,
        heading_choice_states_train,
        heading_selections_train,
        N=20,
    )
    print(
        "Train:",
        decision_tree.test_model(
            heading_states_train, heading_choice_states_train, heading_selections_train
        ),
    )
    print(
        "Test:",
        decision_tree.test_model(
            heading_states_test, heading_choice_states_test, heading_selections_test
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return decision_tree.export_models(recurse=False)
