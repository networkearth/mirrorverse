"""
Run Movement Model for Chinook salmon
"""

# pylint: disable=duplicate-code, protected-access

from time import time

import pandas as pd
import h3
from tqdm import tqdm
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.chinook import utils


class RunMovementChoiceBuilder:
    """
    Run Movement choice builder for Chinook salmon.
    """

    STATE = ["h3_index", "month"]
    CHOICE_STATE = ["mean_heading"]
    COLUMNS = [
        "h3_index",
        "temp",
        "elevation",
        "remain",
        "diff_heading",
        "heading",
        "mean_heading",
    ]

    def __init__(self, enrichment):
        self.neighbors = enrichment["neighbors"]
        self.surface_temps = enrichment["surface_temps"]
        self.elevation = enrichment["elevation"]

    def __call__(self, state, choice_state):
        h3_index = state["h3_index"]

        if h3_index not in self.neighbors:
            utils.find_neighbors(h3_index, self.neighbors)
        neighbors = list(self.neighbors.get(h3_index))

        choices = pd.DataFrame(neighbors, columns=["h3_index"])

        # might be good to put some assertions around here
        choices["month"] = state["month"]
        choices = choices.merge(
            self.surface_temps, on=["h3_index", "month"], how="inner"
        )
        choices = choices.merge(self.elevation, on="h3_index", how="inner")

        del choices["month"]

        choices["mean_heading"] = choice_state["mean_heading"]
        choices["remain"] = choices["h3_index"] == h3_index
        choices["heading"] = choices.apply(
            lambda row: utils.get_heading(
                *h3.h3_to_geo(h3_index), *h3.h3_to_geo(row["h3_index"])
            ),
            axis=1,
        ).fillna(0)

        choices["diff_heading"] = choices.apply(
            lambda r: utils.diff_heading(r["heading"], r["mean_heading"]), axis=1
        )

        return choices


class RunMovementLeaf(DecisionTree):
    """
    Run Movement model for Chinook salmon.
    """

    BUILDERS = [RunMovementChoiceBuilder]
    FEATURE_COLUMNS = [
        "temp",
        "elevation",
        "remain",
        "diff_heading",
    ]
    OUTCOMES = ["h3_index", "heading"]
    BRANCHES = {}
    PARAM_GRID = {"n_estimators": [10, 20], "min_samples_leaf": [50, 100]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    # pylint: disable=unused-argument
    @staticmethod
    def get_identifier(choice):
        """
        Input:
        - choice (dict): the choice made

        Does nothing
        """

    @staticmethod
    def update_branch(choice, choice_state):
        """
        Input:
        - choice (dict): the choice made
        - choice_state (dict): the state of the choice
            thus far

        Updates the choice with a "heading" and
        "h3_index" key.
        """
        choice_state["heading"] = choice["heading"]
        choice_state["h3_index"] = choice["h3_index"]

    @staticmethod
    def _stitch_selection(choices, selection):
        """
        Input:
        - choices (pd.DataFrame): the choices possible
        - selection (dict): the selection made

        Returns the choices with a "selected" column.
        Selection is based on the h3_index.
        """
        choices["selected"] = choices["h3_index"] == selection
        return choices


def train_run_movement_model(training_data, testing_data, enrichment):
    """
    Trains a Run Movement model.
    """
    print("Training Run Movement Model...")
    start_time = time()
    run_states_train = []
    run_choice_states_train = []
    run_selections_train = []
    run_states_test = []
    run_choice_states_test = []
    run_selections_test = []

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
            }
            choice_state = {
                "mean_heading": end["mean_heading"],
            }
            selection = end["h3_index"]
            if ptt in training_ptt:
                run_states_train.append(state)
                run_choice_states_train.append(choice_state)
                run_selections_train.append(selection)
            else:
                run_states_test.append(state)
                run_choice_states_test.append(choice_state)
                run_selections_test.append(selection)

    decision_tree = RunMovementLeaf(enrichment)
    model_data = decision_tree._build_model_data(
        run_states_train, run_choice_states_train, run_selections_train, quiet=True
    )
    model_data.to_csv("RunMovementLeaf.csv")
    decision_tree.train_model(
        run_states_train,
        run_choice_states_train,
        run_selections_train,
        N=20,
        quiet=True,
    )
    print(
        "Train:",
        decision_tree.test_model(
            run_states_train, run_choice_states_train, run_selections_train, quiet=True
        ),
    )
    print(
        "Test:",
        decision_tree.test_model(
            run_states_test, run_choice_states_test, run_selections_test, quiet=True
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return decision_tree.export_models(recurse=False)
