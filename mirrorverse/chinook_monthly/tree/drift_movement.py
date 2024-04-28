"""
Drift Movement Model for Chinook salmon
"""

# pylint: disable=duplicate-code, protected-access

from time import time

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.utility import get_proposed_utility
from mirrorverse.chinook_monthly import utils


class DriftMovementChoiceBuilder:
    """
    Drift Movement choice builder for Chinook salmon.
    """

    STATE = ["h3_index", "month", "home_latitiude"]
    CHOICE_STATE = []
    COLUMNS = ["h3_index", "temp", "elevation", "month", "home_latitude"]

    def __init__(self, enrichment):
        self.neighbors = enrichment["neighbors"]
        self.surface_temps = enrichment["surface_temps"]
        self.elevation = enrichment["elevation"]

    def __call__(self, state, choice_state):
        h3_index = state["h3_index"]

        if h3_index not in self.neighbors:
            utils.find_neighbors(h3_index, self.neighbors)
        neighbors = self.neighbors.get(h3_index)

        choices = pd.DataFrame(neighbors, columns=["h3_index"])

        # might be good to put some assertions around here
        choices["month"] = state["month"]
        choices = choices.merge(
            self.surface_temps, on=["h3_index", "month"], how="inner"
        )
        choices = choices.merge(self.elevation, on="h3_index", how="inner")

        choices["home_latitude"] = state["home_latitude"]

        return choices


class DriftMovementLeaf(DecisionTree):
    """
    Drift Movement model for Chinook salmon.
    """

    BUILDERS = [DriftMovementChoiceBuilder]
    FEATURE_COLUMNS = [
        "temp",
        "elevation",
        "month",
        "home_latitude",
    ]
    OUTCOMES = ["h3_index"]
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

        Updates the choice with a "h3_index" key.
        """
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


def train_drift_movement_model(training_data, testing_data, enrichment):
    """
    Trains a Drift Movement model.
    """
    print("Training Drift Movement Model...")
    start_time = time()
    drift_states_train = []
    drift_choice_states_train = []
    drift_selections_train = []
    identifiers_train = []
    drift_states_test = []
    drift_choice_states_test = []
    drift_selections_test = []
    identifiers_test = []

    data = pd.concat([training_data, testing_data])
    training_ptt = set(training_data["ptt"].unique())

    for ptt in tqdm(data["ptt"].unique()):
        ptt_data = data[data["ptt"] == ptt].sort_values(
            ["year", "month"], ascending=True
        )
        rows = [row for _, row in ptt_data.iterrows()]
        for start, end in zip(rows[:-1], rows[1:]):
            state = {
                "h3_index": start["h3_index"],
                "month": start["month"],
                "home_latitude": start["home_latitude"],
            }
            choice_state = {}
            selection = end["h3_index"]
            if ptt in training_ptt:
                drift_states_train.append(state)
                drift_choice_states_train.append(choice_state)
                drift_selections_train.append(selection)
                identifiers_train.append(ptt)
            else:
                drift_states_test.append(state)
                drift_choice_states_test.append(choice_state)
                drift_selections_test.append(selection)
                identifiers_test.append(ptt)

    decision_tree = DriftMovementLeaf(enrichment)
    model_data = decision_tree._build_model_data(
        drift_states_train,
        drift_choice_states_train,
        drift_selections_train,
        identifiers_train,
        quiet=True,
    )
    decision_tree.train_model(
        drift_states_train,
        drift_choice_states_train,
        drift_selections_train,
        identifiers_train,
        N=20,
        quiet=True,
    )
    model_data["utility"] = decision_tree.model.predict(
        model_data[decision_tree.FEATURE_COLUMNS]
    )
    model_data = get_proposed_utility(model_data)
    model_data.to_csv("DriftMovementLeaf.csv")
    print(
        "Train",
        decision_tree.test_model(
            drift_states_train,
            drift_choice_states_train,
            drift_selections_train,
            identifiers_train,
            quiet=True,
        ),
    )
    print(
        "Test:",
        decision_tree.test_model(
            drift_states_test,
            drift_choice_states_test,
            drift_selections_test,
            identifiers_test,
            quiet=True,
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return decision_tree.export_models(recurse=False)
