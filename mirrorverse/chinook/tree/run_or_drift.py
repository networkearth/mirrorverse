"""
RunOrDrift model for Chinook salmon.
"""

# pylint: disable=duplicate-code, protected-access

from time import time

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.utility import get_proposed_utility
from mirrorverse.chinook.tree.run_heading import RunHeadingBranch
from mirrorverse.chinook.tree.drift_movement import DriftMovementLeaf


class RunOrDriftBuilder:
    """
    Run or Drift choice builder for Chinook salmon.
    """

    STATE = ["drifting", "steps_in_state", "month", "home_region"]
    CHOICE_STATE = []
    COLUMNS = [
        "was_drifting",
        "steps_in_state",
        "drift",
        "month",
        "unknown",
        "seak",
        "wa/or",
        "bc",
    ]

    def __init__(self, enrichment):
        pass

    def __call__(self, state, choice_state):
        choices = pd.DataFrame(
            [
                {
                    "was_drifting": state["drifting"],
                    "steps_in_state": state["steps_in_state"],
                    "month": state["month"],
                    "drift": True,
                },
                {
                    "was_drifting": state["drifting"],
                    "steps_in_state": state["steps_in_state"],
                    "month": state["month"],
                    "drift": False,
                },
            ]
        )

        one_is_true = False
        for option in ["SEAK", "Unknown", "WA/OR", "BC"]:
            choices[option.lower()] = state["home_region"] == option
            one_is_true |= state["home_region"] == option
        assert one_is_true

        return choices


class RunOrDriftBranch(DecisionTree):
    """
    Run or Drift model for Chinook salmon.
    """

    BUILDERS = [RunOrDriftBuilder]
    FEATURE_COLUMNS = [
        "was_drifting",
        "steps_in_state",
        "drift",
        "month",
        "unknown",
        "seak",
        "wa/or",
        "bc",
    ]
    OUTCOMES = ["drifting"]
    BRANCHES = {
        "run": RunHeadingBranch,
        "drift": DriftMovementLeaf,
    }
    PARAM_GRID = {"n_estimators": [10, 20], "min_samples_leaf": [50, 100]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    @staticmethod
    def get_identifier(choice):
        """
        Input:
        - choice (dict): the choice made

        Returns either "drift" or "run" based on the choice.
        """
        return "drift" if choice["drift"] else "run"

    @staticmethod
    def update_branch(choice, choice_state):
        """
        Input:
        - choice (dict): the choice made
        - choice_state (dict): the state of the choice
            thus far

        Updates the choice with a "drifting" key.
        """
        choice_state["drifting"] = choice["drift"]

    @staticmethod
    def _stitch_selection(choices, selection):
        """
        Input:
        - choices (pd.DataFrame): the choices possible
        - selection (dict): the selection made

        Returns the choices with a "selected" column
        """
        if selection == "run":
            choices["selected"] = ~choices["drift"]
        else:
            choices["selected"] = choices["drift"]
        return choices


def train_run_or_drift_model(training_data, testing_data, enrichment):
    """
    Trains a Run or Drift model.
    """
    print("Training Run or Drift Model...")
    start_time = time()
    run_or_drift_states_train = []
    run_or_drift_choice_states_train = []
    run_or_drift_selections_train = []
    identifiers_train = []
    run_or_drift_states_test = []
    run_or_drift_choice_states_test = []
    run_or_drift_selections_test = []
    identifiers_test = []

    data = pd.concat([training_data, testing_data])
    training_ptt = set(training_data["ptt"].unique())

    for ptt in tqdm(data["ptt"].unique()):
        ptt_data = data[data["ptt"] == ptt].sort_values("date", ascending=True)
        rows = [row for _, row in ptt_data.iterrows()]
        for start, end in zip(rows[:-1], rows[1:]):
            state = {
                "drifting": start["drifting"],
                "steps_in_state": start["steps_in_state"],
                "month": start["month"],
                "home_region": start["home_region"],
            }
            choice_state = {}
            selection = "drift" if end["drifting"] else "run"
            if ptt in training_ptt:
                run_or_drift_states_train.append(state)
                run_or_drift_choice_states_train.append(choice_state)
                run_or_drift_selections_train.append(selection)
                identifiers_train.append(ptt)
            else:
                run_or_drift_states_test.append(state)
                run_or_drift_choice_states_test.append(choice_state)
                run_or_drift_selections_test.append(selection)
                identifiers_test.append(ptt)

    decision_tree = RunOrDriftBranch(enrichment)
    model_data = decision_tree._build_model_data(
        run_or_drift_states_train,
        run_or_drift_choice_states_train,
        run_or_drift_selections_train,
        identifiers_train,
    )
    decision_tree.train_model(
        run_or_drift_states_train,
        run_or_drift_choice_states_train,
        run_or_drift_selections_train,
        identifiers_train,
        N=20,
    )
    model_data["utility"] = decision_tree.model.predict(
        model_data[decision_tree.FEATURE_COLUMNS]
    )
    model_data = get_proposed_utility(model_data)
    model_data.to_csv("RunOrDriftBranch.csv", index=False)
    print(
        "Train:",
        decision_tree.test_model(
            run_or_drift_states_train,
            run_or_drift_choice_states_train,
            run_or_drift_selections_train,
            identifiers_train,
        ),
    )
    print(
        "Test:",
        decision_tree.test_model(
            run_or_drift_states_test,
            run_or_drift_choice_states_test,
            run_or_drift_selections_test,
            identifiers_test,
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return decision_tree.export_models(recurse=False)
