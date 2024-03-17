"""
RunOrDrift model for Chinook salmon.
"""

# pylint: disable=duplicate-code

from time import time

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.chinook.tree.run_heading import RunHeadingBranch
from mirrorverse.chinook.tree.drift_movement import DriftMovementLeaf


class RunOrDriftBuilder:
    """
    Run or Drift choice builder for Chinook salmon.
    """

    STATE = ["drifting", "steps_in_state"]
    CHOICE_STATE = []
    COLUMNS = ["was_drifting", "steps_in_state", "drift"]

    def __init__(self, enrichment):
        pass

    def __call__(self, state, choice_state):
        return pd.DataFrame(
            [
                {
                    "was_drifting": state["drifting"],
                    "steps_in_state": state["steps_in_state"],
                    "drift": True,
                },
                {
                    "was_drifting": state["drifting"],
                    "steps_in_state": state["steps_in_state"],
                    "drift": False,
                },
            ]
        )


class RunOrDriftBranch(DecisionTree):
    """
    Run or Drift model for Chinook salmon.
    """

    BUILDERS = [RunOrDriftBuilder]
    FEATURE_COLUMNS = ["was_drifting", "steps_in_state", "drift"]
    BRANCHES = {
        "run": RunHeadingBranch,
        "drift": DriftMovementLeaf,
    }
    PARAM_GRID = {"n_estimators": [10, 20, 50, 100], "min_samples_leaf": [50, 100, 200]}
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
    run_or_drift_states_test = []
    run_or_drift_choice_states_test = []
    run_or_drift_selections_test = []

    data = pd.concat([training_data, testing_data])
    training_ptt = set(training_data["ptt"].unique())

    for ptt in tqdm(data["ptt"].unique()):
        ptt_data = data[data["ptt"] == ptt].sort_values("date", ascending=True)
        rows = [row for _, row in ptt_data.iterrows()]
        for start, end in zip(rows[:-1], rows[1:]):
            state = {
                "drifting": start["drifting"],
                "steps_in_state": start["steps_in_state"],
            }
            choice_state = {}
            selection = "drift" if end["drifting"] else "run"
            if ptt in training_ptt:
                run_or_drift_states_train.append(state)
                run_or_drift_choice_states_train.append(choice_state)
                run_or_drift_selections_train.append(selection)
            else:
                run_or_drift_states_test.append(state)
                run_or_drift_choice_states_test.append(choice_state)
                run_or_drift_selections_test.append(selection)

    decision_tree = RunOrDriftBranch(enrichment)
    decision_tree.train_model(
        run_or_drift_states_train,
        run_or_drift_choice_states_train,
        run_or_drift_selections_train,
    )
    print(
        "Train:",
        decision_tree.test_model(
            run_or_drift_states_train,
            run_or_drift_choice_states_train,
            run_or_drift_selections_train,
        ),
    )
    print(
        "Test:",
        decision_tree.test_model(
            run_or_drift_states_test,
            run_or_drift_choice_states_test,
            run_or_drift_selections_test,
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return decision_tree.export_models(recurse=False)
