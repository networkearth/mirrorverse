"""
RunStayOrGo model for Chinook Salmon
"""

# pylint: disable=duplicate-code, protected-access

from time import time

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.utility import get_proposed_utility
from mirrorverse.chinook.tree.run_movement import RunMovementLeaf


class RunStayOrGoBuilder:
    """
    Run Stay or Go choice builder for Chinook salmon.
    """

    STATE = ["h3_index", "month", "home_region"]
    CHOICE_STATE = ["remain"]
    COLUMNS = [
        "h3_index",
        "remain",
        "unknown",
        "seak",
        "wa/or",
        "bc",
    ]

    def __init__(self, enrichment):
        pass

    def __call__(self, state, choice_state):
        h3_index = state["h3_index"]
        month = state["month"]

        choices = pd.DataFrame(
            [
                {"h3_index": h3_index, "month": month, "remain": True},
                {"h3_index": h3_index, "month": month, "remain": False},
            ]
        )

        one_is_true = False
        for option in ["SEAK", "Unknown", "WA/OR", "BC"]:
            choices[option.lower()] = state["home_region"] == option
            one_is_true |= state["home_region"] == option
        assert one_is_true

        return choices


class RunStayOrGoBranch(DecisionTree):
    """
    Run Movement model for Chinook salmon.
    """

    BUILDERS = [RunStayOrGoBuilder]
    FEATURE_COLUMNS = [
        "month",
        "remain",
        "unknown",
        "seak",
        "wa/or",
        "bc",
    ]
    OUTCOMES = ["h3_index", "heading"]  # heading is just here to keep things consistent
    BRANCHES = {"go": RunMovementLeaf}
    PARAM_GRID = {"n_estimators": [10, 20], "min_samples_leaf": [10, 100]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    # pylint: disable=unused-argument
    @staticmethod
    def get_identifier(choice):
        """
        Input:
        - choice (dict): the choice made

        Let's us know if we decided to go
        """
        return None if choice["remain"] else "go"

    @staticmethod
    def update_branch(choice, choice_state):
        """
        Input:
        - choice (dict): the choice made
        - choice_state (dict): the state of the choice
            thus far

        Sets h3_index to the current h3_index
        and heading to the mean heading.
        If we go rather than stay, this will get overriden.
        """
        choice_state["h3_index"] = choice["h3_index"]
        choice_state["heading"] = choice_state["mean_heading"]

    @staticmethod
    def _stitch_selection(choices, selection):
        """
        Input:
        - choices (pd.DataFrame): the choices possible
        - selection (dict): the selection made

        Returns the choices with a "selected" column.
        Selection should be a boolean indicating if we stayed.
        """
        choices["selected"] = choices["remain"] == selection
        return choices


def train_run_stay_or_go_model(training_data, testing_data, enrichment):
    """
    Trains a Run Stay or Go model.
    """
    print("Training Run Stay or Go Model...")
    start_time = time()
    run_states_train = []
    run_choice_states_train = []
    run_selections_train = []
    identifiers_train = []
    run_states_test = []
    run_choice_states_test = []
    run_selections_test = []
    identifiers_test = []

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
                "home_region": start["home_region"],
            }
            choice_state = {}
            selection = end["h3_index"] == start["h3_index"]
            if ptt in training_ptt:
                run_states_train.append(state)
                run_choice_states_train.append(choice_state)
                run_selections_train.append(selection)
                identifiers_train.append(ptt)
            else:
                run_states_test.append(state)
                run_choice_states_test.append(choice_state)
                run_selections_test.append(selection)
                identifiers_test.append(ptt)

    decision_tree = RunStayOrGoBranch(enrichment)
    model_data = decision_tree._build_model_data(
        run_states_train,
        run_choice_states_train,
        run_selections_train,
        identifiers_train,
        quiet=True,
    )
    decision_tree.train_model(
        run_states_train,
        run_choice_states_train,
        run_selections_train,
        identifiers_train,
        N=20,
        quiet=True,
    )
    model_data["utility"] = decision_tree.model.predict(
        model_data[decision_tree.FEATURE_COLUMNS]
    )
    model_data = get_proposed_utility(model_data)
    model_data.to_csv("RunStayOrGoBranch.csv")
    print(
        "Train:",
        decision_tree.test_model(
            run_states_train,
            run_choice_states_train,
            run_selections_train,
            identifiers_train,
            quiet=True,
        ),
    )
    print(
        "Test:",
        decision_tree.test_model(
            run_states_test,
            run_choice_states_test,
            run_selections_test,
            identifiers_test,
            quiet=True,
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return decision_tree.export_models(recurse=False)
