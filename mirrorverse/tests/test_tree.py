"""
Tree Tests
"""

# pylint: disable=missing-function-docstring, missing-class-docstring, attribute-defined-outside-init, invalid-name

import unittest
from functools import partial

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree


class StepSizeChoiceBuilder:
    STATE = ["min_step_size", "max_step_size"]
    CHOICE_STATE = []
    COLUMNS = ["step_size"]

    def __init__(self, enrichment):
        pass

    def __call__(self, state, choice_state):
        step_size = 0.1
        df = pd.DataFrame(
            {
                "step_size": (
                    np.arange(state["min_step_size"], state["max_step_size"], step_size)
                )
            }
        )
        return df


class LinearGridChoiceBuilder:
    STATE = ["min", "max"]
    CHOICE_STATE = ["step_size"]
    COLUMNS = ["x", "y"]

    def __init__(self, y, enrichment):
        self.y = y
        self.enrichment = enrichment

    def __call__(self, state, choice_state):
        df = pd.DataFrame(
            {
                "x": (
                    np.arange(state["min"], state["max"], choice_state["step_size"])
                    + self.enrichment
                )
            }
        )
        df["y"] = self.y
        return df


class LinearGridDecisionTree(DecisionTree):
    BUILDERS = [
        partial(LinearGridChoiceBuilder, 1),
        partial(LinearGridChoiceBuilder, 2),
    ]
    BRANCHES = {}
    FEATURE_COLUMNS = ["x", "y"]
    OUTCOMES = ["x", "y"]
    PARAM_GRID = {"n_estimators": [10, 20, 50], "min_samples_leaf": [1, 2, 5, 10]}
    CV = KFold(n_splits=3, shuffle=True, random_state=42)

    @staticmethod
    def get_identifier(choice):
        # no branches so no need to identify
        pass

    @staticmethod
    def _stitch_selection(choices, selection):
        x, y = selection["x"], selection["y"]
        choices["selected"] = (choices["x"] == x) & (choices["y"] == y)
        return choices

    @staticmethod
    def update_branch(choice, choice_state):
        choice_state["x"] = choice["x"]
        choice_state["y"] = choice["y"]


def test_get_choices():
    state = {"min": 0, "max": 1}
    choice_state = {"step_size": 0.1}
    enrichment = 0
    choices = LinearGridDecisionTree(enrichment).get_choices(state, choice_state)
    expected_choices = pd.concat(
        [
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 1}),
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 2}),
        ]
    ).reset_index(drop=True)
    assert_frame_equal(choices, expected_choices)


# pylint: disable=protected-access
def test_build_model_data():
    states = [{"min": 0, "max": 1}, {"min": 0, "max": 1}]
    choice_states = [{"step_size": 0.1}, {"step_size": 0.1}]
    selections = [{"x": 0, "y": 1}, {"x": 0, "y": 2}]
    enrichment = 0
    data = LinearGridDecisionTree(enrichment)._build_model_data(
        states, choice_states, selections
    )
    expected_data_1 = pd.concat(
        [
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 1}),
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 2}),
        ]
    )
    expected_data_1["selected"] = (expected_data_1["x"] == 0) & (
        expected_data_1["y"] == 1
    )
    expected_data_1["_decision"] = 0
    expected_data_2 = pd.concat(
        [
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 1}),
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 2}),
        ]
    )
    expected_data_2["selected"] = (expected_data_2["x"] == 0) & (
        expected_data_2["y"] == 2
    )
    expected_data_2["_decision"] = 1
    expected_data = pd.concat([expected_data_1, expected_data_2])
    assert_frame_equal(
        data.reset_index(drop=True), expected_data.reset_index(drop=True)
    )


def test_train_model():
    N = 10
    states = [{"min": 0, "max": 1}] * N
    choice_states = [{"step_size": 0.1}] * N
    selections = [{"x": 0, "y": 2}] * N
    enrichment = 0
    decision_tree = LinearGridDecisionTree(enrichment)
    decision_tree.train_model(states, choice_states, selections, learning_rate=None)
    X = decision_tree.get_choices({"min": 0, "max": 1}, {"step_size": 0.1})
    y = decision_tree.model.predict(X[decision_tree.FEATURE_COLUMNS])
    X["utility"] = y
    # all others should've changed by 0 - 0.05 (1 -> 0.95)
    # except for the selection which should've increased
    # by 1 - 0.05 (1 -> 1.95)
    X["expected_utility"] = (X["x"] == 0) & (X["y"] == 2)
    X["expected_utility"] = X["expected_utility"].astype(int) + 0.95
    assert abs((X["expected_utility"] - X["utility"]).sum()) < 10**-10


def test_test_model():
    N = 10
    states = [{"min": 0, "max": 1}] * N
    choice_states = [{"step_size": 0.1}] * N
    selections = [{"x": 0, "y": 2}] * N
    enrichment = 0
    decision_tree = LinearGridDecisionTree(enrichment)
    decision_tree.train_model(states, choice_states, selections, N=1)
    explained_variance_1 = decision_tree.test_model(states, choice_states, selections)
    decision_tree.train_model(states, choice_states, selections, N=2)
    explained_variance_2 = decision_tree.test_model(states, choice_states, selections)
    assert (
        explained_variance_1["explained_variance"]
        < explained_variance_2["explained_variance"]
    )


class MockModel:

    def __init__(self, to_return):
        self.to_return = np.array(to_return)

    # pylint: disable=unused-argument
    def predict(self, X):
        return self.to_return


def test_choose_w_leaf():
    state = {"min": 0, "max": 1}
    choice_state = {"step_size": 0.1}
    enrichment = 0
    decision_tree = LinearGridDecisionTree(enrichment)
    decision_tree.model = MockModel([0] * 10 + [1] * 10)
    decision_tree.choose(state, choice_state)
    assert choice_state["y"] == 2


def test_zeros_arent_a_problem():
    # if this test doesn't throw an error, then we're good
    state = {"min": 0, "max": 1}
    choice_state = {"step_size": 0.1}
    enrichment = 0
    decision_tree = LinearGridDecisionTree(enrichment)
    decision_tree.model = MockModel([0] * 20)
    decision_tree.choose(state, choice_state)


class StepSizeDecisionTree(DecisionTree):
    BUILDERS = [StepSizeChoiceBuilder]
    BRANCHES = {"step_size": LinearGridDecisionTree}
    FEATURE_COLUMNS = ["step_size"]
    OUTCOMES = ["step_size"]
    PARAM_GRID = {"n_estimators": [10, 20, 50], "min_samples_leaf": [1, 2, 5, 10]}
    CV = KFold(n_splits=3, shuffle=True, random_state=42)

    # pylint: disable=unused-argument
    @staticmethod
    def get_identifier(choice):
        return "step_size"

    @staticmethod
    def _stitch_selection(choices, selection):
        choices["selected"] = choices["step_size"] == selection["step_size"]
        return choices

    @staticmethod
    def update_branch(choice, choice_state):
        choice_state["step_size"] = choice["step_size"]


def test_choose_w_branch():
    state = {"min_step_size": 0, "max_step_size": 1, "min": 0, "max": 1}
    choice_state = {}
    enrichment = 0
    decision_tree = StepSizeDecisionTree(enrichment)
    decision_tree.model = MockModel([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    decision_tree.branches["step_size"].model = MockModel([0] * 10 + [1] * 10)
    decision_tree.choose(state, choice_state)
    assert choice_state["step_size"] == 0.1
    assert choice_state["y"] == 2


def test_export_models():
    decision_tree = StepSizeDecisionTree(0)
    decision_tree.model = "m1"
    decision_tree.branches["step_size"].model = "m2"
    exported = decision_tree.export_models()
    assert exported == {"StepSizeDecisionTree": "m1", "LinearGridDecisionTree": "m2"}


def test_export_models_no_recursion():
    decision_tree = StepSizeDecisionTree(0)
    decision_tree.model = "m1"
    decision_tree.branches["step_size"].model = "m2"
    exported = decision_tree.export_models(recurse=False)
    assert exported == {"StepSizeDecisionTree": "m1"}


def test_import_models():
    decision_tree = StepSizeDecisionTree(0)
    models = {"StepSizeDecisionTree": "m1", "LinearGridDecisionTree": "m2"}
    decision_tree.import_models(models)
    assert decision_tree.model == "m1"
    assert decision_tree.branches["step_size"].model == "m2"


def test_import_models_no_recursion():
    decision_tree = StepSizeDecisionTree(0)
    decision_tree.branches["step_size"].model = None
    models = {"StepSizeDecisionTree": "m1", "LinearGridDecisionTree": "m2"}
    decision_tree.import_models(models, recurse=False)
    assert decision_tree.model == "m1"
    assert decision_tree.branches["step_size"].model is None


def test_what_state():
    decision_tree = StepSizeDecisionTree(0)
    state, choice_state = decision_tree.what_state()
    assert state == {"min_step_size", "max_step_size", "min", "max"}
    assert choice_state == {"step_size"}


# pylint: disable=protected-access
class TestNoSelection(unittest.TestCase):

    def test_no_selection(self):
        states = [{"min": 0, "max": 1}, {"min": 0, "max": 1}]
        choice_states = [{"step_size": 0.1}, {"step_size": 0.1}]
        selections = [{"x": 0, "y": -1}, {"x": 0, "y": 2}]
        enrichment = 0
        self.assertRaises(
            AssertionError,
            LinearGridDecisionTree(enrichment)._build_model_data,
            states,
            choice_states,
            selections,
        )
        data = LinearGridDecisionTree(enrichment)._build_model_data(
            states, choice_states, selections, quiet=True
        )
        expected_data_2 = pd.concat(
            [
                pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 1}),
                pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 2}),
            ]
        )
        expected_data_2["selected"] = (expected_data_2["x"] == 0) & (
            expected_data_2["y"] == 2
        )
        expected_data_2["_decision"] = 1
        expected_data = expected_data_2
        assert_frame_equal(
            data.reset_index(drop=True), expected_data.reset_index(drop=True)
        )
