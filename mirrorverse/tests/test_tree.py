import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree


class StepSizeChoiceBuilder(object):
    STATE = ["min_step_size", "max_step_size"]
    CHOICE_STATE = []
    COLUMNS = ["step_size"]

    def __call__(self, state, choice_state):
        step_size = 0.1
        df = pd.DataFrame(
            {
                "step_size": np.arange(
                    state["min_step_size"], state["max_step_size"], step_size
                )
            }
        )
        return df


class LinearGridChoiceBuilder(object):
    STATE = ["min", "max"]
    CHOICE_STATE = ["step_size"]
    COLUMNS = ["x", "y"]

    def __init__(self, y):
        self.y = y

    def __call__(self, state, choice_state):
        df = pd.DataFrame(
            {"x": np.arange(state["min"], state["max"], choice_state["step_size"])}
        )
        df["y"] = self.y
        return df


class LinearGridDecisionTree(DecisionTree):
    BUILDERS = [
        LinearGridChoiceBuilder(1),
        LinearGridChoiceBuilder(2),
    ]
    BRANCHES = {}
    FEATURE_COLUMNS = ["x", "y"]
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

    @classmethod
    def update_branch(cls, choice, choice_state):
        choice_state["x"] = choice["x"]
        choice_state["y"] = choice["y"]


def test_get_choices():
    state = {"min": 0, "max": 1}
    choice_state = {"step_size": 0.1}
    choices = LinearGridDecisionTree.get_choices(state, choice_state)
    expected_choices = pd.concat(
        [
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 1}),
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 2}),
        ]
    ).reset_index(drop=True)
    assert_frame_equal(choices, expected_choices)


def test_build_model_data():
    states = [{"min": 0, "max": 1}, {"min": 0, "max": 1}]
    choice_states = [{"step_size": 0.1}, {"step_size": 0.1}]
    selections = [{"x": 0, "y": 1}, {"x": 0, "y": 2}]
    data = LinearGridDecisionTree._build_model_data(states, choice_states, selections)
    expected_data_1 = pd.concat(
        [
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 1}),
            pd.DataFrame({"x": np.arange(0, 1, 0.1), "y": 2}),
        ]
    )
    expected_data_1["selected"] = (expected_data_1["x"] == 0) & (
        expected_data_1["y"] == 1
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
    expected_data = pd.concat([expected_data_1, expected_data_2])
    assert_frame_equal(
        data.reset_index(drop=True), expected_data.reset_index(drop=True)
    )


def test_train_model():
    N = 10
    states = [{"min": 0, "max": 1}] * N
    choice_states = [{"step_size": 0.1}] * N
    selections = [{"x": 0, "y": 2}] * N
    decision_tree = LinearGridDecisionTree
    decision_tree.train_model(states, choice_states, selections)
    X = decision_tree.get_choices({"min": 0, "max": 1}, {"step_size": 0.1})
    y = decision_tree.MODEL.predict(X[decision_tree.FEATURE_COLUMNS])
    X["utility"] = y
    X["expected_utility"] = (X["x"] == 0) & (X["y"] == 2)
    assert (X["utility"] == X["expected_utility"]).all()


def test_test_model():
    N = 10
    states = [{"min": 0, "max": 1}] * N
    choice_states = [{"step_size": 0.1}] * N
    selections = [{"x": 0, "y": 2}] * N
    decision_tree = LinearGridDecisionTree
    decision_tree.train_model(states, choice_states, selections)
    explained_variance = decision_tree.test_model(states, choice_states, selections)
    assert explained_variance["explained_variance"] == 1.0


class MockModel(object):

    def __init__(self, to_return):
        self.to_return = np.array(to_return)

    def predict(self, X):
        return self.to_return


def test_choose_w_leaf():
    state = {"min": 0, "max": 1}
    choice_state = {"step_size": 0.1}
    decision_tree = LinearGridDecisionTree
    decision_tree.MODEL = MockModel([0] * 10 + [1] * 10)
    decision_tree.choose(state, choice_state)
    assert choice_state["y"] == 2


def test_zeros_arent_a_problem():
    # if this test doesn't throw an error, then we're good
    state = {"min": 0, "max": 1}
    choice_state = {"step_size": 0.1}
    decision_tree = LinearGridDecisionTree
    decision_tree.MODEL = MockModel([0] * 20)
    decision_tree.choose(state, choice_state)


class StepSizeDecisionTree(DecisionTree):
    BUILDERS = [StepSizeChoiceBuilder()]
    BRANCHES = {"step_size": LinearGridDecisionTree}
    FEATURE_COLUMNS = ["step_size"]
    PARAM_GRID = {"n_estimators": [10, 20, 50], "min_samples_leaf": [1, 2, 5, 10]}
    CV = KFold(n_splits=3, shuffle=True, random_state=42)

    @staticmethod
    def get_identifier(choice):
        return "step_size"

    @staticmethod
    def _stitch_selection(choices, selection):
        choices["selected"] = choices["step_size"] == selection["step_size"]
        return choices

    @classmethod
    def update_branch(cls, choice, choice_state):
        choice_state["step_size"] = choice["step_size"]


def test_choose_w_branch():
    state = {"min_step_size": 0, "max_step_size": 1, "min": 0, "max": 1}
    choice_state = {}
    decision_tree = StepSizeDecisionTree
    decision_tree.MODEL = MockModel([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    decision_tree.BRANCHES["step_size"].MODEL = MockModel([0] * 10 + [1] * 10)
    decision_tree.choose(state, choice_state)
    assert choice_state["step_size"] == 0.1
    assert choice_state["y"] == 2


def test_export_models():
    decision_tree = StepSizeDecisionTree
    decision_tree.MODEL = "m1"
    decision_tree.BRANCHES["step_size"].MODEL = "m2"
    exported = decision_tree.export_models()
    assert exported == {"StepSizeDecisionTree": "m1", "LinearGridDecisionTree": "m2"}


def test_import_models():
    decision_tree = StepSizeDecisionTree
    models = {"StepSizeDecisionTree": "m1", "LinearGridDecisionTree": "m2"}
    decision_tree.import_models(models)
    assert decision_tree.MODEL == "m1"
    assert decision_tree.BRANCHES["step_size"].MODEL == "m2"


def test_what_state():
    decision_tree = StepSizeDecisionTree
    state, choice_state = decision_tree.what_state()
    assert state == {"min_step_size", "max_step_size", "min", "max"}
    assert choice_state == {"step_size"}
