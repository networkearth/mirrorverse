import pandas as pd
from tqdm import tqdm
from time import time
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.chinook import utils


class DriftMovementChoiceBuilder(object):
    STATE = ["h3_index", "month"]
    CHOICE_STATE = []
    COLUMNS = ["h3_index", "temp", "elevation", "remain"]

    def __call__(self, state, choice_state):
        h3_index = state["h3_index"]

        if h3_index not in utils.NEIGHBORS:
            utils.find_neighbors(h3_index)
        neighbors = utils.NEIGHBORS.get(h3_index)

        choices = pd.DataFrame(neighbors, columns=["h3_index"])

        # might be good to put some assertions around here
        choices["month"] = state["month"]
        choices = choices.merge(
            utils.SURFACE_TEMPS_ENRICHMENT, on=["h3_index", "month"], how="inner"
        )
        choices = choices.merge(utils.ELEVATION_ENRICHMENT, on="h3_index", how="inner")
        del choices["month"]

        choices["remain"] = choices["h3_index"] == h3_index
        return choices


class DriftMovementLeaf(DecisionTree):
    BUILDERS = [DriftMovementChoiceBuilder()]
    FEATURE_COLUMNS = ["temp", "elevation", "remain"]
    BRANCHES = {}
    PARAM_GRID = {"n_estimators": [10, 20, 50, 100], "min_samples_leaf": [50, 100, 200]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    @staticmethod
    def get_identifier(choice):
        pass

    @classmethod
    def update_branch(cls, choice, choice_state):
        choice_state["h3_index"] = choice["h3_index"]

    @staticmethod
    def _stitch_selection(choices, selection):
        choices["selected"] = choices["h3_index"] == selection
        return choices


def train_drift_movement_model(training_data, testing_data):
    print("Training Drift Movement Model...")
    start_time = time()
    drift_states_train = []
    drift_choice_states_train = []
    drift_selections_train = []
    drift_states_test = []
    drift_choice_states_test = []
    drift_selections_test = []

    data = pd.concat([training_data, testing_data])
    training_ptt = set(training_data["ptt"].unique())

    for ptt in tqdm(data["ptt"].unique()):
        ptt_data = data[data["ptt"] == ptt].sort_values("date", ascending=True)
        rows = [row for _, row in ptt_data.iterrows()]
        for start, end in zip(rows[:-1], rows[1:]):
            if not end["drifting"]:
                continue
            state = {
                "h3_index": start["h3_index"],
                "month": start["month"],
            }
            choice_state = {}
            selection = end["h3_index"]
            if ptt in training_ptt:
                drift_states_train.append(state)
                drift_choice_states_train.append(choice_state)
                drift_selections_train.append(selection)
            else:
                drift_states_test.append(state)
                drift_choice_states_test.append(choice_state)
                drift_selections_test.append(selection)

    DriftMovementLeaf.train_model(
        drift_states_train, drift_choice_states_train, drift_selections_train
    )
    print(
        "Train",
        DriftMovementLeaf.test_model(
            drift_states_train, drift_choice_states_train, drift_selections_train
        ),
    )
    print(
        "Test:",
        DriftMovementLeaf.test_model(
            drift_states_test, drift_choice_states_test, drift_selections_test
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return DriftMovementLeaf.export_models(recurse=False)
