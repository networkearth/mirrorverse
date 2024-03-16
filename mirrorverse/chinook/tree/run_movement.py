import pandas as pd
import h3
from tqdm import tqdm
from time import time
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.chinook import utils


class RunMovementChoiceBuilder(object):
    STATE = ["h3_index", "month"]
    CHOICE_STATE = ["mean_heading"]
    COLUMNS = ["h3_index", "temp", "elevation", "heading", "mean_heading", "remain"]

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

        choices["mean_heading"] = choice_state["mean_heading"]
        choices["remain"] = choices["h3_index"] == h3_index
        choices["heading"] = choices.apply(
            lambda row: utils.get_heading(
                *h3.h3_to_geo(h3_index), *h3.h3_to_geo(row["h3_index"])
            ),
            axis=1,
        ).fillna(0)
        return choices


class RunMovementLeaf(DecisionTree):
    BUILDERS = [RunMovementChoiceBuilder()]
    FEATURE_COLUMNS = ["temp", "elevation", "heading", "mean_heading", "remain"]
    BRANCHES = {}
    PARAM_GRID = {"n_estimators": [10, 20, 50, 100], "min_samples_leaf": [50, 100, 200]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    @staticmethod
    def get_identifier(choice):
        pass

    @classmethod
    def update_branch(cls, choice, choice_state):
        choice_state["heading"] = choice["heading"]
        choice_state["h3_index"] = choice["h3_index"]

    @staticmethod
    def _stitch_selection(choices, selection):
        choices["selected"] = choices["h3_index"] == selection
        return choices


def train_run_movement_model(training_data, testing_data):
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

    RunMovementLeaf.train_model(
        run_states_train, run_choice_states_train, run_selections_train
    )
    print(
        "Train:",
        RunMovementLeaf.test_model(
            run_states_train, run_choice_states_train, run_selections_train
        ),
    )
    print(
        "Test:",
        RunMovementLeaf.test_model(
            run_states_test, run_choice_states_test, run_selections_test
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return RunMovementLeaf.export_models(recurse=False)
