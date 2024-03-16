import pandas as pd
import numpy as np
import h3
from tqdm import tqdm
from time import time
from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree
from mirrorverse.chinook import utils

from mirrorverse.chinook.tree.run_movement import RunMovementLeaf


class RunHeadingChoiceBuilder(object):
    STATE = ["h3_index", "month", "mean_heading", "drifting"]
    CHOICE_STATE = []
    COLUMNS = ["mean_heading", "elevation", "temp", "last_mean_heading", "was_drifting"]

    def __call__(self, state, choice_state):
        slices = 24
        step_size = 1

        choices = pd.DataFrame(
            {"mean_heading": np.linspace(2 * np.pi / slices, 2 * np.pi, slices)}
        )
        start_lat, start_lon = h3.h3_to_geo(state["h3_index"])
        choices["end_lat"] = start_lat + step_size * np.sin(choices["mean_heading"])
        choices["end_lon"] = start_lon + step_size * np.cos(choices["mean_heading"])

        choices["h3_index"] = choices.apply(
            lambda row: h3.geo_to_h3(row["end_lat"], row["end_lon"], utils.RESOLUTION),
            axis=1,
        )
        del choices["end_lat"]
        del choices["end_lon"]

        choices["month"] = state["month"]
        choices = choices.merge(
            utils.SURFACE_TEMPS_ENRICHMENT, on=["h3_index", "month"], how="inner"
        )
        choices = choices.merge(utils.ELEVATION_ENRICHMENT, on="h3_index", how="inner")
        del choices["month"]

        choices["last_mean_heading"] = (
            state["mean_heading"] if state["mean_heading"] is not np.nan else 0.0
        )
        choices["was_drifting"] = state["drifting"]

        return choices


class RunHeadingBranch(DecisionTree):
    BUILDERS = [RunHeadingChoiceBuilder()]
    FEATURE_COLUMNS = [
        "mean_heading",
        "elevation",
        "temp",
        "last_mean_heading",
        "was_drifting",
    ]
    BRANCHES = {"run_movement": RunMovementLeaf}
    PARAM_GRID = {"n_estimators": [10, 20, 50, 100], "min_samples_leaf": [50, 100, 200]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    @staticmethod
    def get_identifier(choice):
        return "run_movement"

    @classmethod
    def update_branch(cls, choice, choice_state):
        choice_state["mean_heading"] = choice["mean_heading"]

    @staticmethod
    def _stitch_selection(choices, selection):
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


def train_run_heading_model(training_data, testing_data):
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

    RunHeadingBranch.train_model(
        heading_states_train, heading_choice_states_train, heading_selections_train
    )
    print(
        "Train:",
        RunHeadingBranch.test_model(
            heading_states_train, heading_choice_states_train, heading_selections_train
        ),
    )
    print(
        "Test:",
        RunHeadingBranch.test_model(
            heading_states_test, heading_choice_states_test, heading_selections_test
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return RunHeadingBranch.export_models(recurse=False)
