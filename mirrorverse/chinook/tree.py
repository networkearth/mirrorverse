import pandas as pd
import numpy as np

import h3
import geopy.distance

from sklearn.model_selection import KFold

from mirrorverse.tree import DecisionTree

# at run time this will need to be filled with
# a dataframe of h3 index, elevation pairs
ELEVATION_ENRICHMENT = None

# at run time this will need to be filled with
# a dataframe of h3 index, month, temp triples
SURFACE_TEMPS_ENRICHMENT = None

# Some Basic Configuration
NEIGHBORS = {}
MAX_KM = 100
RESOLUTION = 4


def find_neighbors(h3_index):
    h3_coords = h3.h3_to_geo(h3_index)
    checked = set()
    neighbors = set()
    distance = 1
    found_neighbors = True
    while found_neighbors:
        found_neighbors = False
        candidates = h3.k_ring(h3_index, distance)
        new_candidates = set(candidates) - checked
        for candidate in new_candidates:
            if geopy.distance.geodesic(h3_coords, h3.h3_to_geo(candidate)).km <= MAX_KM:
                neighbors.add(candidate)
                found_neighbors = True
            checked.add(candidate)
        distance += 1
    NEIGHBORS[h3_index] = neighbors


def get_heading(lat1, lon1, lat2, lon2):
    x = lon2 - lon1
    y = lat2 - lat1
    if x == 0 and y == 0:
        return np.nan
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += 2 * np.pi
    return angle


def diff_heading(heading1, heading2):
    if heading1 < heading2:
        heading1, heading2 = heading2, heading1

    diff = heading1 - heading2
    return diff if diff <= np.pi else 2 * np.pi - diff


class DriftMovementChoiceBuilder(object):
    STATE = ["h3_index", "month"]
    CHOICE_STATE = []
    COLUMNS = ["h3_index", "temp", "elevation", "remain"]

    def __call__(self, state, choice_state):
        h3_index = state["h3_index"]

        if h3_index not in NEIGHBORS:
            find_neighbors(h3_index)
        neighbors = NEIGHBORS.get(h3_index)

        choices = pd.DataFrame(neighbors, columns=["h3_index"])

        # might be good to put some assertions around here
        choices["month"] = state["month"]
        choices = choices.merge(
            SURFACE_TEMPS_ENRICHMENT, on=["h3_index", "month"], how="inner"
        )
        choices = choices.merge(ELEVATION_ENRICHMENT, on="h3_index", how="inner")
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
    def update_network(cls, choice, choice_state):
        choice_state["h3_index"] = choice["h3_index"]

    @staticmethod
    def _stitch_selection(choices, selection):
        choices["selected"] = choices["h3_index"] == selection
        return choices


class RunMovementChoiceBuilder(object):
    STATE = ["h3_index", "month"]
    CHOICE_STATE = ["mean_heading"]
    COLUMNS = ["h3_index", "temp", "elevation", "heading", "mean_heading", "remain"]

    def __call__(self, state, choice_state):
        h3_index = state["h3_index"]

        if h3_index not in NEIGHBORS:
            find_neighbors(h3_index)
        neighbors = NEIGHBORS.get(h3_index)

        choices = pd.DataFrame(neighbors, columns=["h3_index"])

        # might be good to put some assertions around here
        choices["month"] = state["month"]
        choices = choices.merge(
            SURFACE_TEMPS_ENRICHMENT, on=["h3_index", "month"], how="inner"
        )
        choices = choices.merge(ELEVATION_ENRICHMENT, on="h3_index", how="inner")
        del choices["month"]

        choices["mean_heading"] = choice_state["mean_heading"]
        choices["remain"] = choices["h3_index"] == h3_index
        choices["heading"] = choices.apply(
            lambda row: get_heading(
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
    def update_network(cls, choice, choice_state):
        choice_state["heading"] = choice["heading"]
        choice_state["h3_index"] = choice["h3_index"]

    @staticmethod
    def _stitch_selection(choices, selection):
        choices["selected"] = choices["h3_index"] == selection
        return choices


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
            lambda row: h3.geo_to_h3(row["end_lat"], row["end_lon"], RESOLUTION), axis=1
        )
        del choices["end_lat"]
        del choices["end_lon"]

        choices["month"] = state["month"]
        choices = choices.merge(
            SURFACE_TEMPS_ENRICHMENT, on=["h3_index", "month"], how="inner"
        )
        choices = choices.merge(ELEVATION_ENRICHMENT, on="h3_index", how="inner")
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
    def update_network(cls, choice, choice_state):
        choice_state["mean_heading"] = choice["mean_heading"]

    @staticmethod
    def _stitch_selection(choices, selection):
        df = choices[["mean_heading"]]
        df["selected_heading"] = selection
        df["diff"] = df.apply(
            lambda row: diff_heading(row["mean_heading"], row["selected_heading"]),
            axis=1,
        )
        best_choice = df.sort_values("diff", ascending=True)["mean_heading"].values[0]

        choices["selected"] = choices["mean_heading"] == best_choice
        return choices


class RunOrDriftBuilder(object):
    STATE = ["drifting", "steps_in_state"]
    CHOICE_STATE = []
    COLUMNS = ["was_drifting", "steps_in_state", "drift"]

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
    BUILDERS = [RunOrDriftBuilder()]
    FEATURE_COLUMNS = ["was_drifting", "steps_in_state", "drift"]
    BRANCHES = {
        "run": RunHeadingBranch,
        "drift": DriftMovementLeaf,
    }
    PARAM_GRID = {"n_estimators": [10, 20, 50, 100], "min_samples_leaf": [50, 100, 200]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    @staticmethod
    def get_identifier(choice):
        return "drift" if choice["drift"] else "run"

    @classmethod
    def update_network(cls, choice, choice_state):
        choice_state["drifting"] = choice["drift"]

    @staticmethod
    def _stitch_selection(choices, selection):
        if selection == "run":
            choices["selected"] = choices["drift"] == False
        else:
            choices["selected"] = choices["drift"] == True
        return choices
