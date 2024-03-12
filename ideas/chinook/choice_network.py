import pandas as pd
import numpy as np
import h3
import geopy.distance

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import explained_variance_score

pd.options.mode.chained_assignment = None

NEIGHBORS = {}
RESOLUTION = 4
MAX_KM = 100


def spatial_key_to_index(spatial_key):
    return hex(spatial_key)[2:]


ELEVATION = pd.read_csv("data/bathymetry.csv")
ELEVATION["h3_index"] = ELEVATION["h3_index"].astype(np.int64).astype(str)
ELEVATION["h3_index"] = ELEVATION.apply(
    lambda row: spatial_key_to_index(np.int64(row["h3_index"])), axis=1
)

SURFACE_TEMPS = pd.read_csv("data/surface_temps.csv").rename(
    {
        "H3 Key 4": "h3_index",
        "Dates - Date Key → Month": "month",
        "Temperature C": "temp",
    },
    axis=1,
)[["h3_index", "month", "temp"]]
SURFACE_TEMPS["h3_index"] = SURFACE_TEMPS["h3_index"].astype(np.int64).astype(str)
SURFACE_TEMPS["h3_index"] = SURFACE_TEMPS.apply(
    lambda row: spatial_key_to_index(np.int64(row["h3_index"])), axis=1
)


def find_neighbors(h3_index, threshold_km, neighbors_index):
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
            if (
                geopy.distance.geodesic(h3_coords, h3.h3_to_geo(candidate)).km
                <= threshold_km
            ):
                neighbors.add(candidate)
                found_neighbors = True
            checked.add(candidate)
        distance += 1
    neighbors_index[h3_index] = neighbors


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


class ChoiceNetwork(object):
    MODEL = None
    BUILDERS = []
    FEATURE_COLUMNS = []
    NETWORK = {}

    # Model Tuning
    PARAM_GRID = {"n_estimators": [10, 20, 50, 100], "min_samples_leaf": [50, 100, 200]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    @staticmethod
    def get_identifier(choice):
        pass

    @classmethod
    def update_network(cls, choice, choice_state):
        pass

    @classmethod
    def get_choices(cls, state, choice_state):
        choices = []
        for builder in cls.BUILDERS:
            choices += [builder(state, choice_state)]
        return pd.concat(choices)

    @classmethod
    def choose(cls, state, choice_state):
        choices = cls.get_choices(state, choice_state)
        utility = cls.MODEL.predict(choices[cls.FEATURE_COLUMNS])
        if utility.sum() == 0:
            probs = np.ones(len(utility)) / len(utility)
        else:
            probs = utility / utility.sum()
        choice = choices.loc[np.random.choice(choices.index, p=probs)]

        cls.update_network(choice, choice_state)

        identifier = cls.get_identifier(choice)
        if cls.NETWORK.get(identifier) is not None:
            cls.NETWORK[identifier].choose(state, choice_state)

    @staticmethod
    def _stitch_selection(choices, selection):
        pass

    @classmethod
    def _build_model_data(cls, states, choice_states, selections):
        dataframes = []
        for state, choice_state, selection in zip(states, choice_states, selections):
            choices = cls.get_choices(state, choice_state)
            dataframe = cls._stitch_selection(choices, selection)
            dataframes.append(dataframe)
        return pd.concat(dataframes)

    @classmethod
    def test_model(cls, states, choice_states, selections):
        data = cls._build_model_data(states, choice_states, selections)
        X = data[cls.FEATURE_COLUMNS]
        y = data["selected"]
        y_pred = cls.MODEL.predict(X)
        return {"explained_variance": round(explained_variance_score(y, y_pred), 3)}

    @classmethod
    def train_model(cls, states, choice_states, selections):
        data = cls._build_model_data(states, choice_states, selections)
        X = data[cls.FEATURE_COLUMNS]
        y = data["selected"]
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=3),
            param_grid=cls.PARAM_GRID,
            return_train_score=True,
            cv=cls.CV,
            refit=True,
        ).fit(X, y)
        cls.MODEL = grid_search.best_estimator_

    @classmethod
    def what_state(cls):
        state = set()
        choice_state = set()
        for builder in cls.BUILDERS:
            state.update(builder.STATE)
            choice_state.update(builder.CHOICE_STATE)
        for network in cls.NETWORK.values():
            network_state, network_choice_state = network.what_state()
            state.update(network_state)
            choice_state.update(network_choice_state)
        return state, choice_state

    @classmethod
    def export_models(cls):
        models = {cls.__name__: cls.MODEL}
        for network_cls in cls.NETWORK.values():
            models.update(network_cls.export_models())
        return models

    @classmethod
    def import_models(cls, models):
        cls.MODEL = models[cls.__name__]
        for network_cls in cls.NETWORK.values():
            network_cls.import_models(models)


class DriftMovementChoiceBuilder(object):
    STATE = ["h3_index", "month"]
    CHOICE_STATE = []
    COLUMNS = ["h3_index", "temp", "elevation", "remain"]

    def __call__(self, state, choice_state):
        h3_index = state["h3_index"]

        if h3_index not in NEIGHBORS:
            find_neighbors(h3_index, MAX_KM, NEIGHBORS)
        neighbors = NEIGHBORS.get(h3_index)

        choices = pd.DataFrame(neighbors, columns=["h3_index"])

        # might be good to put some assertions around here
        choices["month"] = state["month"]
        choices = choices.merge(SURFACE_TEMPS, on=["h3_index", "month"], how="inner")
        choices = choices.merge(ELEVATION, on="h3_index", how="inner")
        del choices["month"]

        choices["remain"] = choices["h3_index"] == h3_index
        return choices


class DriftMovementNetwork(ChoiceNetwork):
    MODEL = None
    BUILDERS = [DriftMovementChoiceBuilder()]
    FEATURE_COLUMNS = ["temp", "elevation", "remain"]
    NETWORK = {}

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
            find_neighbors(h3_index, MAX_KM, NEIGHBORS)
        neighbors = NEIGHBORS.get(h3_index)

        choices = pd.DataFrame(neighbors, columns=["h3_index"])

        # might be good to put some assertions around here
        choices["month"] = state["month"]
        choices = choices.merge(SURFACE_TEMPS, on=["h3_index", "month"], how="inner")
        choices = choices.merge(ELEVATION, on="h3_index", how="inner")
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


class RunMovementNetwork(ChoiceNetwork):
    MODEL = None
    BUILDERS = [RunMovementChoiceBuilder()]
    FEATURE_COLUMNS = ["temp", "elevation", "heading", "mean_heading", "remain"]
    NETWORK = {}

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
        choices = choices.merge(SURFACE_TEMPS, on=["h3_index", "month"], how="inner")
        choices = choices.merge(ELEVATION, on="h3_index", how="inner")
        del choices["month"]

        choices["last_mean_heading"] = (
            state["mean_heading"] if state["mean_heading"] is not np.nan else 0.0
        )
        choices["was_drifting"] = state["drifting"]

        return choices


class RunHeadingNetwork(ChoiceNetwork):
    MODEL = None
    BUILDERS = [RunHeadingChoiceBuilder()]
    FEATURE_COLUMNS = [
        "mean_heading",
        "elevation",
        "temp",
        "last_mean_heading",
        "was_drifting",
    ]
    NETWORK = {"run_movement": RunMovementNetwork}

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


class RunOrDriftNetwork(ChoiceNetwork):
    MODEL = None
    BUILDERS = [RunOrDriftBuilder()]
    FEATURE_COLUMNS = ["was_drifting", "steps_in_state", "drift"]
    NETWORK = {
        "run": RunHeadingNetwork,
        "drift": DriftMovementNetwork,
    }

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
