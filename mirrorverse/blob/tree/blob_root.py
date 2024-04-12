"""
The Blob
"""

from time import time

import click
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from mirrorverse.tree import DecisionTree
from mirrorverse.chinook.utils import diff_heading
from mirrorverse.utility import get_proposed_utility


class BlobBuilder:
    """
    Blob choice builder.
    """

    STATE = ["type"]
    CHOICE_STATE = []
    COLUMNS = ["heading", "type"]

    def __init__(self, enrichment):
        pass

    def __call__(self, state, choice_state):
        slices = 24

        choices = pd.DataFrame(
            {"heading": np.linspace(2 * np.pi / slices, 2 * np.pi, slices)}
        )

        choices["type"] = state["type"]

        return choices


class BlobGround:
    """
    Build the Blob Data
    """

    def __init__(self, favored_heading):
        self.favored_heading = favored_heading
        self.choice_builder = BlobBuilder({})
        self.choices = self.choice_builder({"type": 0}, {})
        self.choices["diff_heading"] = self.choices["heading"].apply(
            lambda x: diff_heading(self.favored_heading, x)
        )
        self.choices["utility"] = self.choices["diff_heading"].apply(
            lambda x: 0.5 ** (np.abs(x) / (np.pi / 4))
        )
        self.choices["probability"] = (
            self.choices["utility"] / self.choices["utility"].sum()
        )

    def new_state(self, state):
        """
        Builds a new state for the Blob.

        State should have x, y, type
        """
        heading = np.random.choice(
            self.choices["heading"], p=self.choices["probability"]
        )
        x, y = state["x"], state["y"]
        return {
            "x": x + np.cos(heading),
            "y": y + np.sin(heading),
            "heading": heading,
            "type": state["type"],
        }


class BlobRoot(DecisionTree):
    """
    Blob Model
    """

    BUILDERS = [BlobBuilder]
    FEATURE_COLUMNS = ["heading", "type"]
    OUTCOMES = ["heading"]
    BRANCHES = {}
    PARAM_GRID = {"n_estimators": [10, 20], "min_samples_leaf": [25, 50]}
    CV = KFold(n_splits=5, shuffle=True, random_state=42)

    # pylint: disable=unused-argument
    @staticmethod
    def get_identifier(choice):
        """
        Input:
        - choice (dict): the choice made

        Pass
        """

    @staticmethod
    def update_branch(choice, choice_state):
        """
        Input:
        - choice (dict): the choice made
        - choice_state (dict): the state of the choice
            thus far

        Updates the choice with a "heading" key.
        """
        choice_state["heading"] = choice["heading"]

    @staticmethod
    def _stitch_selection(choices, selection):
        """
        Input:
        - choices (pd.DataFrame): the choices possible
        - selection (dict): the selection made

        Returns the choices with a "selected" column.
        Selected is given to be the closest heading
        to the given selection.
        """
        df = choices[["heading"]]
        df["selected_heading"] = selection
        df["diff"] = df.apply(
            lambda row: diff_heading(row["heading"], row["selected_heading"]),
            axis=1,
        )
        best_choice = df.sort_values("diff", ascending=True)["heading"].values[0]

        choices["selected"] = choices["heading"] == best_choice
        return choices


# pylint: disable=protected-access
def train_blob_model(training_data, testing_data, enrichment):
    """
    Trains a Blob model.
    """
    print("Training Blob Model...")
    start_time = time()
    heading_states_train = []
    heading_choice_states_train = []
    heading_selections_train = []
    identifiers_train = []
    heading_states_test = []
    heading_choice_states_test = []
    heading_selections_test = []
    identifiers_test = []

    data = pd.concat([training_data, testing_data])
    training_ptt = set(training_data["_identifier"].unique())

    for _identifier in tqdm(data["_identifier"].unique()):
        ptt_data = data[data["_identifier"] == _identifier].sort_values(
            "_time", ascending=True
        )
        rows = [row for _, row in ptt_data.iterrows()]
        for _, end in zip(rows[:-1], rows[1:]):
            state = {"type": 0}
            choice_state = {}
            selection = end["heading"]
            if _identifier in training_ptt:
                heading_states_train.append(state)
                heading_choice_states_train.append(choice_state)
                heading_selections_train.append(selection)
                identifiers_train.append(_identifier)
            else:
                heading_states_test.append(state)
                heading_choice_states_test.append(choice_state)
                heading_selections_test.append(selection)
                identifiers_test.append(_identifier)

    decision_tree = BlobRoot(enrichment)
    model_data = decision_tree._build_model_data(
        heading_states_train,
        heading_choice_states_train,
        heading_selections_train,
        identifiers_train,
    )
    decision_tree.train_model(
        heading_states_train,
        heading_choice_states_train,
        heading_selections_train,
        identifiers_train,
        N=20,
    )
    model_data["utility"] = decision_tree.model.predict(
        model_data[decision_tree.FEATURE_COLUMNS]
    )
    model_data = get_proposed_utility(model_data)
    model_data.to_csv("BlobRoot.csv", index=False)
    print(
        "Train:",
        decision_tree.test_model(
            heading_states_train,
            heading_choice_states_train,
            heading_selections_train,
            identifiers_train,
        ),
    )
    print(
        "Test:",
        decision_tree.test_model(
            heading_states_test,
            heading_choice_states_test,
            heading_selections_test,
            identifiers_test,
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")
    return decision_tree.export_models(recurse=False)


@click.command()
@click.option(
    "--favored_heading", "-f", type=float, required=True, help="The favored heading"
)
@click.option(
    "--individuals", "-n", required=True, type=int, help="The number of individuals"
)
@click.option("--steps", "-s", required=True, type=int, help="The number of steps")
@click.option("--output", "-o", required=True, help="The output file")
def build_blob(favored_heading, individuals, steps, output):
    """
    Build the Blob data.
    """
    blob_ground = BlobGround(favored_heading)
    states = []
    for i in range(individuals):
        state = {
            "x": 0,
            "y": 0,
            "heading": favored_heading,
            "_identifier": i,
            "_time": 0,
            "type": 0,
        }
        states.append(state)
        for j in range(steps):
            state = blob_ground.new_state(state)
            state["_identifier"] = i
            state["_time"] = j + 1
            states.append(state)
    pd.DataFrame(states).to_csv(output, index=False)
