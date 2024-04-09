"""
The Blob
"""

import click
import pandas as pd
import numpy as np

from mirrorverse.chinook.utils import diff_heading


class BlobBuilder:
    """
    Blob choice builder.
    """

    STATE = []
    CHOICE_STATE = []
    COLUMNS = ["heading"]

    def __init__(self, enrichment):
        pass

    def __call__(self, state, choice_state):
        slices = 24

        choices = pd.DataFrame(
            {"heading": np.linspace(2 * np.pi / slices, 2 * np.pi, slices)}
        )

        return choices


class BlobGround:
    """
    Build the Blob Data
    """

    def __init__(self, favored_heading):
        self.favored_heading = favored_heading
        self.choice_builder = BlobBuilder({})
        self.choices = self.choice_builder({}, {})
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

        State should have x, y
        """
        heading = np.random.choice(
            self.choices["heading"], p=self.choices["probability"]
        )
        x, y = state["x"], state["y"]
        return {"x": x + np.cos(heading), "y": y + np.sin(heading), "heading": heading}


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
        }
        states.append(state)
        for j in range(steps):
            state = blob_ground.new_state(state)
            state["_identifier"] = i
            state["_time"] = j + 1
            states.append(state)
    pd.DataFrame(states).to_csv(output, index=False)
