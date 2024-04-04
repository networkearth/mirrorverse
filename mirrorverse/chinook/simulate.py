"""
Code for simulating using the Chinook decision tree.
"""

# pylint: disable=duplicate-code

import os
import pickle
from multiprocessing import Pool

import click
import pandas as pd
import h3


from mirrorverse.chinook.tree.run_or_drift import RunOrDriftBranch
from mirrorverse.chinook.states import (
    get_elevation,
    get_surface_temps,
)


def simulate(args):
    """
    Input:
    - ptt: ptt
    - h3_index (str): starting h3 index
    - date (pd.Timestamp): starting date
    - home_region (str): the home region of the fish
    - fork_length_cm (float): fork length of the fish
    - steps (int): number of steps to simulate
    - decision_tree (DecisionTree): decision tree to use

    Returns a DataFrame with the simulated data.
    """
    ptt, h3_index, date, home_region, fork_length_cm, steps, decision_tree = args
    state = {
        "drifting": True,
        "steps_in_state": 1,
        "h3_index": h3_index,
        "month": date.month,
        "mean_heading": 0,
        "home_region": home_region,
        "fork_length_cm": fork_length_cm,
    }
    row = dict(state)
    row["ptt"] = ptt
    row["date"] = date
    rows = [row]
    for _ in range(steps):
        choice_state = {}
        decision_tree.choose(state, choice_state)

        date = date + pd.Timedelta(days=1)
        steps_in_state = (
            state["steps_in_state"] + 1
            if state["drifting"] == choice_state["drifting"]
            and (
                not choice_state["drifting"]
                and state["mean_heading"] == choice_state["mean_heading"]
            )
            else 1
        )
        new_state = {
            "drifting": choice_state["drifting"],
            "steps_in_state": steps_in_state,
            "h3_index": choice_state["h3_index"],
            "month": date.month,
            "heading": 0 if choice_state["drifting"] else choice_state["heading"],
            "mean_heading": (
                0 if choice_state["drifting"] else choice_state["mean_heading"]
            ),
            "home_region": home_region,
            "fork_length_cm": fork_length_cm,
        }

        row = dict(new_state)
        row["ptt"] = ptt
        row["date"] = date
        rows.append(row)

        state = new_state

    return pd.DataFrame(rows)


@click.command()
@click.option("--data_path", "-d", help="path to data file", required=True)
@click.option("--temps_path", "-t", help="path to surface temps file", required=True)
@click.option("--elevation_path", "-e", help="path to elevation file", required=True)
@click.option("--model_path", "-m", help="path to model file", required=True)
@click.option("--simulation_path", "-si", help="path to simulation file", required=True)
def main(data_path, temps_path, elevation_path, model_path, simulation_path):
    """
    Main function for simulating using the Chinook decision tree.
    """
    pd.options.mode.chained_assignment = None

    print("Pulling Enrichment...")
    enrichment = {
        "elevation": get_elevation(elevation_path),
        "surface_temps": get_surface_temps(temps_path),
        "neighbors": {},
    }

    print("Loading Data...")
    data = pd.read_csv(data_path)

    decision_tree = RunOrDriftBranch(enrichment)

    print("Loading Models...")
    with open(model_path, "rb") as fh:
        models = pickle.load(fh)
    for model in models.values():
        model.n_jobs = 1
    decision_tree.import_models(models)

    print("Simulating...")
    jobs = []
    for ptt in data["ptt"].unique():
        df = data[data["ptt"] == ptt].sort_values("date", ascending=True).iloc[0]
        steps = data[data["ptt"] == ptt].shape[0]
        date = pd.to_datetime(df["date"])
        jobs.append(
            (
                df["ptt"],
                df["h3_index"],
                date,
                df["home_region"],
                df["fork_length_cm"],
                steps,
                decision_tree,
            )
        )

    with Pool(os.cpu_count() - 2) as p:
        dfs = p.map(simulate, jobs)

    df = pd.concat(dfs)

    df["lat"] = df.apply(lambda row: h3.h3_to_geo(row["h3_index"])[0], axis=1)
    df["lon"] = df.apply(lambda row: h3.h3_to_geo(row["h3_index"])[1], axis=1)

    print("Saving...")
    df.to_csv(simulation_path, index=False)
