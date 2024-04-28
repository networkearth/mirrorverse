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


from mirrorverse.chinook_monthly.tree.drift_movement import DriftMovementLeaf
from mirrorverse.chinook.states import (
    get_elevation,
    get_surface_temps,
)
from mirrorverse.chinook_monthly.train import regroup


def simulate(args):
    """
    Input:
    - ptt: ptt
    - h3_index (str): starting h3 index
    - year (int): starting year
    - month (int): starting month
    - home_latitude (float): the home latitude of the fish
    - fork_length_cm (float): fork length of the fish
    - steps (int): number of steps to simulate
    - decision_tree (DecisionTree): decision tree to use

    Returns a DataFrame with the simulated data.
    """
    ptt, h3_index, year, month, home_latitude, fork_length_cm, steps, decision_tree = (
        args
    )
    state = {
        "h3_index": h3_index,
        "month": month,
        "home_latitude": home_latitude,
        "fork_length_cm": fork_length_cm,
    }
    row = dict(state)
    row["ptt"] = ptt
    row["year"] = year
    rows = [row]
    for _ in range(steps):
        choice_state = {}
        decision_tree.choose(state, choice_state)

        month = month + 1
        if month == 13:
            month = 1
            year = year + 1
        new_state = {
            "h3_index": choice_state["h3_index"],
            "month": month,
            "home_latitude": home_latitude,
            "fork_length_cm": fork_length_cm,
        }

        row = dict(new_state)
        row["ptt"] = ptt
        row["year"] = year
        rows.append(row)

        state = new_state

    return pd.DataFrame(rows)


@click.command()
@click.option("--data_path", "-d", help="path to data file", required=True)
@click.option("--temps_path", "-t", help="path to surface temps file", required=True)
@click.option("--elevation_path", "-e", help="path to elevation file", required=True)
@click.option("--model_path", "-m", help="path to model file", required=True)
@click.option("--simulation_path", "-si", help="path to simulation file", required=True)
@click.option("--resolution", "-r", help="resolution", type=int, required=True)
def main(
    data_path, temps_path, elevation_path, model_path, simulation_path, resolution
):
    """
    Main function for simulating using the Chinook decision tree.
    """
    pd.options.mode.chained_assignment = None

    print("Pulling Enrichment...")
    enrichment = {
        "elevation": regroup(get_elevation(elevation_path), resolution),
        "surface_temps": regroup(
            get_surface_temps(temps_path), resolution, extra_keys=["month"]
        ),
        "neighbors": {},
    }

    print("Loading Data...")
    data = pd.read_csv(data_path)

    decision_tree = DriftMovementLeaf(enrichment)

    print("Loading Models...")
    with open(model_path, "rb") as fh:
        models = pickle.load(fh)
    for model in models.values():
        model.n_jobs = 1
    decision_tree.import_models(models)

    print("Simulating...")
    jobs = []
    for ptt in data["ptt"].unique():
        df = (
            data[data["ptt"] == ptt]
            .sort_values(["month", "year"], ascending=True)
            .iloc[0]
        )
        steps = data[data["ptt"] == ptt].shape[0]
        jobs.append(
            (
                df["ptt"],
                df["h3_index"],
                df["year"],
                df["month"],
                df["home_latitude"],
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
