import click
import pandas as pd
import numpy as np
import pickle
import h3
from tqdm import tqdm


from mirrorverse.chinook.tree.run_or_drift import RunOrDriftBranch
from mirrorverse.chinook.states import (
    get_elevation,
    get_surface_temps,
)


def simulate(ptt, h3_index, date, steps, decision_tree):
    state = {
        "drifting": True,
        "steps_in_state": 1,
        "h3_index": h3_index,
        "month": date.month,
        "mean_heading": 0,
    }
    row = {k: v for k, v in state.items()}
    row["ptt"] = ptt
    row["date"] = date
    rows = [row]
    for i in range(steps):
        choice_state = {}
        decision_tree.choose(state, choice_state)

        date = date + pd.Timedelta(days=1)
        steps_in_state = (
            state["steps_in_state"] + 1
            if state["drifting"] == choice_state["drifting"]
            else 1
        )
        new_state = {
            "drifting": choice_state["drifting"],
            "steps_in_state": steps_in_state,
            "h3_index": choice_state["h3_index"],
            "month": date.month,
            "mean_heading": (
                0 if choice_state["drifting"] else choice_state["mean_heading"]
            ),
        }

        row = {k: v for k, v in new_state.items()}
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
    decision_tree.import_models(models)

    print("Simulating...")
    dfs = []
    ptts = list(data["ptt"].unique())
    for ptt in tqdm(ptts):
        df = data[data["ptt"] == ptt].sort_values("date", ascending=True).iloc[0]
        steps = data[data["ptt"] == ptt].shape[0]
        date = pd.to_datetime(df["date"])
        df = simulate(df["ptt"], df["h3_index"], date, steps, decision_tree)
        dfs.append(df)

    df = pd.concat(dfs)

    df["lat"] = df.apply(lambda row: h3.h3_to_geo(row["h3_index"])[0], axis=1)
    df["lon"] = df.apply(lambda row: h3.h3_to_geo(row["h3_index"])[1], axis=1)

    print("Saving...")
    df.to_csv(simulation_path, index=False)
