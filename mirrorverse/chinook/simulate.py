import click
import pandas as pd
import numpy as np
import pickle
import h3
from tqdm import tqdm


from mirrorverse.chinook.tree import RunOrDriftBranch
from mirrorverse.chinook.train import (
    create_pairs,
    group_headings,
    get_elevation,
    get_surface_temps,
)


def simulate(ptt, h3_index, date, steps):
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
        RunOrDriftBranch.choose(state, choice_state)

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
@click.option("--state_path", "-st", help="path to state file", required=True)
@click.option("--simulation_path", "-si", help="path to simulation file", required=True)
def main(
    data_path, temps_path, elevation_path, model_path, state_path, simulation_path
):

    pd.options.mode.chained_assignment = None

    print("Pulling Enrichment...")
    get_surface_temps(temps_path)
    get_elevation(elevation_path)

    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path).rename(
        {
            "Ptt": "ptt",
            "Latitude": "lat",
            "Longitude": "lon",
            "Dates - Date Key → Date": "date",
            "Dates - Date Key → Year": "year",
            "Dates - Date Key → Month": "month",
            "Dates - Date Key → Day": "day",
        },
        axis=1,
    )

    print("Loading Models...")
    with open(model_path, "rb") as fh:
        models = pickle.load(fh)

    RunOrDriftBranch.import_models(models)

    print("Creating Pairs...")
    pairs = create_pairs(data)

    print("Grouping Pairs...")
    grouped_pairs = []
    for ptt in tqdm(pairs["ptt"].unique()):
        df = group_headings(pairs[pairs["ptt"] == ptt], np.pi / 4, 150)
        grouped_pairs.append(df)
    grouped_pairs = pd.concat(grouped_pairs)

    print("Building State...")
    data = grouped_pairs.copy()
    data["drifting"] = ~data["momentum"]
    data.loc[data["drifting"], "steps_in_state"] = data.loc[
        data["drifting"], "steps_since_group"
    ]
    data.loc[~data["drifting"], "steps_in_state"] = data.loc[
        ~data["drifting"], "steps_in_group"
    ]
    data["h3_index"] = data["start_h3"]
    data["month"] = data["start_month"]
    data["mean_heading"] = data["mean_heading"].fillna(0)
    data["date"] = data["start_date"]
    data = data[
        [
            "ptt",
            "h3_index",
            "month",
            "mean_heading",
            "drifting",
            "steps_in_state",
            "date",
        ]
    ]

    print("Simulating...")
    dfs = []
    ptts = list(data["ptt"].unique())
    for ptt in tqdm(ptts):
        df = data[data["ptt"] == ptt].sort_values("date", ascending=True).iloc[0]
        steps = data[data["ptt"] == ptt].shape[0]
        date = pd.to_datetime(df["date"])
        df = simulate(df["ptt"], df["h3_index"], date, steps)
        dfs.append(df)

    df = pd.concat(dfs)

    df["lat"] = df.apply(lambda row: h3.h3_to_geo(row["h3_index"])[0], axis=1)
    df["lon"] = df.apply(lambda row: h3.h3_to_geo(row["h3_index"])[1], axis=1)

    print("Saving...")
    df.to_csv(simulation_path, index=False)
    data.to_csv(state_path, index=False)


if __name__ == "__main__":
    main()
