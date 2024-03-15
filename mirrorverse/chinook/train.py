import click
import numpy as np
import pandas as pd
import h3
import geopy.distance
import pickle
from tqdm import tqdm
from time import time

from mirrorverse.chinook.tree import (
    RESOLUTION,
    diff_heading,
    get_heading,
    RunOrDriftBranch,
    RunHeadingBranch,
    RunMovementLeaf,
    DriftMovementLeaf,
)

import mirrorverse.chinook.tree as chinook_tree


def create_pairs(data):
    pairs = []
    for ptt in tqdm(data["ptt"].unique()):
        rows = [
            row
            for _, row in data[data["ptt"] == ptt]
            .sort_values("date", ascending=True)
            .iterrows()
        ]
        for start, end in zip(rows[:-1], rows[1:]):
            start_h3 = h3.geo_to_h3(start["lat"], start["lon"], RESOLUTION)
            end_h3 = h3.geo_to_h3(end["lat"], end["lon"], RESOLUTION)
            start_lat, start_lon = h3.h3_to_geo(start_h3)
            end_lat, end_lon = h3.h3_to_geo(end_h3)
            heading = get_heading(start_lat, start_lon, end_lat, end_lon)
            new_row = {
                "ptt": ptt,
                "start_lat": start_lat,
                "start_lon": start_lon,
                "end_lat": end_lat,
                "end_lon": end_lon,
                "start_date": start["date"],
                "end_date": end["date"],
                "heading": heading,
                "start_h3": start_h3,
                "end_h3": end_h3,
                "start_month": start["month"],
                "start_day": start["day"],
                "end_month": end["month"],
                "end_day": end["day"],
                "remained": heading is np.nan,
            }
            pairs.append(new_row)
    return pd.DataFrame(pairs)


def squared_error_func(headings, heading):
    return np.mean([diff_heading(h, heading) ** 2 for h in headings])


def find_average_heading(headings, error_func=squared_error_func, tolerance=0.001):
    step_size = np.pi / 8
    direction = 1
    proposed_heading = 0
    error = error_func(headings, proposed_heading)
    while step_size >= tolerance:
        proposed_heading = proposed_heading + step_size * direction
        if proposed_heading < 0:
            proposed_heading += 2 * np.pi
        elif proposed_heading > 2 * np.pi:
            proposed_heading -= 2 * np.pi
        new_error = error_func(headings, proposed_heading)
        if new_error > error:
            direction *= -1
            step_size /= 2
        error = new_error
    return proposed_heading


def group_headings(df, max_allowable_error, min_allowable_distance):
    groups = []
    group = None
    for _, row in df.sort_values("start_date", ascending=True).iterrows():
        if not group:
            # initialize
            group = {
                "rows": [row],
                "headings": [row["heading"]] if not row["remained"] else [],
            }
            continue

        if row["remained"]:
            group["rows"].append(row)
        else:
            headings = group["headings"] + [row["heading"]]
            mean_heading = find_average_heading(headings)
            max_error = max([diff_heading(h, mean_heading) for h in headings])
            if max_error <= max_allowable_error:
                group["rows"].append(row)
                group["headings"] = headings
            else:
                groups.append(group)
                group = {
                    "rows": [row],
                    "headings": [row["heading"]],
                }
    if group:
        groups.append(group)

    group_rows = []
    i = 0
    for group in groups:
        rows = group["rows"]
        distance = geopy.distance.geodesic(
            (rows[0]["start_lat"], rows[0]["start_lon"]),
            (rows[-1]["end_lat"], rows[-1]["end_lon"]),
        ).km
        mean_heading = find_average_heading(group["headings"])
        if distance >= min_allowable_distance:
            for j, row in enumerate(rows):
                new_row = {
                    "ptt": row["ptt"],
                    "start_date": row["start_date"],
                    "group": i,
                    "steps_in_group": j,
                    "momentum": True,
                    "mean_heading": mean_heading,
                }
                group_rows.append(new_row)
            i += 1
    if group_rows:
        df = df.merge(pd.DataFrame(group_rows), on=["ptt", "start_date"], how="left")
        df["group"] = df["group"].fillna(-1)
        df["momentum"] = df["momentum"].fillna(False)
    else:
        df["group"] = -1
        df["momentum"] = False

    rows = []
    last_ptt = None
    i = 1
    j = 0
    for _, row in df.sort_values(["ptt", "start_date"], ascending=True).iterrows():
        if row["ptt"] != last_ptt:
            i = 1
            last_ptt = row["ptt"]
        if not row["momentum"]:
            if i == 1:
                j += 1
            row["steps_since_group"] = i
            row["drift_group"] = j
            i += 1
        else:
            row["steps_since_group"] = np.nan
            row["drift_group"] = np.nan
            i = 1
        rows.append(row)
    df = pd.DataFrame(rows)

    return df


# ENRICHMENT FUNCTIONS
def spatial_key_to_index(spatial_key):
    return hex(spatial_key)[2:]


def get_surface_temps(file_path):
    chinook_tree.SURFACE_TEMPS_ENRICHMENT = pd.read_csv(file_path).rename(
        {
            "H3 Key 4": "h3_index",
            "Dates - Date Key → Month": "month",
            "Temperature C": "temp",
        },
        axis=1,
    )[["h3_index", "month", "temp"]]
    chinook_tree.SURFACE_TEMPS_ENRICHMENT["h3_index"] = (
        chinook_tree.SURFACE_TEMPS_ENRICHMENT["h3_index"].astype(np.int64).astype(str)
    )
    chinook_tree.SURFACE_TEMPS_ENRICHMENT["h3_index"] = (
        chinook_tree.SURFACE_TEMPS_ENRICHMENT.apply(
            lambda row: spatial_key_to_index(np.int64(row["h3_index"])), axis=1
        )
    )


def get_elevation(file_path):
    chinook_tree.ELEVATION_ENRICHMENT = pd.read_csv(file_path)
    chinook_tree.ELEVATION_ENRICHMENT["h3_index"] = (
        chinook_tree.ELEVATION_ENRICHMENT["h3_index"].astype(np.int64).astype(str)
    )
    chinook_tree.ELEVATION_ENRICHMENT["h3_index"] = (
        chinook_tree.ELEVATION_ENRICHMENT.apply(
            lambda row: spatial_key_to_index(np.int64(row["h3_index"])), axis=1
        )
    )


# TRAINING FUNCTIONS
def train_drift_movement_model(data, training_ptt):
    print("Training Drift Movement Model...")
    start_time = time()
    drift_states_train = []
    drift_choice_states_train = []
    drift_selections_train = []
    drift_states_test = []
    drift_choice_states_test = []
    drift_selections_test = []

    for ptt in tqdm(data["ptt"].unique()):
        ptt_data = data[data["ptt"] == ptt].sort_values("date", ascending=True)
        rows = [row for _, row in ptt_data.iterrows()]
        for start, end in zip(rows[:-1], rows[1:]):
            if not end["drifting"]:
                continue
            state = {
                "h3_index": start["h3_index"],
                "month": start["month"],
            }
            choice_state = {}
            selection = end["h3_index"]
            if ptt in training_ptt:
                drift_states_train.append(state)
                drift_choice_states_train.append(choice_state)
                drift_selections_train.append(selection)
            else:
                drift_states_test.append(state)
                drift_choice_states_test.append(choice_state)
                drift_selections_test.append(selection)

    DriftMovementLeaf.train_model(
        drift_states_train, drift_choice_states_train, drift_selections_train
    )
    print(
        "Train",
        DriftMovementLeaf.test_model(
            drift_states_train, drift_choice_states_train, drift_selections_train
        ),
    )
    print(
        "Test:",
        DriftMovementLeaf.test_model(
            drift_states_test, drift_choice_states_test, drift_selections_test
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")


def train_run_movement_model(data, training_ptt):
    print("Training Run Movement Model...")
    start_time = time()
    run_states_train = []
    run_choice_states_train = []
    run_selections_train = []
    run_states_test = []
    run_choice_states_test = []
    run_selections_test = []

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


def train_run_heading_model(data, training_ptt):
    print("Training Run Heading Model...")
    start_time = time()
    heading_states_train = []
    heading_choice_states_train = []
    heading_selections_train = []
    heading_states_test = []
    heading_choice_states_test = []
    heading_selections_test = []

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


def train_run_or_drift_model(data, training_ptt):
    print("Training Run or Drift Model...")
    start_time = time()
    run_or_drift_states_train = []
    run_or_drift_choice_states_train = []
    run_or_drift_selections_train = []
    run_or_drift_states_test = []
    run_or_drift_choice_states_test = []
    run_or_drift_selections_test = []

    for ptt in tqdm(data["ptt"].unique()):
        ptt_data = data[data["ptt"] == ptt].sort_values("date", ascending=True)
        rows = [row for _, row in ptt_data.iterrows()]
        for start, end in zip(rows[:-1], rows[1:]):
            state = {
                "drifting": start["drifting"],
                "steps_in_state": start["steps_in_state"],
            }
            choice_state = {}
            selection = "drift" if end["drifting"] else "run"
            if ptt in training_ptt:
                run_or_drift_states_train.append(state)
                run_or_drift_choice_states_train.append(choice_state)
                run_or_drift_selections_train.append(selection)
            else:
                run_or_drift_states_test.append(state)
                run_or_drift_choice_states_test.append(choice_state)
                run_or_drift_selections_test.append(selection)

    RunOrDriftBranch.train_model(
        run_or_drift_states_train,
        run_or_drift_choice_states_train,
        run_or_drift_selections_train,
    )
    print(
        "Train:",
        RunOrDriftBranch.test_model(
            run_or_drift_states_train,
            run_or_drift_choice_states_train,
            run_or_drift_selections_train,
        ),
    )
    print(
        "Test:",
        RunOrDriftBranch.test_model(
            run_or_drift_states_test,
            run_or_drift_choice_states_test,
            run_or_drift_selections_test,
        ),
    )

    end_time = time()
    print("Time:", round(end_time - start_time, 1), "seconds")


@click.command()
@click.option("--data_path", "-d", help="path to data file", required=True)
@click.option("--temps_path", "-t", help="path to surface temps file", required=True)
@click.option("--elevation_path", "-e", help="path to elevation file", required=True)
@click.option("--model_path", "-m", help="path to model file", required=True)
def main(data_path, temps_path, elevation_path, model_path):

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

    print("Filtering Data...")
    data = data[~data["ptt"].isin(["129843"])]

    print("Creating Pairs...")

    pairs = create_pairs(data)

    print("Grouping Into Drifts and Runs...")

    grouped_pairs = []
    for ptt in tqdm(pairs["ptt"].unique()):
        df = group_headings(pairs[pairs["ptt"] == ptt], np.pi / 4, 150)
        grouped_pairs.append(df)
    grouped_pairs = pd.concat(grouped_pairs)

    print("Creating Context...")
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

    print("Splitting into Train and Test...")
    training_ptt = set(
        np.random.choice(
            data["ptt"].unique(),
            round(data["ptt"].unique().shape[0] * 0.7),
            replace=False,
        )
    )
    testing_ptt = set(data["ptt"].unique()) - training_ptt
    print(len(training_ptt), len(testing_ptt))

    train_drift_movement_model(data, training_ptt)
    train_run_movement_model(data, training_ptt)
    train_run_heading_model(data, training_ptt)
    train_run_or_drift_model(data, training_ptt)

    print("Exporting Models...")
    models = RunOrDriftBranch.export_models()
    with open(model_path, "wb") as fh:
        pickle.dump(models, fh)


if __name__ == "__main__":
    main()
