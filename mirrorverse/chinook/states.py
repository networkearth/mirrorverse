"""
Code for building Chinook salmon states
from tag data.
"""

import click
import pandas as pd
import numpy as np
import h3
import geopy.distance
from tqdm import tqdm

from mirrorverse.chinook.utils import get_heading, diff_heading


def load_tag_tracks(file_path):
    """
    Loads the tag tracks
    """
    data = pd.read_csv(file_path).rename(
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
    return data


def create_pairs(data, context):
    """
    Input:
    - data (pd.DataFrame): the tag tracks
    - context (dict): context including "resolution"

    Returns a DataFrame of pairs of tag tracks
    """
    resolution = context["resolution"]
    pairs = []
    for ptt in tqdm(data["ptt"].unique()):
        rows = [
            row
            for _, row in data[data["ptt"] == ptt]
            .sort_values("date", ascending=True)
            .iterrows()
        ]
        for start, end in zip(rows[:-1], rows[1:]):
            start_h3 = h3.geo_to_h3(start["lat"], start["lon"], resolution)
            end_h3 = h3.geo_to_h3(end["lat"], end["lon"], resolution)
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
    """
    Input:
    - headings (list): the headings
    - heading (float): the heading

    Returns the squared error between the headings
    and the heading.
    """
    return np.mean([diff_heading(h, heading) ** 2 for h in headings])


def find_average_heading(headings, error_func=squared_error_func, tolerance=0.001):
    """
    Input:
    - headings (list): the headings
    - error_func (function): the error function
    - tolerance (float): the tolerance in radians

    Uses a binary search to find the heading that
    minimizes the error function against the input
    headings.
    """
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


def group_headings(df, context):
    """
    Input:
    - df (pd.DataFrame): the pairs
    - context (dict): context including "max_allowable_error"
        and "min_allowable_distance"

    Returns the pairs with the headings grouped into
    drifts and runs. The pairs are also assigned a
    "group" and "momentum" column.
    """
    max_allowable_error, min_allowable_distance = (
        context["max_allowable_error"],
        context["min_allowable_distance"],
    )
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
            max_error = max(diff_heading(h, mean_heading) for h in headings)
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


def spatial_key_to_index(spatial_key):
    """
    Input:
    - spatial_key (int): the spatial key

    Returns the index of the spatial key.
    """
    return hex(spatial_key)[2:]


def get_surface_temps(file_path):
    """
    Loads surface temperature data.
    """
    df = pd.read_csv(file_path).rename(
        {
            "H3 Key 4": "h3_index",
            "Dates - Date Key → Month": "month",
            "Temperature C": "temp",
        },
        axis=1,
    )[["h3_index", "month", "temp"]]
    df["h3_index"] = df["h3_index"].astype(np.int64).astype(str)
    df["h3_index"] = df.apply(
        lambda row: spatial_key_to_index(np.int64(row["h3_index"])), axis=1
    )
    return df


def get_elevation(file_path):
    """
    Loads elevation data.
    """
    df = pd.read_csv(file_path)
    df["h3_index"] = df["h3_index"].astype(np.int64).astype(str)
    df["h3_index"] = df.apply(
        lambda row: spatial_key_to_index(np.int64(row["h3_index"])), axis=1
    )
    return df


@click.command()
@click.option("--data_path", "-d", help="path to data file", required=True)
@click.option("--output_dir", "-o", help="directory to store outputs", required=True)
@click.option("--resolution", "-r", help="resolution", type=int, required=True)
@click.option("--max_allowable_error", "-mae", help="max allowable error", type=float)
@click.option(
    "--min_allowable_distance", "-mad", help="min allowable distance", type=float
)
@click.option("--training_size", "-ts", help="training size", type=float)
def main(
    data_path,
    output_dir,
    resolution,
    max_allowable_error,
    min_allowable_distance,
    training_size,
):
    """
    Main function for building states.
    """
    pd.options.mode.chained_assignment = None
    np.random.seed(42)

    print("Loading Data...")
    data = load_tag_tracks(data_path)
    context = {
        "resolution": resolution,
        "max_allowable_error": max_allowable_error,
        "min_allowable_distance": min_allowable_distance,
    }

    print("Creating Pairs...")
    pairs = create_pairs(data, context)

    print("Grouping Into Drifts and Runs...")
    grouped_pairs = []
    for ptt in tqdm(pairs["ptt"].unique()):
        df = group_headings(pairs[pairs["ptt"] == ptt], context)
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
            round(data["ptt"].unique().shape[0] * training_size),
            replace=False,
        )
    )
    testing_ptt = set(data["ptt"].unique()) - training_ptt
    print(len(training_ptt), len(testing_ptt))

    print("Saving...")
    data[data["ptt"].isin(training_ptt)].to_csv(
        f"{output_dir}/training_states.csv", index=False
    )
    if testing_ptt:
        data[data["ptt"].isin(testing_ptt)].to_csv(
            f"{output_dir}/testing_states.csv", index=False
        )
