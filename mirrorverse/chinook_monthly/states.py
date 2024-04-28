import click
import numpy as np
import pandas as pd
import h3

from mirrorverse.chinook.states import load_tag_tracks


@click.command()
@click.option("--data_path", "-d", help="path to data file", required=True)
@click.option("--context_path", "-c", help="path to context file", required=True)
@click.option("--output_dir", "-o", help="directory to store outputs", required=True)
@click.option("--resolution", "-r", help="resolution", type=int, required=True)
@click.option("--training_size", "-ts", help="training size", type=float)
def main(
    data_path,
    context_path,
    output_dir,
    resolution,
    training_size,
):
    """
    Main function for building states.
    """
    pd.options.mode.chained_assignment = None
    np.random.seed(42)

    data = load_tag_tracks(data_path)

    _filter = (
        data.groupby(["ptt", "year", "month"])
        .size()
        .reset_index()
        .groupby("ptt")
        .size()
        .reset_index()
    )
    allowable = set(_filter[_filter[0] > 3]["ptt"])

    data = (
        data[data["ptt"].isin(allowable)]
        .sort_values(["year", "month"])
        .groupby(["ptt", "year", "month"])[["lon", "lat"]]
        .first()
        .reset_index()
    )

    data["h3_index"] = data.apply(
        lambda r: h3.geo_to_h3(r["lat"], r["lon"], resolution), axis=1
    )

    tag_context = pd.read_csv(context_path)
    tag_context.rename({"tag_key": "ptt"}, axis=1, inplace=True)

    # TODO: make this more general
    tag_context["home_latitude"] = tag_context["home_region"].apply(
        lambda x: {"BC": 53.5, "SEAK": 60.0, "Unknown": -1, "WA/OR": 45.0}[x]
    )

    data = data.merge(tag_context, on="ptt", how="inner")

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
