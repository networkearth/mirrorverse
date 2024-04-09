"""
Split Train and Test
"""

import click
import pandas as pd
import numpy as np


@click.command()
@click.option("--data_path", "-d", help="path to data file", required=True)
@click.option("--output_dir", "-o", help="directory to store outputs", required=True)
@click.option("--training_size", "-ts", help="training size", type=float)
def main(data_path, output_dir, training_size):
    """
    Main function for splitting data into training and testing sets.
    """

    print("Reading Data...")
    data = pd.read_csv(data_path)

    print("Splitting into Train and Test...")
    training_ptt = set(
        np.random.choice(
            data["_identifier"].unique(),
            round(data["_identifier"].unique().shape[0] * training_size),
            replace=False,
        )
    )
    testing_ptt = set(data["_identifier"].unique()) - training_ptt
    print(len(training_ptt), len(testing_ptt))

    training_data = data[data["_identifier"].isin(training_ptt)]
    testing_data = data[data["_identifier"].isin(testing_ptt)]

    print("Saving...")
    training_data.to_csv(f"{output_dir}/training_states.csv", index=False)
    if testing_ptt:
        testing_data.to_csv(f"{output_dir}/testing_states.csv", index=False)
