import click
import json
import numpy as np
from copy import copy

def build_gridded_param_sets(param_grids):
    """
    Inputs:
    - param_grids: dict, parameter names to grid of values
    - M: int, number of parameter sets to generate
    - max_attempts: int, maximum number of attempts to generate a unique parameter set

    Outputs:
    - list of dicts, parameter sets
    """
    param_sets = [{}]
    for param, grid in param_grids.items():
        new_param_sets = []
        for el in grid:
            for set in param_sets:
                new_set = copy(set)
                new_set[param] = el
                new_param_sets.append(new_set)
        param_sets = new_param_sets

    return param_sets

@click.command()
def main():
    grids = {
        "batch_size": [7500, 12500],
        "random_seed": list(range(7)),
        "epochs": [100],
        "dropout": [0], #[0],
        "num_layers": [3, 4], #[2, 3],
        "layer_size": [16, 24], #[16, 24],
        "learning_rate": [0.001], #[0.0005, 0.001]
    }
    param_sets = build_gridded_param_sets(grids)
    for param_set in param_sets:
        dropout = param_set["dropout"]
        layer_size = param_set["layer_size"]
        if dropout == 0:
            param_set["layers"] = [
                f"D{layer_size}",
            ] * param_set["num_layers"]
        else:
            param_set["layers"] = [
                f"D{layer_size}", f"Dropout{dropout}",
            ] * param_set["num_layers"]
        param_set["optimizer"] = "Adam"
        param_set["optimizer_kwargs"] = {"learning_rate": param_set["learning_rate"]}

    print(param_sets)
    print(len(param_sets))
    with open("param_sets.json", "w") as f:
        json.dump(param_sets, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()
