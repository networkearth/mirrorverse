import click
import json
import numpy as np

def build_randomized_param_sets(param_grids, M, max_attempts):
    """
    Inputs:
    - param_grids: dict, parameter names to grid of values
    - M: int, number of parameter sets to generate
    - max_attempts: int, maximum number of attempts to generate a unique parameter set

    Outputs:
    - list of dicts, parameter sets
    """
    param_sets = []
    attempts = 0
    while len(param_sets) < M:
        assert attempts < max_attempts

        param_set = {}
        for param, grid in param_grids.items():
            param_set[param] = int(np.random.choice(grid))
        if param_set in param_sets:
            attempts += 1
        else:
            attempts = 0
            param_sets.append(param_set)

    return param_sets

@click.command()
@click.option('-n', '--num_sets', required=True, type=int)
def main(num_sets):
    grids = {
        "batch_size": [500, 600, 700],
        "epochs": [15, 20, 25],
        "layer1": [8, 16, 16],
        "layer2": [8, 8, 16, 16, 16],
        "layer3": [4, 8, 16, 0, 0, 0],
    }
    param_sets = build_randomized_param_sets(grids, num_sets, 10)
    for param_set in param_sets:
        param_set["layers"] = [
            "D" + str(param_set["layer1"]), "D" + str(param_set["layer2"])
        ]
        if param_set["layer3"] != 0:
            param_set["layers"].append("D" + str(param_set["layer3"]))
        del param_set["layer1"]
        del param_set["layer2"]
        del param_set["layer3"]
    print(param_sets)
    with open("param_sets.json", "w") as f:
        json.dump(param_sets, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()
