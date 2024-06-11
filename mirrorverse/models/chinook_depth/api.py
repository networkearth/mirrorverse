"""
API for the Chinook Depth Model
"""

import click

from mirrorverse.models.chinook_depth.warehouse import (
    load_depth_data,
    add_depth_classes,
    load_context_data,
    join_in_context_data,
)
from mirrorverse.models.chinook_depth.time import add_time_features


@click.command()
@click.option("--function", required=True, type=str)
@click.option("--output_file", type=str)
@click.option("--input_file", type=str)
@click.option("--depth_classes", type=str)
def main(**kwargs):
    functions = {
        "load_depth_data": (load_depth_data, ["output_file"]),
        "add_depth_classes": (
            add_depth_classes,
            ["input_file", "depth_classes", "output_file"],
        ),
        "load_context_data": (
            load_context_data,
            ["output_file"],
        ),
        "join_in_context_data": (
            join_in_context_data,
            ["input_files", "output_file"],
        ),
        "add_time_features": (
            add_time_features,
            ["input_file", "output_file"],
        ),
    }
    _callable, required_args = functions[kwargs["function"]]
    args = [kwargs[arg] for arg in required_args]
    _callable(*args)
