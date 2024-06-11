"""
API for the Chinook Depth Model
"""

import click

from mirrorverse.models.chinook_depth.data import (
    load_depth_data,
    add_depth_classes,
)


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
    }
    _callable, required_args = functions[kwargs["function"]]
    args = [kwargs[arg] for arg in required_args]
    _callable(*args)
