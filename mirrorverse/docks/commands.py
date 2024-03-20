"""
Main Functions for the Docks
"""

import click

from mirrorverse.docks.api import FILE_IMPORTERS


@click.command()
@click.option("--importer", "-i", required=True, help="Importer to use")
@click.option("--input_path", "-f", required=True, help="Path to input file")
@click.option("--output_path", "-o", required=True, help="Path to output file")
def file_import(importer, input_path, output_path):
    """
    Main function for file import
    """
    if importer not in FILE_IMPORTERS:
        raise ValueError(f"Importer {importer} not found")

    FILE_IMPORTERS[importer](input_path, output_path)
