"""
Queries for Chinook Model
"""

import click
import pandas as pd

from mirrorverse.warehouse.utils import get_engine


def get_elevation():
    """
    Retrieve elevation information from the warehouse

    Returns pd.DataFrame
    """
    sql = """
    select
        elevation,
        h3_level_4_key as h3_index
    from 
        elevation
    """
    return pd.read_sql_query(sql, get_engine())


def get_surface_temperature():
    """
    Retrieve surface temperature information from the warehouse

    Returns pd.DataFrame
    """
    sql = """
    select
        s.temperature_c,
        s.h3_level_4_key as h3_index,
        d.month
    from 
        surface_temperature s
        inner join dates d
        on s.date_key = d.date_key
    """
    return pd.read_sql_query(sql, get_engine())


def get_tag_tracks():
    """
    Retrieve tag tracks information from the warehouse

    Returns pd.DataFrame
    """
    sql = """
    select
        tt.tag_key,
        tt.latitude,
        tt.longitude,
        d.date,
        d.year,
        d.month,
        d.day
    from 
        tag_tracks tt 
        inner join dates d
        on tt.date_key = d.date_key
    """
    return pd.read_sql_query(sql, get_engine())


@click.command()
@click.option("--elevation", "-e", required=True, help="file to save elevation data")
@click.option(
    "--surface_temperature",
    "-s",
    required=True,
    help="file to save surface temperature data",
)
@click.option("--tag_tracks", "-t", required=True, help="file to save tag tracks data")
def main(elevation, surface_temperature, tag_tracks):
    """
    Retrieve data from the warehouse and save it to files
    """
    elevation_df = get_elevation()
    elevation_df.to_csv(elevation, index=False)

    surface_temperature_df = get_surface_temperature()
    surface_temperature_df.to_csv(surface_temperature, index=False)

    tag_tracks_df = get_tag_tracks()
    tag_tracks_df.to_csv(tag_tracks, index=False)
