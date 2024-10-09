import json
import h3
import geopandas as gpd
from shapely.geometry import Polygon
import plotly.graph_objects as go

def get_coords(h3_index):
    """
    Input:
    - h3_index (str): A hex string representing the h3 index

    Returns a tuple of (lon, lat) coordinates for the h3 cell
        geometry
    """
    coords = h3.h3_to_geo_boundary(h3_index, True)
    coords = tuple((lon, lat) for lon, lat in coords)
    lons = [lon for lon, _ in coords]
    if max(lons) - min(lons) > 180:
        coords = tuple(
            (lon if lon > 0 else 180 + (180 + lon), lat) for lon, lat in coords
        )
    return coords

def add_h3_geoms(df, h3_col='h3_index'):
    df['geometry'] = df[h3_col].apply(lambda x:  Polygon(get_coords(x)))


def build_geojson(df, h3_col='h3_index'):
    geoms = df[[h3_col]].drop_duplicates()
    add_h3_geoms(geoms)
    geoms = gpd.GeoDataFrame(geoms)
    geoms = json.loads(geoms.to_json())
    for feature in geoms['features']:
        feature['id'] = feature["properties"][h3_col]
    return geoms


def plot_h3(df, value_col, h3_col='h3_index'):
    geojson = build_geojson(df, h3_col)
    fig = go.Figure()
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        locations=df[h3_col],
        z=df[value_col],
        zmin=df[value_col].min(),
        zmax=df[value_col].max(),
        colorscale="Blues",
        visible=True
    ))
    return fig
