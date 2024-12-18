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
    add_h3_geoms(geoms, h3_col)
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

def plot_h3_slider(df, value_col, h3_col, slider_col, line_color_col=None, bold_colors=None, zoom=2, center={"lat": 60, "lon": -180}, zmin=None, zmax=None, colorscale="Blues"):
    zmin = zmin if zmin is not None else df[value_col].min()
    zmax = zmax if zmax is not None else df[value_col].max()

    geojson = build_geojson(df, h3_col)
    fig = go.Figure()
    slider_vals = sorted(df[slider_col].unique())
    for slider_val in slider_vals:
        sub_df = df[df[slider_col] == slider_val]
        if line_color_col is not None:
            line_colors = sub_df[line_color_col].to_list()
            line_widths = [3 if col in bold_colors else 1 for col in line_colors]
        else:
            line_colors = ['black' for _ in range(sub_df.shape[0])]
            line_widths = [1 for _ in line_colors]
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=sub_df[h3_col],
            z=sub_df[value_col],
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            visible=False,
            marker_opacity=0.5,
            marker_line=dict(
                color=line_colors,
                width=line_widths
            )
        ))

    fig.data[0].visible = True

    steps = []
    for i, slider_val in enumerate(slider_vals):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(slider_vals)},
                {"title": f"{slider_col}: {slider_val}"},
            ],
            label=f"{slider_val}"
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": f"{slider_col}: "},
        pad={"t": 50, "b": 25, "l": 25},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0}, mapbox=dict(style="carto-positron", zoom=zoom, center = center)
    )
    return fig


def plot_h3_animation(df, value_col, h3_col, slider_col, line_color_col=None, bold_colors=None, zoom=2, center={"lat": 60, "lon": -180}, duration=500, zmin=None, zmax=None, colorscale="Blues"):
    zmin = zmin if zmin is not None else df[value_col].min()
    zmax = zmax if zmax is not None else df[value_col].max()

    geojson = build_geojson(df, h3_col)
    fig = go.Figure()
    slider_vals = sorted(df[slider_col].unique())

    # just need one trace to fill out with each frame
    for slider_val in slider_vals[:1]:
        sub_df = df[df[slider_col] == slider_val]
        if line_color_col is not None:
            line_colors = sub_df[line_color_col].to_list()
            line_widths = [3 if col in bold_colors else 1 for col in line_colors]
        else:
            line_colors = ['black' for _ in range(sub_df.shape[0])]
            line_widths = [1 for _ in line_colors]
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=sub_df[h3_col],
            z=sub_df[value_col],
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            visible=True,
            marker_opacity=0.5,
            marker_line=dict(
                color=line_colors,
                width=line_widths
            )
        ))

    # Create animation frames
    frames = []
    for slider_val in slider_vals:
        sub_df = df[df[slider_col] == slider_val]
        frame = go.Frame(
            data=[
                go.Choroplethmapbox(
                    geojson=geojson,
                    locations=sub_df[h3_col],
                    z=sub_df[value_col],
                    zmin=zmin,
                    zmax=zmax,
                    colorscale=colorscale,
                    marker_opacity=0.5,
                    marker_line=dict(
                        color=sub_df[line_color_col].to_list() if line_color_col else ['black'] * len(sub_df),
                        width=[3 if col in bold_colors else 1 for col in sub_df[line_color_col].to_list()] if line_color_col else [1 for _ in range(len(sub_df))]
                    )
                )
            ],
            traces=[0],
            name=str(slider_val),  # Name the frame according to the slider value
            layout=go.Layout(
                title_text=f"{slider_col}: {slider_val}"  # Set the title text for each frame
            )
        )
        frames.append(frame)

    # Add buttons for play/pause
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": duration, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])]
        )]
    )

    # Set the layout for mapbox and animation
    fig.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0}, 
        mapbox=dict(style="carto-positron", zoom=zoom, center=center)
    )

    # Attach frames to the figure
    fig.frames = frames

    return fig
