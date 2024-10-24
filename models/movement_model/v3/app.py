import pandas as pd 
import plotly.graph_objects as go

data = pd.read_parquet('cached_hypercube.snappy.parquet')

def build_figure(x_axis, y_axis, feature_map):
    x = x_axis
    y = y_axis

    _filter_features = [
        feature for feature in feature_map.keys()
        if feature not in (x, y)
    ]

    start = _filter_features[0]
    _filter = (data[start] >= feature_map[start] - 0.001) & (data[start] <= feature_map[start] + 0.001)
    for col in _filter_features[1:]:
        _filter &= (data[col] >= feature_map[col] - 0.001) & (data[col] <= feature_map[col] + 0.001)

    df = data[_filter].groupby([x, y])[['odds']].mean().reset_index()
    df = df.pivot(index=x, columns=y, values='odds')

    fig = go.Figure(data=[go.Surface(
        x=df.index,
        y=df.columns,
        z=df.T,
    )])
    fig.update_scenes(xaxis_title_text=x,  
                    yaxis_title_text=y,  
                    zaxis_title_text='odds')
    return fig

from dash import Dash, html, dcc, Input, Output, callback

def build_slider(data, col):
    _min, _max = data[col].min(), data[col].max()
    values = sorted(data[col].unique())
    return html.Div([
        html.H3(col),
        dcc.Slider(
            _min, _max, value=_min,
            marks = {
                value: str(round(value, 2))
                for value in values
            },
            id=f'slider-{col}'
        )
    ])

features = sorted([
    'water_heading', 'movement_heading', 'normed_distance',
    'normed_log_mlt', 'normed_log_npp'
])

sliders = [
    build_slider(data, col)
    for col in features
]

app = Dash()
app.layout = html.Div([
    html.Div([
        html.H3('X-axis'),
        dcc.Dropdown(features, features[0], id='dropdown-x')
    ]),
    html.Div([
        html.H3('Y-axis'),
        dcc.Dropdown(features, features[1], id='dropdown-y')
    ]),
    *sliders,
    html.Div(id='div-figure')
])

def update_figure(*args):
    x_axis, y_axis = args[:2]
    feature_values = args[2:]
    feature_map = {
        feature: value 
        for feature, value in zip(
            features, feature_values
        )
    }
    print(x_axis, y_axis, feature_map)
    return [dcc.Graph(figure=build_figure(x_axis, y_axis, feature_map))]

callback_inputs = [
    Output('div-figure', 'children'),
    Input('dropdown-x', 'value'),
    Input('dropdown-y', 'value'),
] + [
    Input(f'slider-{col}', 'value')
    for col in features
]
callback(
    *callback_inputs
)(
    update_figure
)


if __name__ == '__main__':
    app.run(debug=True)