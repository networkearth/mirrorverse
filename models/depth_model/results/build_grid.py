import plotly.express as px

df = hypercube[hypercube['period_progress'] == 0.5]
df['month'] = df['month'].astype(int)
fig = px.line(
    df, x='elevation', y='probability', color='depth_class', facet_row='daytime', facet_col='month',
    width=3000, height=500, log_x=True
)
fig.show()