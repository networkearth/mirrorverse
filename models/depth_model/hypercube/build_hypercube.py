import os
import haven.db as db
import pandas as pd
import numpy as np

os.environ['HAVEN_DATABASE'] = 'haven'
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['AWS_PROFILE'] = 'admin'

sql = """
select 
    avg_depth_class,
    std_depth_class,
    avg_period_progress,
    std_period_progress,
    avg_elevation,
    std_elevation,
    avg_month,
    std_month
from 
    depth_model_features_mk3
limit 1
"""
norm_params = db.read_data(sql).to_dict(orient='records')[0]
norm_params

hypercube = (
    pd.DataFrame({'period_progress': np.arange(0, 1.05, 0.05)})
    .merge(
        pd.DataFrame({'month': np.arange(1, 13)}),
        how='cross'
    )
    .merge(
        pd.DataFrame({'elevation': np.power(10, np.arange(2, 4, 0.1))}),
        how='cross'
    )
    .merge(
        pd.DataFrame({'daytime': [0.0,1.0]}),
        how='cross'
    )
)

hypercube['_individual'] = 0
hypercube['_train'] = False
hypercube['_selected'] = False
hypercube = hypercube.reset_index().rename({'index': '_decision'}, axis=1)
hypercube = hypercube.merge(
    pd.DataFrame({'depth_class': [25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0]}),
    how='cross'
)
hypercube = hypercube.reset_index().rename({'index': '_choice'}, axis=1)

for col in ['period_progress', 'month', 'elevation', 'depth_class']:
    hypercube[f'{col}'] = (hypercube[col] - norm_params[f'avg_{col}']) / norm_params[f'std_{col}']
    hypercube[f'avg_{col}'] = norm_params[f'avg_{col}']
    hypercube[f'std_{col}'] = norm_params[f'std_{col}']
print(hypercube.shape)
hypercube.head()

db.write_data(
    hypercube,
    'depth_model_hypercube_mk3',
    ['_individual'],
)