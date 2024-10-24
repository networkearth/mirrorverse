import os

import pandas as pd
import haven.db as db 

os.environ['HAVEN_DATABASE'] = 'haven'
os.environ['AWS_PROFILE'] = 'admin'

sql = '''
select 
    round(normed_log_npp, 2) as normed_log_npp,
    round(normed_log_mlt, 2) as normed_log_mlt,
    round(normed_distance, 2) as normed_distance,
    avg(log_odds) as log_odds,
    avg(odds) as odds
from 
    movement_model_hypercube_inference_v3_s2
group by 
    1, 2, 3
'''
cached_file_path = 'cached_hypercube.snappy.parquet'
if os.path.exists(cached_file_path):
    data = pd.read_parquet(cached_file_path)
else:
    data = db.read_data(sql)
data.to_parquet(cached_file_path)