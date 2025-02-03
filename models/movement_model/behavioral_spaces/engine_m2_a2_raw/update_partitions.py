import os

import haven.spark as db

if __name__ == '__main__':
    os.environ['AWS_PROFILE'] = 'admin'
    os.environ['HAVEN_DATABASE'] = 'haven'

    db.register_partitions('movement_model_raw_features_m2_a2', public_internet_access=True)