import os

import haven.spark as db

if __name__ == '__main__':
    os.environ['AWS_PROFILE'] = 'admin'
    os.environ['HAVEN_DATABASE'] = 'haven'

    db.register_partitions('movement_model_simulation_v3_s2', public_internet_access=True)
    db.register_partitions('spark_test_10', public_internet_access=True)