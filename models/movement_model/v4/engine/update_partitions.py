import os

import haven.spark as db

if __name__ == '__main__':
    os.environ['AWS_PROFILE'] = 'admin'
    os.environ['HAVEN_DATABASE'] = 'haven'

    db.register_partitions('spark_test_9', public_internet_access=True)
    db.register_partitions('spark_test_10', public_internet_access=True)