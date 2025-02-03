import os

import haven.spark as db

if __name__ == '__main__':
    os.environ['AWS_PROFILE'] = 'admin'
    os.environ['HAVEN_DATABASE'] = 'haven'

    db.register_partitions('movement_model_diffusion_m9_a2_v9_t3', public_internet_access=True)