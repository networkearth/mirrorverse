import os
import pandas as pd
import netCDF4 as nc
import haven.db as db
import numpy as np

from time import time

def read_data(input_path, max_rows=None):
    dataset = nc.Dataset(input_path)

    elevation = dataset["elevation"][:]
    lats = dataset["lat"][:]
    lons = dataset["lon"][:]
    print(elevation.shape)

    lats_flat = []
    lons_flat = []
    elevation_flat = []

    if max_rows:
        expected_parts = np.ceil((len(lats) * len(lons)) / max_rows)
        print(f"Expected parts: {expected_parts}")

    for i in range(len(lats)):
        for j in range(len(lons)):
            lats_flat.append(lats[i])
            lons_flat.append(lons[j])
            elevation_flat.append(elevation[i, j])

            if len(elevation_flat) == max_rows:
                yield pd.DataFrame({
                    "lat": lats_flat,
                    "lon": lons_flat,
                    "elevation": elevation_flat
                })
                lats_flat = []
                lons_flat = []
                elevation_flat = []

    if elevation_flat:
        yield pd.DataFrame({
            "lat": lats_flat,
            "lon": lons_flat,
            "elevation": elevation_flat
        })


if __name__ == '__main__':
    os.environ['HAVEN_DATABASE'] = 'haven'

    for file_path in os.listdir("."):
        if file_path.endswith(".nc"):
            start = time()
            for part, data in enumerate(read_data(file_path, max_rows=1e6)):
                print(f"Processing part {part + 1} of {file_path}")
                data["lon_bin"] = (data["lon"] // 10) * 10
                data["lat_bin"] = (data["lat"] // 10) * 10
                data["file_name"] = file_path
                data["part"] = part

                db.write_data(
                    data, 'elevation_uploads', 
                    ['lon_bin', 'lat_bin', 'file_name', 'part']
                )
                end = time()
                print(f"Time taken: {end - start}")
                start = time()