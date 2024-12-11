import os
import hashlib

import haven.db as db
import pandas as pd
import h3
import geopy.distance

def read_data_w_cache(sql):
    if not os.path.exists('cache'):
        os.mkdir('cache')
    cache_entry = os.path.join(
        'cache', 
        hashlib.sha256(sql.encode('utf-8')).hexdigest() + '.snappy.parquet'
    )
    if os.path.exists(cache_entry):
        data = pd.read_parquet(cache_entry)
    else:
        data = db.read_data(sql)
        data.to_parquet(cache_entry)
    return data

def find_neighbors(max_km, h3_index):
    """
    Input:
    - h3_index (str): the H3 index

    Finds all the h3 indices whose centroids are 
    within `max_km`. 
    """
    h3_coords = h3.h3_to_geo(h3_index)
    checked = set([h3_index])
    neighbors = set([h3_index])
    distance = 1
    found_neighbors = True

    while found_neighbors:
        found_neighbors = False
        candidates = h3.k_ring(h3_index, distance)
        new_candidates = set(candidates) - checked
        for candidate in new_candidates:
            if geopy.distance.geodesic(h3_coords, h3.h3_to_geo(candidate)).km <= max_km:
                neighbors.add(candidate)
                found_neighbors = True
            checked.add(candidate)
        distance += 1
    return list(neighbors)
