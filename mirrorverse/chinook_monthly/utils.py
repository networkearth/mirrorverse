"""
Utility functions for the Chinook salmon model.
"""

import h3
import geopy.distance

MAX_KM = 800
RESOLUTION = 3


def find_neighbors(h3_index, neighbors_index):
    """
    Input:
    - h3_index (str): the H3 index
    - neighbors_index (dict): the neighbors index

    Updates the neighbors index with the neighbors of the H3 index.
    """
    h3_coords = h3.h3_to_geo(h3_index)
    checked = set()
    neighbors = set()
    distance = 1
    found_neighbors = True
    while found_neighbors:
        found_neighbors = False
        candidates = h3.k_ring(h3_index, distance)
        new_candidates = set(candidates) - checked
        for candidate in new_candidates:
            if geopy.distance.geodesic(h3_coords, h3.h3_to_geo(candidate)).km <= MAX_KM:
                neighbors.add(candidate)
                found_neighbors = True
            checked.add(candidate)
        distance += 1
    neighbors_index[h3_index] = neighbors
