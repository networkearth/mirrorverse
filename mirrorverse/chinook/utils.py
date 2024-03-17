"""
Utility functions for the Chinook salmon model.
"""

import numpy as np
import h3
import geopy.distance

MAX_KM = 100
RESOLUTION = 4


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


def get_heading(lat1, lon1, lat2, lon2):
    """
    Input:
    - lat1 (float): the latitude of the first point
    - lon1 (float): the longitude of the first point
    - lat2 (float): the latitude of the second point
    - lon2 (float): the longitude of the second point

    Returns the heading between the two
    points in radians.
    """
    x = lon2 - lon1
    y = lat2 - lat1
    if x == 0 and y == 0:
        return np.nan
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += 2 * np.pi
    return angle


def diff_heading(heading1, heading2):
    """
    Input:
    - heading1 (float): the first heading
    - heading2 (float): the second heading

    Returns the difference between the two headings
    as the smallest angle between them in radians.
    """
    if heading1 < heading2:
        heading1, heading2 = heading2, heading1

    diff = heading1 - heading2
    return diff if diff <= np.pi else 2 * np.pi - diff
