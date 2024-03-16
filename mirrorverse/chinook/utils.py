import numpy as np
import h3
import geopy.distance

# at run time this will need to be filled with
# a dataframe of h3 index, elevation pairs
ELEVATION_ENRICHMENT = None

# at run time this will need to be filled with
# a dataframe of h3 index, month, temp triples
SURFACE_TEMPS_ENRICHMENT = None

# Some Basic Configuration
NEIGHBORS = {}
MAX_KM = 100
RESOLUTION = 4


def find_neighbors(h3_index):
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
    NEIGHBORS[h3_index] = neighbors


def get_heading(lat1, lon1, lat2, lon2):
    x = lon2 - lon1
    y = lat2 - lat1
    if x == 0 and y == 0:
        return np.nan
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += 2 * np.pi
    return angle


def diff_heading(heading1, heading2):
    if heading1 < heading2:
        heading1, heading2 = heading2, heading1

    diff = heading1 - heading2
    return diff if diff <= np.pi else 2 * np.pi - diff
