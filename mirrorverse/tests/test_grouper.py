"""
Grouper Tests
"""

# pylint: disable=missing-function-docstring

import pandas as pd
import numpy as np

from mirrorverse.grouper import (
    Individual,
    create_individual,
    mutate,
    crossover,
    get_central_likelihoods,
    evaluate,
)


def test_clone():
    individual = Individual(np.zeros(10000))
    individual.fitness = 0.1
    clone = individual.clone()
    assert len(clone) == 10000
    assert clone.fitness is None
    assert individual.fitness == 0.1


def test_create_individual():
    identifiers = [1, 2, 3, 4, 5]
    groups = [1, 2, 3]
    individual = create_individual(identifiers, groups)
    assert len(individual) == len(identifiers)
    assert set(individual) - set(groups) == set()


def test_mutate():
    individual = Individual(np.zeros(10000))
    individual.fitness = 0.1
    groups = [0, 1]
    mutation_rate = 0.2
    mutate(individual, groups, mutation_rate)
    assert len(individual) == 10000
    assert set(individual) - set(groups) == set()
    # note that half of the mutations will be
    # still zeros
    assert abs(np.mean(individual) - 0.1) < 0.01
    assert individual.fitness is None


def test_crossover():
    ind1 = Individual(np.zeros(10000))
    ind1.fitness = 0.1
    ind2 = Individual(np.ones(10000))
    ind2.fitness = 0.2
    crossover_rate = 0.2
    crossover(ind1, ind2, crossover_rate)
    assert len(ind1) == 10000
    assert len(ind2) == 10000
    assert set(ind1) - set([0, 1]) == set()
    assert set(ind2) - set([0, 1]) == set()
    assert abs(np.mean(ind1) - 0.2) < 0.02
    assert abs(np.mean(ind2) - 0.8) < 0.02
    assert ind1.fitness is None
    assert ind2.fitness is None


def test_get_central_likelihoods():
    groups = [1, 1, 2, 2, 3]
    identifiers = [1, 2, 3, 3, 4]
    data = {
        "_identifier": [1, 2, 3, 3, 4],
        "_selected": [False, True, False, True, True],
        "probability": [0, 1, 0.0, 0.5, 0.25],
    }
    data = pd.DataFrame(data)
    likelihoods, counts = get_central_likelihoods(groups, identifiers, data)

    assert likelihoods == {1: 1.0, 2: 0.5, 3: 0.25}
    assert counts == {1: 2, 2: 1, 3: 1}


def test_evaulate():
    groups = [1, 1, 2, 2, 3]
    identifiers = [1, 2, 3, 3, 4]
    data = {
        "_identifier": [1, 2, 3, 3, 4],
        "_selected": [False, True, False, True, True],
        "probability": [0, 1, 0.0, 0.5, 0.25],
    }
    data = pd.DataFrame(data)
    result = evaluate(groups, identifiers, data, 0.5)
    expected_result = ((1 - 0.5) * 2 + (0.5 - 0.5) * 1 + (0.5 - 0.25) * 1) / 4
    assert result == expected_result
