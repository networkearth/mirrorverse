"""
Grouper - to help us group data by divergence
or conformance with a utility model.
"""

import os
import random
from multiprocessing import Pool

import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score
from tqdm import tqdm


# pylint: disable=missing-class-docstring, missing-function-docstring
class Individual(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fitness = None

    def invalidate(self):
        self.fitness = None

    def clone(self):
        return Individual(self)


def create_individual(identifiers, groups):
    """
    Inputs:
    - identifiers (list): list of identifiers
    - groups (list): list of groups

    Returns a list of which group each identifier belongs to.
    """
    return Individual([np.random.choice(groups) for _ in identifiers])


def mutate(individual, groups, mutation_gene_rate):
    """
    Inputs:
    - individual (list): the individual to mutate
    - groups (list): the groups to choose from
    - mutation_gene_rate (float): the rate of individual gene
        mutation

    Mutates the individual by changing one of its elements.
    """
    individual.invalidate()
    for i, _ in enumerate(individual):
        if np.random.random() < mutation_gene_rate:
            individual[i] = np.random.choice(groups)


def crossover(ind1, ind2, crossover_gene_rate):
    """
    Inputs:
    - ind1 (list): the first individual
    - ind2 (list): the second individual
    - crossover_gene_rate (float): the rate of crossover

    Crosses over the two individuals by swapping elements.
    """
    ind1.invalidate()
    ind2.invalidate()
    for i, _ in enumerate(ind1):
        if np.random.random() < crossover_gene_rate:
            ind1[i], ind2[i] = ind2[i], ind1[i]


def get_explained_variances(groups, identifiers, data):
    """
    Inputs:
    - groups (list): the groups
    - identifiers (list): the identifiers
    - data (pd.DataFrame): the data to evaluate over
        should have columns "_identifier", "selected",
        and "probability"

    Returns the explained variances of the groups and the
    counts of the groups.

    dict, dict
    """
    groups = pd.DataFrame({"_identifier": identifiers, "group": groups})
    merged_data = data.merge(groups, on="_identifier")
    variances = {}
    counts = {}
    for group in merged_data["group"].unique():
        filtered_data = merged_data[merged_data["group"] == group]
        variances[group] = explained_variance_score(
            filtered_data["selected"], filtered_data["probability"]
        )
        counts[group] = filtered_data["_identifier"].nunique()
    return variances, counts


def evaluate(individual, identifiers, data, explained_variance):
    """
    Inputs:
    - individual (list): the individual to evaluate
    - identifiers (list): the identifiers
    - data (pd.DataFrame): the data to evaluate over
        should have columns "_identifier", "selected",
        and "probability"
    - explained_variance (float): the explained variance
        of the whole dataset

    Returns the average difference between the explained
    variance of the groups and the whole dataset.
    """
    variances, counts = get_explained_variances(individual, identifiers, data)
    differences, counts_array = [], []
    for group, variance in variances.items():
        differences.append(np.abs(variance - explained_variance))
        counts_array.append(counts[group])
    result = np.sum(np.array(differences) * np.array(counts_array)) / np.sum(
        counts_array
    )
    return result


def tournament(population, tournament_size):
    """
    Inputs:
    - population (list): the population
    - tournament_size (int): the size of the tournament

    Returns the winner of the tournament.
    """
    tournament_ = random.sample(population, tournament_size)
    return max(tournament_, key=lambda x: x.fitness)


def tournament_selection(population, tournament_size, num_individuals):
    """
    Inputs:
    - population (list): the population
    - tournament_size (int): the size of the tournaments
    - num_individuals (int): the number of individuals
        to select

    Returns a list of selected individuals.
    """
    return [tournament(population, tournament_size) for _ in range(num_individuals)]


def build_statistics(generation, population):
    """
    Inputs:
    - generation (int): the generation number
    - population (list): the population

    Returns a dictionary of statistics about the
    population.
    """
    fitnesses = [individual.fitness for individual in population]
    return {
        "generation": generation,
        "mean_fitness": np.mean(fitnesses),
        "std_fitness": np.std(fitnesses),
        "max_fitness": np.max(fitnesses),
        "min_fitness": np.min(fitnesses),
    }


# pylint: disable=missing-function-docstring
def run(args):
    return evaluate(*args)


def group_data(
    data,
    num_groups,
    num_individuals=75,
    num_generations=300,
    crossover_rate=0.2,
    crossover_gene_rate=0.3,
    mutation_rate=0.2,
    mutation_gene_rate=0.02,
    num_elites=10,
    tournament_size=3,
):
    """
    Inputs:
    - data (pd.DataFrame): the data to group should
        have columns "_identifier", "selected", and
        "probability"
    - num_groups (int): the number of groups to divide
        the data into
    - num_individuals (int): the number of individuals
        in the population
    - num_generations (int): the number of generations
        to run
    - crossover_rate (float): the rate of crossover
    - crossover_gene_rate (float): the rate of gene
        crossover
    - mutation_rate (float): the rate of mutation
    - mutation_gene_rate (float): the rate of gene
        mutation
    - num_elites (int): the number of elites to keep
    - tournament_size (int): the size of the tournaments

    Returns:
    - a dataframe with a mapping of identifiers to
        groups and the attendant explained variance
    - a dataframe of statistics about the genetic
        algorithm
    """

    groups = list(range(num_groups))
    identifiers = data["_identifier"].unique()
    explained_variance = explained_variance_score(data["selected"], data["probability"])

    population = [
        create_individual(identifiers, groups) for _ in range(num_individuals)
    ]
    statistics = []

    with Pool(os.cpu_count() - 2) as p:
        fitnesses = p.map(
            run,
            [
                (individual, identifiers, data, explained_variance)
                for individual in population
            ],
        )
        for individual, fitness in zip(population, fitnesses):
            individual.fitness = fitness

        statistics.append(build_statistics(-1, population))

        for generation in tqdm(range(num_generations)):
            elites = sorted(population, key=lambda x: x.fitness, reverse=True)[
                :num_elites
            ]
            elites = [elite.clone() for elite in elites]

            new_population = tournament_selection(
                population, tournament_size, num_individuals - num_elites
            )

            for ind1, ind2 in zip(new_population[::2], new_population[1::2]):
                if np.random.random() < crossover_rate:
                    crossover(ind1, ind2, crossover_gene_rate)

            for individual in new_population:
                if np.random.random() < mutation_rate:
                    mutate(individual, groups, mutation_gene_rate)

            population = elites + new_population
            invalid_ind = [ind for ind in population if ind.fitness is None]
            fitnesses = p.map(
                eval,
                [
                    (individual, identifiers, data, explained_variance)
                    for individual in invalid_ind
                ],
            )
            for individual, fitness in zip(invalid_ind, fitnesses):
                individual.fitness = fitness

            statistics.append(build_statistics(generation, population))

    best_individual = max(population, key=lambda x: x.fitness)
    explained_variances, _ = get_explained_variances(best_individual, identifiers, data)
    groupings = pd.DataFrame(
        [
            {
                "_identifier": identifier,
                "group": group,
                "explained_variance": explained_variances[group],
            }
            for group, identifier in zip(best_individual, identifiers)
        ]
    )
    return groupings, pd.DataFrame(statistics)
