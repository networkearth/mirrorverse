"""
Utility Training
"""

# pylint: disable=invalid-name

from collections import defaultdict

import numpy as np
from tqdm import tqdm


def get_proposed_utility(dataframe):
    """
    Inputs:
    - dataframe (pd.DataFrame): a dataframe with columns "utility",
        "selected", and "_decision"

    Returns a pd.DataFrame with proposed utility values
    """
    dataframe["sum_utility"] = dataframe.groupby("_decision")["utility"].transform(
        "sum"
    )
    dataframe["probability"] = dataframe["utility"] / dataframe["sum_utility"]
    dataframe["score"] = dataframe["selected"] - dataframe["probability"]
    dataframe["proposed"] = dataframe["utility"] * (1 + dataframe["score"])
    return dataframe


def train_utility_model(model, dataframe, feature_columns, N=1, diagnostics=None):
    """
    Inputs:
    - model: a model object with a "fit" method and a "predict" method
    - dataframe (pd.DataFrame): a dataframe with both feature columns
        and "selected" and "_decision" columns
    - feature_columns (list): a list of column names to use as features
    - N (int): the number of iterations to train the model
    - diagnostics: a list to store diagnostics

    Returns a trained model
    """
    assert "selected" not in feature_columns
    assert "_decision" not in feature_columns

    X = dataframe[feature_columns]
    y = np.ones(X.shape[0])

    model.fit(X, y)
    dataframe["utility"] = model.predict(X)

    diagnostics_results = defaultdict(list)

    for _ in tqdm(range(N)):
        dataframe = get_proposed_utility(dataframe)

        if diagnostics is not None:
            for diagnostic in diagnostics:
                diagnostics_results[diagnostic.__name__].append(
                    diagnostic(dataframe["proposed"], dataframe["utility"])
                )

        y = dataframe["proposed"]
        model.fit(X, y)
        dataframe["utility"] = model.predict(X)

    dataframe = get_proposed_utility(dataframe)

    if diagnostics is not None:
        for diagnostic in diagnostics:
            diagnostics_results[diagnostic.__name__].append(
                diagnostic(dataframe["proposed"], dataframe["utility"])
            )

    return model, diagnostics_results
