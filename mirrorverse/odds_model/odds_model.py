"""
The Odds Model
"""

# pylint: disable=invalid-name

import pandas as pd
import numpy as np
from tqdm import tqdm


def get_central_likelihood(dataframe):
    """
    Inputs:
    - dataframe (pd.DataFrame): a dataframe with columns "_selected" and "probability"

    Returns:
    - float: the likelihood of the "_selected" given the "probability" column taken
        to the Nth root, where N is the number of rows in the dataframe
    """
    return np.exp(np.mean(np.log(dataframe[dataframe["_selected"]]["probability"])))


def get_probability(dataframe):
    """
    Inputs:
    - dataframe (pd.DataFrame): a dataframe with columns "utility"
        and "_decision"

    Adds/modifies the following columns to the dataframe:
    - "sum_utility": the sum of the "utility" column for each decision
    - "probability": the probability of selecting each decision
    """
    dataframe["sum_utility"] = dataframe.groupby("_decision")["utility"].transform(
        "sum"
    )
    dataframe["probability"] = dataframe["utility"] / dataframe["sum_utility"]


def get_proposed_utility(dataframe, learning_rate):
    """
    Inputs:
    - dataframe (pd.DataFrame): a dataframe with columns "utility",
        "_selected", "sum_utility", "probability",
    - learning_rate (float): a float between 0 and 1 that determines
        how much the proposed utility values should be adjusted

    Adds/modifies the following columns to the dataframe:
    - "partial": the partial derivative of the probability with respect to the utility
    - "step": the maximum step that can be taken without causing the probability to be negative
    - "proposed": the proposed utility for each decision
    """
    assert 0 < learning_rate < 1, "learning_rate must be between 0 and 1"

    # calculate the partial derivative of the probability with respect to the utility
    dataframe["partial"] = (
        1
        / dataframe["sum_utility"]
        * (
            dataframe["_selected"]
            * ((1 - dataframe["probability"]) / dataframe["probability"])
            - (1 - dataframe["_selected"])
        )
    )

    # this is the maximum step that can be taken without causing the probability to be negative
    worst_case = (dataframe["partial"] / dataframe["utility"]).min()
    factor = (1 / abs(worst_case)) * learning_rate
    dataframe["step"] = dataframe["partial"] * factor

    # propose new utility values
    dataframe["proposed"] = dataframe["utility"] + dataframe["step"]


class OddsModel:
    """
    The Odds Model
    """

    def __init__(self, model):
        self.model = model
        self.diagnostics = None

    def fit(self, X_train, X_test, learning_rate, iterations):
        """
        Inputs:
        - X_train (pd.DataFrame): a dataframe with columns "_selected" and "_decision"
        - X_test (pd.DataFrame): a dataframe with columns "_selected" and "_decision"
        - learning_rate (float): a float between 0 and 1 that determines
            how much the proposed utility values should be adjusted
        - iterations (int): the number of iterations to run

        Fits an odds model to the training data.
        """
        assert (
            "_selected" in X_train.columns and "_selected" in X_test.columns
        ), "X must have a '_selected' column"
        assert (
            "_decision" in X_train.columns and "_decision" in X_test.columns
        ), "X must have a '_decision' column"
        assert set(X_train.columns) == set(
            X_test.columns
        ), "X_train and X_test must have the same columns"

        features = [c for c in X_train.columns if c not in ["_selected", "_decision"]]

        # initialize the diagnostics
        diagnostics = []

        # setup an initial guess
        X_train["utility"] = 1
        X_test["utility"] = 1

        get_probability(X_train)
        get_probability(X_test)
        get_proposed_utility(X_train, learning_rate)
        get_proposed_utility(X_test, learning_rate)

        diagnostics.append(
            {
                "iteration": 0,
                "train_likelihood": get_central_likelihood(X_train),
                "test_likelihood": get_central_likelihood(X_test),
            }
        )

        for i in tqdm(range(1, iterations + 1)):
            self.model.fit(X_train[features], X_train["proposed"])

            X_train["utility"] = self.model.predict(X_train[features])
            X_test["utility"] = self.model.predict(X_test[features])

            get_probability(X_train)
            get_probability(X_test)
            get_proposed_utility(X_train, learning_rate)
            get_proposed_utility(X_test, learning_rate)

            diagnostics.append(
                {
                    "iteration": i,
                    "train_likelihood": get_central_likelihood(X_train),
                    "test_likelihood": get_central_likelihood(X_test),
                }
            )

        X_train.drop(
            columns=[
                "utility",
                "sum_utility",
                "probability",
                "partial",
                "step",
                "proposed",
            ],
            inplace=True,
        )
        X_test.drop(
            columns=[
                "utility",
                "sum_utility",
                "probability",
                "partial",
                "step",
                "proposed",
            ],
            inplace=True,
        )

        self.diagnostics = pd.DataFrame(diagnostics)

    def predict(self, X):
        """
        Inputs:
        - X (pd.DataFrame): a dataframe with a "_decision" column

        Adds a utility and probability column to the dataframe.
        """
        assert "_decision" in X.columns, "X must have a '_decision' column"

        features = [c for c in X.columns if c not in ["_decision", "_selected"]]

        X["utility"] = self.model.predict(X[features])
        get_probability(X)
        X.drop(columns=["sum_utility"], inplace=True)
