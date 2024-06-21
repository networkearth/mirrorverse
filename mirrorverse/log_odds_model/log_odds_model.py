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
    - dataframe (pd.DataFrame): a dataframe with columns "log_odds"
        and "_decision"

    Adds/modifies the following columns to the dataframe:
    - "odds": the odds for of selecting each decision
    - "sum_odds": the sum of the "odds" column for each decision
    - "probability": the probability of selecting each decision
    """
    dataframe["odds"] = np.exp(dataframe["log_odds"])
    dataframe["sum_odds"] = dataframe.groupby("_decision")["odds"].transform("sum")
    dataframe["probability"] = dataframe["odds"] / dataframe["sum_odds"]


def get_proposed_log_odds(dataframe, learning_rate):
    """
    Inputs:
    - dataframe (pd.DataFrame): a dataframe with columns "log_odds",
        "odds", "_selected", "sum_odds", "probability",
    - learning_rate (float): a float between that determines
        how much the proposed log odds values should be adjusted

    Adds/modifies the following columns to the dataframe:
    - "partial": the partial derivative of the probability with respect to the log odds
    - "step": the step that will be taken in the direction of the gradient
    - "proposed": the proposed log odds for each decision
    """

    # calculate the partial derivative of the probability with respect to the utility
    # dataframe["partial"] = (
    #    1
    #    / dataframe["sum_odds"]
    #    * (
    #        dataframe["_selected"]
    #        * ((1 - dataframe["probability"]) / dataframe["probability"])
    #        - (1 - dataframe["_selected"])
    #    )
    #    * dataframe["odds"]
    # )

    dataframe["partial"] = dataframe["_selected"] * (1 - dataframe["probability"]) - (
        (1 - dataframe["_selected"]) * dataframe["probability"]
    )

    dataframe["step"] = dataframe["partial"] * learning_rate

    # propose new utility values
    dataframe["proposed"] = dataframe["log_odds"] + dataframe["step"]


class LogOddsModel:
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
        X_train["log_odds"] = 0
        X_test["log_odds"] = 0

        get_probability(X_train)
        get_probability(X_test)
        get_proposed_log_odds(X_train, learning_rate)
        get_proposed_log_odds(X_test, learning_rate)

        diagnostics.append(
            {
                "iteration": 0,
                "train_likelihood": get_central_likelihood(X_train),
                "test_likelihood": get_central_likelihood(X_test),
            }
        )

        for i in tqdm(range(1, iterations + 1)):
            self.model.fit(X_train[features], X_train["proposed"])

            X_train["log_odds"] = self.model.predict(X_train[features])
            X_test["log_odds"] = self.model.predict(X_test[features])

            get_probability(X_train)
            get_probability(X_test)
            get_proposed_log_odds(X_train, learning_rate)
            get_proposed_log_odds(X_test, learning_rate)

            diagnostics.append(
                {
                    "iteration": i,
                    "train_likelihood": get_central_likelihood(X_train),
                    "test_likelihood": get_central_likelihood(X_test),
                }
            )

        X_train.drop(
            columns=[
                "log_odds",
                "odds",
                "sum_odds",
                "probability",
                "partial",
                "step",
                "proposed",
            ],
            inplace=True,
        )
        X_test.drop(
            columns=[
                "log_odds",
                "odds",
                "sum_odds",
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

        X["log_odds"] = self.model.predict(X[features])
        get_probability(X)
        X.drop(columns=["sum_odds"], inplace=True)
