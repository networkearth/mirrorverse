"""
The Base Tree Class
"""

import os
import json

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score

from mirrorverse.utility import train_utility_model, get_proposed_utility


# pylint: disable=no-member, invalid-name, attribute-defined-outside-init
class DecisionTree:
    """
    This class is not complete and requires subclassing
    that implements the following:

    ATTRIBUTES:
    - BUILDERS: list of choice builders
    - BRANCHES: dictionary of DecisionTrees that will
        be called based on the choice made
    - FEATURE_COLUMNS: list of columns that will be used
        by the model to predict the utility of choices
    - OUTCOMES: a list of keys that will be added
        to the choice_state to track the outcome of the
        choice made
    - PARAM_GRID: dictionary of hyperparameters to be
        searched over in the grid search
    - CV: KFold object to use in the grid search

    STATIC METHODS:
    - get_identifier(choice) - given the chosen vector,
        return a unique identifier for the branch to be
        called next (indexes into BRANCHES)
    - _stich_selection(choices, selection) - given the
        choices built from a state and the selection passed
        to _build_model_data, return a dataframe with a
        column "selected" that is 1 for the choice selected
        and 0 for all other choices
    - update_branch(self, choice, choice_state) - given the
        choice made and the choice_state, update the choice_state
        to reflect the choice made. This choice state will be
        passed to the next branch.
    """

    def __init__(self, enrichment):
        self.builders = [builder(enrichment) for builder in self.BUILDERS]
        self.branches = {
            identifier: branch(enrichment)
            for identifier, branch in self.BRANCHES.items()
        }

    def get_choices(self, state, choice_state):
        """
        Input:
        - state (dict): dictionary of state variables
        - choice_state (dict): dictionary of choice state variables

        Builds the full set of choices given the
        state, choice_state, and builders
        """
        choices = []
        for builder in self.builders:
            choices += [builder(state, choice_state)]
        return pd.concat(choices).reset_index(drop=True)

    def choose(self, state, choice_state):
        """
        Input:
        - state (dict): dictionary of state variables
        - choice_state (dict): dictionary of choice state variables

        Uses the state and choice_state to make a choice
        and update the choice_state if necessary.

        Also calls the next branch if applicable.
        """
        choices = self.get_choices(state, choice_state)
        utility = self.model.predict(choices[self.FEATURE_COLUMNS])
        if utility.sum() == 0:
            probs = np.ones(len(utility)) / len(utility)
        else:
            probs = utility / utility.sum()
        choice = choices.iloc[np.random.choice(choices.index, p=probs)]

        self.update_branch(choice, choice_state)
        for outcome in self.OUTCOMES:
            assert outcome in choice_state

        identifier = self.get_identifier(choice)
        if self.branches.get(identifier) is not None:
            self.branches[identifier].choose(state, choice_state)

    def _build_model_data(
        self, states, choice_states, selections, identifiers, quiet=False
    ):
        """
        Input:
        - states (list): list of dictionaries of state variables
        - choice_states (list): list of dictionaries of choice state variables
        - selections (list): list of selection designations
        - identifiers (list): list of identifiers used to group decisions
            together
        - quiet (boolean): defaults to False - if True, suppresses
            warnings of no selection and instead just doesn't pass
            that data to the model

        Uses the builders to build choices and then
        stitches the selections to the choices to build
        the model data
        """
        dataframes = []
        for i, (state, choice_state, selection, identifier) in enumerate(
            zip(states, choice_states, selections, identifiers)
        ):
            choices = self.get_choices(state, choice_state)
            dataframe = self._stitch_selection(choices, selection)

            num_selected = dataframe["selected"].sum()
            if not quiet:
                assert num_selected == 1
            else:
                if num_selected != 1:
                    continue

            dataframe["_decision"] = i
            dataframe["_identifier"] = identifier
            dataframes.append(dataframe)
        return pd.concat(dataframes)

    def test_model(self, states, choice_states, selections, identifiers, quiet=False):
        """
        Input:
        - states (list): list of dictionaries of state variables
        - choice_states (list): list of dictionaries of choice state variables
        - selections (list): list of selection designations
        - identifiers (list): list of identifiers used to group decisions
            together
        - quiet (boolean): defaults to False - if True, suppresses
            warnings of no selection and instead just doesn't pass
            that data to the model

        Evaluate the model on the given states, choice_states,
        and selections. Returns a dictionary of metrics.
        """
        data = self._build_model_data(
            states, choice_states, selections, identifiers, quiet
        )
        X = data[self.FEATURE_COLUMNS]
        y = data["selected"]
        data["utility"] = self.model.predict(X)
        data = get_proposed_utility(data)
        return {
            "explained_variance": round(
                explained_variance_score(y, data["probability"]), 3
            )
        }

    def train_model(
        self,
        states,
        choice_states,
        selections,
        identifiers,
        N=1,
        diagnostics=None,
        learning_rate=31 / 32,
        quiet=False,
    ):
        """
        Input:
        - states (list): list of dictionaries of state variables
        - choice_states (list): list of dictionaries of choice state variables
        - selections (list): list of selection designations
        - identifiers (list): list of identifiers used to group decisions
            together
        - N (int): the number of iterations to train the model
        - diagnostics (list): a list of diagnostic functions to run
        - learning_rate (float): maximum abs score
        - quiet (boolean): defaults to False - if True, suppresses
            warnings of no selection and instead just doesn't pass
            that data to the model

        Train a utility model on the given states, choice_states,
        and selections.
        """
        data = self._build_model_data(
            states, choice_states, selections, identifiers, quiet
        )
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(
                bootstrap=False, n_jobs=(os.cpu_count() - 2)
            ),
            param_grid=self.PARAM_GRID,
            return_train_score=True,
            cv=self.CV,
            refit=True,
        )
        grid_search, diagnostics_results = train_utility_model(
            grid_search,
            data,
            self.FEATURE_COLUMNS,
            N,
            diagnostics,
            learning_rate=learning_rate,
        )
        if diagnostics is not None:
            with open(f"{self.__class__.__name___}.json", "w") as fh:
                json.dump(diagnostics_results, fh, indent=4, sort_keys=True)
        self.model = grid_search.best_estimator_

    def what_state(self):
        """
        Returns the state and choice state variables
        that are used by this tree and its branches
        """
        state = set()
        choice_state = set()
        for builder in self.builders:
            state.update(builder.STATE)
            choice_state.update(builder.CHOICE_STATE)
        for branch in self.branches.values():
            branch_state, branch_choice_state = branch.what_state()
            state.update(branch_state)
            choice_state.update(branch_choice_state)
        return state, choice_state

    def export_models(self, recurse=True):
        """
        Input:
        - recurse (boolean): defaults to True

        Returns a dictionary of the models used by this
        tree and its branches if recurse is True. Otherwise
        just returns the model used by this tree.
        """
        models = {self.__class__.__name__: self.model}
        if recurse:
            for branch in self.branches.values():
                models.update(branch.export_models())
        return models

    def import_models(self, models, recurse=True):
        """
        Input:
        - models (dict): dictionary of models
        - recurse (boolean): defaults to True

        Imports the models into this tree and its branches
        if recurse is True. Otherwise just imports the model
        into this tree.
        """
        self.model = models[self.__class__.__name__]
        if recurse:
            for branch in self.branches.values():
                branch.import_models(models)
