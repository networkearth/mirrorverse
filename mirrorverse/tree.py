"""
The Base Tree Class
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score


class DecisionTree(object):
    """
    This class is not complete and requires subclassing
    that implements the following:

    ATTRIBUTES:
    - BUILDERS: list of choice builders
    - BRANCHES: dictionary of DecisionTrees that will
        be called based on the choice made
    - FEATURE_COLUMNS: list of columns that will be used
        by the model to predict the utility of choices
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
        choices = []
        for builder in self.builders:
            choices += [builder(state, choice_state)]
        return pd.concat(choices).reset_index(drop=True)

    def choose(self, state, choice_state):
        choices = self.get_choices(state, choice_state)
        utility = self.model.predict(choices[self.FEATURE_COLUMNS])
        if utility.sum() == 0:
            probs = np.ones(len(utility)) / len(utility)
        else:
            probs = utility / utility.sum()
        choice = choices.iloc[np.random.choice(choices.index, p=probs)]

        self.update_branch(choice, choice_state)

        identifier = self.get_identifier(choice)
        if self.branches.get(identifier) is not None:
            self.branches[identifier].choose(state, choice_state)

    def _build_model_data(self, states, choice_states, selections):
        dataframes = []
        for state, choice_state, selection in zip(states, choice_states, selections):
            choices = self.get_choices(state, choice_state)
            dataframe = self._stitch_selection(choices, selection)
            dataframes.append(dataframe)
        return pd.concat(dataframes)

    def test_model(self, states, choice_states, selections):
        data = self._build_model_data(states, choice_states, selections)
        X = data[self.FEATURE_COLUMNS]
        y = data["selected"]
        y_pred = self.model.predict(X)
        return {"explained_variance": round(explained_variance_score(y, y_pred), 3)}

    def train_model(self, states, choice_states, selections):
        data = self._build_model_data(states, choice_states, selections)
        X = data[self.FEATURE_COLUMNS]
        y = data["selected"]
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=3),
            param_grid=self.PARAM_GRID,
            return_train_score=True,
            cv=self.CV,
            refit=True,
        ).fit(X, y)
        self.model = grid_search.best_estimator_

    def what_state(self):
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
        models = {self.__class__.__name__: self.model}
        if recurse:
            for branch in self.branches.values():
                models.update(branch.export_models())
        return models

    def import_models(self, models, recurse=True):
        self.model = models[self.__class__.__name__]
        if recurse:
            for branch in self.branches.values():
                branch.import_models(models)
