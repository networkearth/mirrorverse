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

    CLASS METHODS:
    - update_branch(cls, choice, choice_state) - given the
        choice made and the choice_state, update the choice_state
        to reflect the choice made. This choice state will be
        passed to the next branch.
    """

    @classmethod
    def get_choices(cls, state, choice_state):
        choices = []
        for builder in cls.BUILDERS:
            choices += [builder(state, choice_state)]
        return pd.concat(choices).reset_index(drop=True)

    @classmethod
    def choose(cls, state, choice_state):
        choices = cls.get_choices(state, choice_state)
        utility = cls.MODEL.predict(choices[cls.FEATURE_COLUMNS])
        if utility.sum() == 0:
            probs = np.ones(len(utility)) / len(utility)
        else:
            probs = utility / utility.sum()
        choice = choices.iloc[np.random.choice(choices.index, p=probs)]

        cls.update_branch(choice, choice_state)

        identifier = cls.get_identifier(choice)
        if cls.BRANCHES.get(identifier) is not None:
            cls.BRANCHES[identifier].choose(state, choice_state)

    @classmethod
    def _build_model_data(cls, states, choice_states, selections):
        dataframes = []
        for state, choice_state, selection in zip(states, choice_states, selections):
            choices = cls.get_choices(state, choice_state)
            dataframe = cls._stitch_selection(choices, selection)
            dataframes.append(dataframe)
        return pd.concat(dataframes)

    @classmethod
    def test_model(cls, states, choice_states, selections):
        data = cls._build_model_data(states, choice_states, selections)
        X = data[cls.FEATURE_COLUMNS]
        y = data["selected"]
        y_pred = cls.MODEL.predict(X)
        return {"explained_variance": round(explained_variance_score(y, y_pred), 3)}

    @classmethod
    def train_model(cls, states, choice_states, selections):
        data = cls._build_model_data(states, choice_states, selections)
        X = data[cls.FEATURE_COLUMNS]
        y = data["selected"]
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=3),
            param_grid=cls.PARAM_GRID,
            return_train_score=True,
            cv=cls.CV,
            refit=True,
        ).fit(X, y)
        cls.MODEL = grid_search.best_estimator_

    @classmethod
    def what_state(cls):
        state = set()
        choice_state = set()
        for builder in cls.BUILDERS:
            state.update(builder.STATE)
            choice_state.update(builder.CHOICE_STATE)
        for branch in cls.BRANCHES.values():
            branch_state, branch_choice_state = branch.what_state()
            state.update(branch_state)
            choice_state.update(branch_choice_state)
        return state, choice_state

    @classmethod
    def export_models(cls, recurse=True):
        models = {cls.__name__: cls.MODEL}
        if recurse:
            for branch in cls.BRANCHES.values():
                models.update(branch.export_models())
        return models

    @classmethod
    def import_models(cls, models, recurse=True):
        cls.MODEL = models[cls.__name__]
        if recurse:
            for branch in cls.BRANCHES.values():
                branch.import_models(models)
