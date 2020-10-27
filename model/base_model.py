"""
This modules defines the base class for all machine learning models to analyse
reusability rate.

Last updated: MB 29/08/2020 - created module.
"""
# import external libraries.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

# import local modules.
from utils import data_loader

"""
Base class that all models inherit from.
"""
class BaseModel:
    """
    store dataset. data is a dictionary.
    """
    def __init__(self, data, normalize=False, **kwargs):
        print(">> initialising model...")

        # if we are normalizing data, save the normalized x value.
        if normalize is True:
            self.normalization_params = data_loader.get_normalization_params(data['train_x'])
            self.train_x = self.normalize_x(data['train_x'])
            self.test_x = self.normalize_x(data['test_x'])

        # if we are not normalizing data, use regular x values.
        else:
            self.train_x = data['train_x']
            self.test_x = data['test_x']

        # save the y values and other attributes.
        self.train_y = data['train_y']
        self.test_y = data['test_y']
        self.test_predictions = pd.Series()    # placeholder for 'test' function.
        self.train_predictions = pd.Series()    # placeholder for 'test' function.
        self.is_trained = False

    def hyperparameter_tuning(self, type, param_space):

        # definees the type and number of cross validation splits - refer to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html
        # Repeated Stratified K Fold -> This repeats a stratified k fold n number of times
        # Stratified k fold -> Shuffles the data once before splitting into n different parts,
        # where each part is used as a test set
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

        if type == 'Grid':
            # Set all the variables for the grid search cross validation
            search = GridSearchCV(estimator=self.model, param_grid=param_space, cv=cv, scoring='accuracy')

        elif type == 'Bayesian':
            # defines the bayes search cv with parameters - refer to: https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
            # Bayesian optimisation is a type of sequential method in which it learns from each step what the optimal hyper-parameters are
            # (in contrast to grid or random search) - using some complicated maths model (im not sure about)
            search = BayesSearchCV(estimator=self.model, param_grid=param_space, n_jobs=-1, cv=cv)

        # perform the search - i.e. it fits the model on the training data set for the different hyper-parameter settings
        search_result = search.fit(self.train_x, self.train_y)

        # Prints the results - optimal hyper-parameters and the accuracy score
        print("The best parameters are %s with a score of %0.2f"
            % (search_result.best_params_, search_result.best_score_))

        # Displays all of the hyper-parameters combination in descending order of accuracy score
        grid_results = pd.concat([pd.DataFrame(search_result.cv_results_["params"]),pd.DataFrame(search_result.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
        grid_results.sort_values(by=['Accuracy'], inplace=True, ascending=False)
        print(grid_results.head)


    """
    train the model with current train and test XY values saved as attributes.
    """
    def train(self):
        print(">> training model...")

    """
    output a description of the model.
    """
    def describe(self):
        print(">> describing model...")

        # throw an error if model is not trained yet.
        if self.is_trained is False:
            raise Exception('Train model before describing coefficients.')
            return

    """
    generate prdictions for the test_x data.
    """
    def test(self):
        print(">> predicting test data...")

        # throw an error if model is not trained yet.
        if self.is_trained is False:
            raise Exception('Train model before describing coefficients.')
            return

    """
    analyse the performance of the predictions.
    """
    def assess_performance(self):
        # if there is no 'test_predictions' data generated, throw error.
        if self.test_predictions is None:
            raise Exception('Run the `test` function to predict test data.')

        print(">> assessing prediction performance...")

    """
    Convert a pandas dataframe of values into normalized values based on the
    normalized params attribute. x_values is a pandas dataframe.
    """
    def normalize_x(self, x_values):
        # throw an error if this model was not setup to use normalized values.
        if not self.normalization_params:
            raise Exception("This model was not setup to use normalized values.")
            return

        # copy the dataframe.
        normalized_values = pd.DataFrame()

        # iterate over each column and normalize.
        for column in x_values.columns:
            # retrieve normalization parameters.
            mean = self.normalization_params[column]['mean']
            std = self.normalization_params[column]['std']

            # if std is zero, set to 1 to prevent NaNs.
            if std == 0: std = 1

            # save the normalized column.
            normalized_values[column] = (x_values[column] - mean) / std

        # return the normalized dataframe.
        return normalized_values
