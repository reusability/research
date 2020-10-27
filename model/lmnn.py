"""
This module will perform a trees classification on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 29/8/2020 - created module from template created by Ibrahim.
"""
# import external modules.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from sklearn.neighbors import KNeighborsClassifier
from GPyOpt.methods import BayesianOptimization
from pylmnn import LargeMarginNearestNeighbor as LMNN

# import local modules.
from model.base_model import BaseModel

"""
Define the tree classifier class.
"""
class LMNN(BaseModel):
    """
    initialise class instance.
    """
    def __init__(self, data, normalize=False):
        # call parent function.
        BaseModel.__init__(self, data, normalize=normalize)

        # placeholders specific to this class.
        self.model = None

        # Reference to the library used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        # Selecting the most important features using a tress classifer algorithm# initialise a statsmodels OLS instance.
        self.model = LMNN()
        self.knnmodel = KNeighborsClassifier()


    """
    fit the model with the training data.
    """
    def train(self):
        # call parent function.
        BaseModel.train(self)

        # building of a forest of tress based on the the untrained data set
        self.model.fit(self.train_x, self.train_y)
        self.knnmodel.fit(self.model.transform(self.train_x), self.train_y)

        # update the is_trained variable.
        self.is_trained = True


    """
    display coefficient information.
    """
    def describe(self):
        # call parent function.
        BaseModel.describe(self)

    """
    generate test predictions based on the fitted model.
    """
    def test(self):
        # call parent function.
        BaseModel.test(self)

        # call predict method on the sklearn.ExtraTreesClassifier object to
        # predict out of sample oversvations.
        numpy_predictions = self.knnmodel.predict(self.test_x).astype(int)

        self.test_predictions = pd.Series(data=numpy_predictions, dtype="int64")

        # assess the performance of the predictions.
        self.assess_performance()


