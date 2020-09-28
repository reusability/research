"""
This module will perform a trees classification on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 29/8/2020 - created module from template created by Ibrahim.
"""
# import external modules.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# import local modules.
from model.base_model import BaseModel

"""
Define the support-vector-machine class.
"""
class SupportVectorMachine(BaseModel):
    """
    initialise class instance.
    """
    def __init__(self, data, normalize=False):
        # call parent function.
        BaseModel.__init__(self, data, normalize=normalize)

        # placeholders specific to this class.
        self.model = None

        # Reference to the library used: https://scikit-learn.org/stable/modules/svm.html
        # Initlising the class
        self.model = svm.SVC()

    """
    fit the model with the training data.
    """
    def train(self):
        # call parent function.
        BaseModel.train(self)

        #trains the support-vector-machine model 
        self.model.fit(self.train_x, self.train_y)

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

        # Using model built with training data, to predict the test data - using the predict method in 
        # sklearn.neighbors.KNeighborsClassifier
        y_pred = self.model.predict(self.test_x)
        self.test_predictions = pd.Series(data=y_pred, dtype="int64")

        # assess the performance of the predictions.
        self.assess_performance()