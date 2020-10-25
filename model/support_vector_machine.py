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
from skopt.space import Real, Categorical, Integer

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
        self.model = svm.SVC(C=10,gamma=1750,kernel='rbf')

    def hyperparameter_tuning(self):

        # Defines the parameter search space
        param_space = {
                'C': Integer(1,500),  
                'gamma': Real(0.001,30), 
                'kernel': Categorical(['rbf','linear','sigmoid']),
                'decision_function_shape': Categorical(['ovo','ovr'])
            }

        # Calls the parent function - finds the combination of parameters from the given param_space that 
        # yields the highest score - set to accuracy currently
        BaseModel.hyperparameter_tuning(self, 'Bayesian', param_space)

        # Plotting decision region
        #plot_decision_regions(self.train_x, self.train_y, self.model, legend=2)    

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