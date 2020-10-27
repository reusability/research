"""
This module will perform an OLS regression on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 23/10/2020 - inherits from RegressionModel instead of BaseModel.
"""
# import external modules.
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math

# import local modules.
from model.regression.regression_model import RegressionModel

"""
Define the GLS class.
"""
class LeastSquares(RegressionModel):
    """
    initialise class instance.
    """
    def __init__(self, data, normalize=False, t_value_threshold=2.3, **kwargs):
        # call parent function.
        RegressionModel.__init__(self, data, normalize=normalize, **kwargs)

        # placeholders specific to this class.
        self.model = None
        self.trained_model = None
        self.t_value_threshold = t_value_threshold

        # initialise a statsmodels OLS instance.
        self.model = sm.GLS(self.train_y, self.train_x)

    """
    fit the model with the training data.
    """
    def train(self):
        # call parent function.
        RegressionModel.train(self)

        # fit the model.
        self.trained_model = self.model.fit()

        # drop all clumns with a 'nan' tvalue.
        for column in self.train_x.columns:
            if str(self.trained_model.tvalues[column]) == 'nan':
                print('removing column: '+column+' due to nan tvalue')
                self.train_x = self.train_x.drop(columns=[column])
                self.test_x = self.test_x.drop(columns=[column])

        # while there are still metrics that have a t_value less than 2.2:
        while not all(abs(t) > self.t_value_threshold for t in self.trained_model.tvalues):
            min_significance = math.inf
            min_column = ''

            # determine the metric with the lowest significance.
            for column in self.train_x.columns:
                if abs(self.trained_model.tvalues[column]) < abs(min_significance):
                    min_column = column
                    min_significance = self.trained_model.tvalues[column]

            # remove this column from the model.
            print('removing column: '+min_column+' due to siginificance: '+str(min_significance))
            self.train_x = self.train_x.drop(columns=[min_column])
            self.test_x = self.test_x.drop(columns=[min_column])

            # retrain and refit the model.
            self.model = sm.GLS(self.train_y, self.train_x)
            self.trained_model = self.model.fit()

        # update the is_trained variable.
        self.is_trained = True

    """
    display coefficient information.
    """
    def describe(self):
        # call parent function.
        RegressionModel.describe(self)

        # display coefficient information.
        print(self.trained_model.summary())

    """
    generate test predictions based on the fitted model.
    """
    def test(self):
        # call parent function.
        RegressionModel.test(self)

        # call predict method on the statsmodels.OLS object to predict out of
        # sample oversvations. if prediction is less than 0, change value to 0.
        self.train_predictions = self.trained_model.predict(self.train_x).astype(int).clip(lower=0)
        self.test_predictions = self.trained_model.predict(self.test_x).astype(int).clip(lower=0)

        # to help with finding outliers.
        #for index, value in self.test_predictions.items():
            #print('index: '+str(index)+' value: '+str(value))
            #print(self.test_x.loc[index, :])

        # assess the performance of the predictions.
        self.assess_performance()
