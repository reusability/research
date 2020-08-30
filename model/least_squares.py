"""
This module will perform an OLS regression on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 29/8/2020 - created module from template created by Ibrahim.
"""
# import external modules.
import pandas as pd
import statsmodels.api as sm

# import local modules.
from model.base_model import BaseModel

"""
Define the GLS class.
"""
class LeastSquares(BaseModel):
    """
    initialise class instance.
    """
    def __init__(self, data):
        # call parent function.
        BaseModel.__init__(self, data)

        # placeholders specific to this class.
        self.model = None
        self.trained_model = None

        # initialise a statsmodels OLS instance.
        self.model = sm.GLS(self.train_y, self.train_x)

    """
    fit the model with the training data.
    """
    def train(self):
        # call parent function.
        BaseModel.train(self)

        # fit the model.
        self.trained_model = self.model.fit()

        # update the is_trained variable.
        self.is_trained = True

    """
    display coefficient information.
    """
    def describe(self):
        # call parent function.
        BaseModel.describe(self)

        # display coefficient information.
        print(self.trained_model.summary())

    """
    generate test predictions based on the fitted model.
    """
    def test(self):
        # call parent function.
        BaseModel.test(self)

        # call predict method on the statsmodels.OLS object to predict out of
        # sample oversvations.
        self.test_predictions = self.trained_model.predict(self.test_x).astype(int)

        # assess the performance of the predictions.
        self.assess_performance()
