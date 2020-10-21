"""
This module will perform a trees classification on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 29/8/2020 - created module from template created by Ibrahim.
"""
# import external modules.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# import local modules.
from model.base_model import BaseModel

"""
Define the tree classifier class.
"""
class TreeClassifier(BaseModel):
    """
    initialise class instance.
    """
    def __init__(self, data, normalize=False):
        # call parent function.

        # placeholders specific to this class.
        self.model = None

        # Reference to the library used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        # Selecting the most important features using a tress classifer algorithm# initialise a statsmodels OLS instance.
        self.model = ExtraTreesClassifier()

    """
    fit the model with the training data.
    """
    def train(self):
        # call parent function.
        BaseModel.train(self)

        # building of a forest of tress based on the the untrained data set
        self.model.fit(self.train_x, self.train_y)

        # update the is_trained variable.
        self.is_trained = True

    """
    display coefficient information.
    """
    def describe(self):
        # call parent function.
        BaseModel.describe(self)

        # uses an inbuilt class feature_importances of tree based classifiers -
        # which selects the most important features based on gini importance/ mean decrease impurity
        # in more laymen terms: along the lines of the less probability/ samples that read that particular node/ variable ->
        # the less important that variable is
        print(self.model.feature_importances_)

        # plot a bar graph of feature importances - selecting all the features
        feat_importances = pd.Series(self.model.feature_importances_, index=self.train_x.columns)
        feat_importances.nlargest(len(self.train_x.columns)).plot(kind='barh')
        plt.show()

    """
    generate test predictions based on the fitted model.
    """
    def test(self):
        # call parent function.
        BaseModel.test(self)

        # predict TRAINING data. convert to pandas series.
        numpy_predictions_train = self.model.predict(self.train_x).flatten().astype(int)
        self.train_predictions = pd.Series(numpy_predictions_train, dtype="int32").clip(lower=0)

        # predict TESTING data. convert to pandas series.
        numpy_predictions_test = self.model.predict(self.test_x).flatten().astype(int)
        self.test_predictions = pd.Series(numpy_predictions_test, dtype="int32").clip(lower=0)

        # assess the performance of the predictions.
        self.assess_performance()
