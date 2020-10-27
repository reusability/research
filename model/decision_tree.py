"""
This module will perform a trees classification on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 29/8/2020 - created module from template created by Ibrahim.
"""
# import external modules.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Real, Categorical, Integer

# import local modules.
from model.base_model import BaseModel

"""
Define the tree classifier class.
"""
class DecisionTree(BaseModel):
    """
    initialise class instance.
    """
    def __init__(self, data, normalize=False, model=DecisionTreeClassifier()):
        # call parent function.
        BaseModel.__init__(self, data, normalize=normalize)

        # placeholders specific to this class.
        self.model = None

        # Reference to the library used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        # Selecting the most important features using a tress classifer algorithm# initialise a statsmodels OLS instance.
        self.model = model 
        #DecisionTreeClassifier(criterion='gini',max_depth=2,max_features=21,min_samples_leaf=21, min_samples_split=2)


#('criterion', 'gini'), ('max_depth', 6), ('max_features', 3), ('min_samples_leaf', 2), ('min_samples_split', 2)

    def hyperparameter_tuning(self):

        # Defines the parameter search space
        param_space = {
            'max_features': Integer(1, 7),  # integer valued parameter
            'max_depth': Integer(1,7),
            'criterion': Categorical(['gini', 'entropy']),
            'splitter': Categorical(['best','random']),
            'min_samples_split': Integer(2,7),
            'min_samples_leaf': Integer(2,7),
        }

        # Calls the parent function - finds the combination of parameters from the given param_space that 
        # yields the highest score - set to accuracy currently
        BaseModel.hyperparameter_tuning(self, 'Bayesian', param_space)

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

        # call predict method on the sklearn.ExtraTreesClassifier object to
        # predict out of sample oversvations.
        numpy_predictions = self.model.predict(self.test_x).astype(int)
        self.test_predictions = pd.Series(data=numpy_predictions, dtype="int64")

        # assess the performance of the predictions.
        self.assess_performance()


