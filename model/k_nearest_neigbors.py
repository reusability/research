"""
This module will perform a trees classification on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 29/8/2020 - created module from template created by Ibrahim.
"""
# import external modules.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# import local modules.
from model.base_model import BaseModel

#from mlxtend.plotting import plot_decision_regions


"""
Define the k-nearest-neighbors class.
"""
class KNearestNeighbors(BaseModel):
    """
    initialise class instance.
    """
    def __init__(self, data, normalize=False):
        # call parent function.
        BaseModel.__init__(self, data, normalize=normalize)

        # placeholders specific to this class.
        self.model = None

        # Reference to the library used: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
        # Initlising the class
        self.model = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 5, weights = 'distance')

    def hyperparameter_tuning(self):

        # Defines the parameter search space
        param_space = {
            'n_neighbors': (1, 100),  # integer valued parameter
            'weights': ['uniform', 'distance'],  # categorical parameter
            'metric': ['euclidean', 'manhattan', 'minkowski'] # categorical parameter
        }

        # Calls the parent function - finds the combination of parameters from the given param_space that 
        # yields the highest score - set to accuracy currently
        BaseModel.hyperparameter_tuning(self, 'Grid', param_space)

        # Plotting decision region
        #plot_decision_regions(self.train_x, self.train_y, self.model, legend=2)

    """
    fit the model with the training data.
    """
    def train(self):
        # call parent function.
        BaseModel.train(self)

        #trains the k-nearest-neighbor model 
        self.model.fit(self.train_x, self.train_y)

        # update the is_trained variable.
        self.is_trained = True

    """
    display coefficient information.
    """
    def describe(self):
        # call parent function.
        BaseModel.describe(self)

        # printing out the values of the hyper-parameter settings
        print(self.model)

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

        """  # idk how to get graph plot of this
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from sklearn import neighbors, datasets

        h = .02  # step size in the mesh

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = self.test_x.min() - 1, self.test_x.max() + 1
        y_min, y_max = self.test_y.min() - 1, self.test_y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(self.test_x, self.test_y, c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification")

        plt.show()
 """
        
