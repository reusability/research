"""
This modules defines the base class for all machine learning models to analyse
reusability rate.

Last updated: MB 29/08/2020 - created module.
"""
# import external libraries.
import pandas as pd
import matplotlib.pyplot as plt

# import local modules.
from utils import data_loader

"""
Base class that all models inherit from.
"""
class BaseModel:
    """
    store dataset. data is a dictionary.
    """
    def __init__(self, data, normalize=False):
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
        self.is_trained = False

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

        # configure plot.
        plt.figure(2)
        plt.axes(aspect='equal')
        plt.xlabel('True Value [ReuseRate]')
        plt.ylabel('Predicted Value [ReuseRate]')
        lims = [0, 600] # axis limits.
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)

        # Scatter plot of the predicted ReuseRate data set against the actual ReuseRate data set (as per mavern)
        plt.scatter(self.test_y, self.test_predictions)

        # calculate the R-squared value between the prediction and actuals.
        ESS = int(((self.test_predictions - int(self.test_y.mean())).pow(2)).sum())
        RSS = int(((self.test_y - self.test_predictions).pow(2)).sum())
        TSS = int(((self.test_y - int(self.test_y.mean())).pow(2)).sum())

        # the Rsquared value is only valid if TSS = ESS + RSS.
        # (i am using an epsilon of 10000).
        if abs(TSS - ESS - RSS) < 100000:
            print("ESS: "+str(ESS))
            print("RSS: "+str(RSS))
            print("TSS: "+str(TSS))
            r2 = 1 - RSS / TSS
            print("R-squared: %.2f" % r2)

        # calculate the correlation.
        print("correlation: %.2f" % self.test_y.corr(self.test_predictions))

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

            # save the normalized column.
            normalized_values[column] = (x_values[column] - mean) / std

        # return the normalized dataframe.
        return normalized_values
