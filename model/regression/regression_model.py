"""
This modules extends the base class for all machine learning models and implements
functionality that is shared accross all regression models to analyse reusability
rate.

Last updated: MB 23/10/2020 - created module.
"""
# import external libraries.
import matplotlib.pyplot as plt

# import local modules.
from model.base_model import BaseModel

"""
Define the RegressionModel class that extends BaseModel and defines shared
functionality for all regression models.
"""
class RegressionModel(BaseModel):
    """
    initialise class instance.
    """
    def __init__(self, data, **kwargs):
        # call parent function.
        BaseModel.__init__(self, data, **kwargs)

    """
    analyse the performance of the predictions.
    """
    def assess_performance(self):
        # if there is no 'test_predictions' data generated, throw error.
        if self.test_predictions is None:
            raise Exception('Run the `test` function to predict test data.')

        print(">> assessing prediction performance...")

        # TRAINING plot. configure Scatter plot of the predicted ReuseRate data
        # set against the actual ReuseRate data set (as per mavern)
        plt.figure(2)
        plt.axes(aspect='equal')
        plt.title('TRAINING data - predictions vs actual')
        plt.ylabel('True Value [ReuseRate]')
        plt.xlabel('Predicted Value [ReuseRate]')
        lims_train = [min(min(self.train_predictions), min(self.train_y))-1000, max(max(self.train_predictions), max(self.train_y))+1000] # axis limits.
        plt.xlim(lims_train)
        plt.ylim(lims_train)
        plt.plot(lims_train, lims_train)
        plt.scatter(self.train_predictions, self.train_y)

        # print the performance of the predictions on train data.
        MSE = (self.train_y - self.train_predictions).pow(2).mean()
        MAE = abs(self.train_y - self.train_predictions).mean()
        print('train MSE: %.0f' % MSE)
        print('train MAE: %.0f' % MAE)

        # TESTING plot. configure Scatter plot of the predicted ReuseRate data
        # set against the actual ReuseRate data set (as per mavern)
        plt.figure(3)
        plt.axes(aspect='equal')
        plt.title('TESTING data - predictions vs actual')
        plt.ylabel('True Value [ReuseRate]')
        plt.xlabel('Predicted Value [ReuseRate]')
        lims_test = [min(min(self.test_predictions), min(self.test_y))-1000, max(max(self.test_predictions), max(self.test_y))+1000] # axis limits.
        plt.xlim(lims_test)
        plt.ylim(lims_test)
        plt.plot(lims_test, lims_test)
        plt.scatter(self.test_predictions, self.test_y)

        # print the performance of the predictions on test data.
        MSE = (self.test_y - self.test_predictions).pow(2).mean()
        MAE = abs(self.test_y - self.test_predictions).mean()
        print('test MSE: %.0f' % MSE)
        print('test MAE: %.0f' % MAE)

        # create plot of residuals.
        residuals = sorted([list(self.test_predictions)[i] - list(self.test_y)[i] for i in range(len(self.test_predictions))])
        plt.figure(4)
        plt.title('TESTING data - residual distribution')
        plt.hist(residuals, density=True, histtype='stepfilled', alpha=0.1)
