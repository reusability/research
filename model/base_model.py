"""
This modules defines the base class for all machine learning models to analyse
reusability rate.

Last updated: MB 29/08/2020 - created module.
"""
# import external libraries.
import tensorflow as tf
import matplotlib.pyplot as plt

"""
Base class that all models inherit from.
"""
class BaseModel:
    """
    store dataset. data is a dictionary.
    """
    def __init__(self, data):
        print(">> initialising model...")
        self.train_x = data['train_x']
        self.train_y = data['train_y']
        self.test_x = data['test_x']
        self.test_y = data['test_y']
        self.test_predictions = None    # placeholder for 'test' function.
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
        # https://stackoverflow.com/questions/42351184/how-to-calculate-r2-in-tensorflow
        residual = tf.reduce_sum(tf.square(self.test_y - self.test_predictions))
        total = tf.reduce_sum(tf.square(self.test_y - tf.reduce_mean(self.test_y)))
        print(residual)
        print(total)
        r2 = 1 - tf.divide(residual, total)
        print("R-squared: %.2f" % r2.numpy())
