"""
This module will perform a neural network regression on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 29/8/2020 - created module from old 'basic_regression' module.
"""
# import external modules.
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# import local modules.
from model.base_model import BaseModel
from utils import data_loader

"""
Define the Neural network class.
"""
class NN(BaseModel):
    """
    initialise class instance.
    """
    def __init__(self, data, hidden_layers = [], epochs=100, validation_split=0.2):
        # call parent function.
        BaseModel.__init__(self, data)

        # additional attributes specific to this model.
        self.is_trained = False
        self.epochs = epochs
        self.validation_split = validation_split
        self.normalization_params = data_loader.get_normalization_params(self.test_x)

        # if there are no hidden layers:
        if len(hidden_layers) == 0:
            # setup the NN structure to have 1 layer. ie, a basic regression.
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(1, activation='linear', input_shape=[len(self.train_x.keys())]),
            ])

        # if there are hidden layers.
        else:
            # define the model and the first layer.
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_layers[0], activation='relu', input_shape=[len(self.train_x.keys())]),
            ])

            # for each hidden layer, add aother layer.
            for i in range(1, len(hidden_layers)):
                # define the layer to add.
                self.model.add(tf.keras.layers.Dense(hidden_layers[i], activation='relu'))

            # specify the final layer.
            self.model.add(tf.keras.layers.Dense(1, activation='relu'))

        # compile the model.
        self.model.compile(loss='mse',
            optimizer=tf.keras.optimizers.RMSprop(0.001),
            metrics=['mae', 'mse'])

    """
    fit the model with the training data.
    """
    def train(self):
        # call parent function.
        BaseModel.train(self)

        # get normalized x values.
        normalized_train_x = self.normalize_x(self.train_x)

        # train the model.
        history = self.model.fit(normalized_train_x, self.train_y, epochs=self.epochs, validation_split=self.validation_split, verbose=0,
            callbacks=[tfdocs.modeling.EpochDots()])

        # plot fitting the error function over time to Jupyter Notebook.
        plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
        plotter.plot({'Basic': history}, metric = "mse")
        plt.ylim([0, 4000])
        plt.ylabel('MSE')

        # update the is_trained variable.
        self.is_trained = True

    """
    display coefficient information.
    """
    def describe(self):
        # call parent function.
        BaseModel.describe(self)

        # print structure of NN to screen.
        print(self.model.summary())

        # list the coefficients.
        #print(self.model.get_weights())

    """
    generate test predictions based on the fitted model.
    """
    def test(self):
        # call parent function.
        BaseModel.test(self)

        # get normalized x values.
        normalized_test_x = self.normalize_x(self.test_x)

        # call predict method on the statsmodels.OLS object to predict out of
        # sample oversvations. convert to pandas series.
        numpy_predictions = self.model.predict(normalized_test_x).flatten().astype(int)
        self.test_predictions = pd.Series(numpy_predictions, dtype="int32")

        # assess the performance of the predictions.
        self.assess_performance()

        # Evaluating the model (based on training data) on the test data
        (loss, mae, mse) = self.model.evaluate(normalized_test_x, self.test_y, verbose=0)
        print('loss: %0.d, mae: %0.d, mse: %0.d' % (loss, mae, mse))

    """
    Convert a pandas dataframe of values into normalized values based on the
    normalized params attribute. x_values is a pandas dataframe.
    """
    def normalize_x(self, x_values):
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
