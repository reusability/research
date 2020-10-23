"""
This module will perform a neural network regression on a dataset to attempt to
predict the reuse rate of classes.

Last update: MB 23/10/2020 - inherits from RegressionModel instead of BaseModel.
"""
# import external modules.
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import tensorflow_model_optimization as tfmot

# import local modules.
from model.regression.regression_model import RegressionModel
"""
Define the Neural network class.
"""
class NN(RegressionModel):
    """
    initialise class instance.
    """
    def __init__(self, data, hidden_layers = [], epochs=2000, validation_split=0.2, normalize=True, **kwargs):
        # call parent function.
        RegressionModel.__init__(self, data, normalize=normalize, **kwargs)

        # additional attributes specific to this model.
        self.is_trained = False
        self.epochs = epochs
        self.validation_split = validation_split

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

        # setup tensorflow object to remove neurons with low magnitude.
        # https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        end_step = np.ceil(300 / 32).astype(np.int32) * epochs

        # Define pruning configuration.
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                final_sparsity=0.80, begin_step=0, end_step=end_step)
            }

        # add the pruning to the model.
        self.model = prune_low_magnitude(self.model, **pruning_params)

        # compile the model.
        self.model.compile(loss='mse',
            optimizer=tf.keras.optimizers.RMSprop(0.05),
            metrics=['mae', 'mse'])

    """
    fit the model with the training data.
    """
    def train(self):
        # call parent function.
        RegressionModel.train(self)

        # define what happens at the end of each set.
        callbacks = [
          tfmot.sparsity.keras.UpdatePruningStep(),
          tfdocs.modeling.EpochDots()
        ]


        # train the model.
        history = self.model.fit(self.train_x, self.train_y, epochs=self.epochs, validation_split=self.validation_split, verbose=0,
            callbacks=callbacks)

        # plot fitting the error function over time to Jupyter Notebook.
        plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
        plotter.plot({'Basic': history}, metric = "mse")
        plt.ylim([0, 5000000])
        plt.ylabel('MSE')

        # update the is_trained variable.
        self.is_trained = True

    """
    display coefficient information.
    """
    def describe(self):
        # call parent function.
        RegressionModel.describe(self)

        # print structure of NN to screen.
        print(self.model.summary())

        # print the weights for the first layer of the neural network.
        for i in range(len(self.test_x.columns)):
            # if weight is 0, move to next weight.
            if self.model.get_weights()[0][i][0] == 0:
                continue

            # if weight is non-zero, print to screen.
            print(self.test_x.columns[i]+': '+str(self.model.get_weights()[0][i][0]))

    """
    generate test predictions based on the fitted model.
    """
    def test(self):
        # call parent function.
        RegressionModel.test(self)

        # predict TRAINING data. convert to pandas series.
        numpy_predictions_train = self.model.predict(self.train_x).flatten().astype(int)
        self.train_predictions = pd.Series(numpy_predictions_train, dtype="int32").clip(lower=0)

        # predict TESTING data. convert to pandas series.
        numpy_predictions_test = self.model.predict(self.test_x).flatten().astype(int)
        self.test_predictions = pd.Series(numpy_predictions_test, dtype="int32").clip(lower=0)

        # assess the performance of the predictions.
        self.assess_performance()
