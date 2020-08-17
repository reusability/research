"""
This module will perform a basic linear regression on the 2019 dataset to
determine which metrics are

Last update: MB 12/8/2020 - created module.
"""
# import external modules.
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# import local modules.
from utils import data_loader

"""
This function handles the training of a linear regression to predict the Y
element in each observation.
https://www.tensorflow.org/tutorials/keras/regression
"""
def regression_2019():
    # load the 2019 dataset.
    (train_x, test_x, train_y, test_y) = data_loader.load_2019_dataset()

    # setup the NN structure to have 1 layer. ie, a basic regression.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='linear', input_shape=[len(train_x.keys())]),
    ])

    # compile the model.
    model.compile(loss='mse',
        optimizer=tf.keras.optimizers.RMSprop(0.001),
        metrics=['mae', 'mse'])

    # print structure of NN to screen.
    print(model.summary())

    # train the model.
    history = model.fit(train_x, train_y, epochs=100, validation_split = 0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])

    # plot the error function to Jupyter Notebook.
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mse")
    plt.ylim([0, 4000])
    plt.ylabel('MSE')
