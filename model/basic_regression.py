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
    plt.figure(1)
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mse")
    plt.ylim([0, 4000])
    plt.ylabel('MSE')
    
    # Evaluating the model (based on training data) on the test data 
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=0)

    # Prints out the mean squared error
    print("\n Testing set Mean Squared Error: {:5.2f}".format(mse))

    # Predicting the model (produced above) on the untestd - test data set
    test_predictions = model.predict(test_x).flatten()
    plt.figure(2)
    a = plt.axes(aspect='equal')

    # Scatter plot of the predicted ReuseRate data set against the actual ReuseRate data set (as per mavern)
    plt.scatter(test_y, test_predictions)

    # defining axis title
    plt.xlabel('True Value [ReuseRate]')
    plt.ylabel('Predicted Value [ReuseRate]')

    # axis range limits
    lims = [0, 100]

    # plotting the values
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
