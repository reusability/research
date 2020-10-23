"""
This module is the entrypoint for the project. It will handle the order of
task execution of training and analysing models.

Last updated: MB 20/10/2020 - add command line capabilities.
"""
# import external libraries.
import sys

# import local modules.
from utils import data_loader, pre_training_analysis_tools

# import models.
from model.regression.least_squares import LeastSquares
from model.regression.tree_regression import TreeRegression
from model.regression.nerual_net import NN
from model.k_nearest_neigbors import KNearestNeighbors

"""
This function will display a correlation heatmap and scatterplots to assess
multicollinearity and performance of metrics against the reuse rate.
"""
def describe_data():
    data = data_loader.load_2020_dataset()

    #pre_training_analysis_tools.display_correlation_heatmap(data['test_x'])
    #pre_training_analysis_tools.display_covariance_heatmap(data['test_x'])


    pre_training_analysis_tools.display_correlation_scatterplots_xy(data['train_x'], data['train_y'])
    #pre_training_analysis_tools.display_correlation_scatterplots_x(data['test_x'])

"""
This is the main function that is run. read a list of keyword arguments to pass
to data_loader and models.
"""
def main(**kwargs):
    # print the keyword arguments.
    print(kwargs)

    # load data. pass through 'constant' and 'squared' keywords.
    data = data_loader.load_2020_dataset(**kwargs)

    # if the model was not specified, throw an error.
    if 'model' not in kwargs: raise Exception('please include the model keyword argument...')

    # switch based on the model keyword. pass through model keywords.
    elif kwargs['model'] == 'NN': model = NN(data, **kwargs)
    elif kwargs['model'] == 'LeastSquares': model = LeastSquares(data, **kwargs)
    elif kwargs['model'] == 'TreeRegression': model = TreeRegression(data, **kwargs)
    elif kwargs['model'] == 'KNearestNeighbors': model = KNearestNeighbors(data, **kwargs)

    # if there was an invalid model input, throw an error.
    else: raise Exception('invalid model keyword argument...')

    #model.hyperparameter_tuning()
    model.train()
    model.describe()
    model.test()

"""
Add command line capabilities.
"""
if __name__ == '__main__':
    # get command line arguments.
    command_strings = sys.argv[1:]

    # placeholder object of kwargs.
    kwargs = {}

    # iterate over the command strings and switch based on the inputs.
    i = 0
    while i < len(command_strings):
        # if --model, insert model type and incrememnt by two.
        # Input is the name of a model.
        if command_strings[i] == '--model': kwargs['model'] = command_strings[i+1]

        # if --constant, constant is True if input is 1. This is used for data_loader.
        # input is 0 or 1.
        elif command_strings[i] == '--constant': kwargs['constant'] = command_strings[i+1] == '1'

        # if --normalize, normalize is True if input is 1.
        # input is 0 or 1.
        elif command_strings[i] == '--normalize': kwargs['normalize'] = command_strings[i+1] == '1'

        # if --squared, squared is True if input is 1. This is used for data_loader.
        # input is 0 or 1.
        elif command_strings[i] == '--squared': kwargs['squared'] = command_strings[i+1] == '1'

        # if --only-proposed, only_proposed is True if input is 1. This is used for data_loader.
        # input is 0 or 1.
        elif command_strings[i] == '--only-proposed': kwargs['only_proposed'] = command_strings[i+1] == '1'

        # if --epochs, read epoch total.
        # input is int.
        elif command_strings[i] == '--epochs': kwargs['epochs'] = int(command_strings[i+1])

        # if --validation-split, read validation_split value for NN.
        # Input is a float.
        elif command_strings[i] == '--validation-split': kwargs['validation_split'] = float(command_strings[i+1])

        # if --hidden-layers, read the structure of hidden_layers for NN.
        # Input is comma separated integers.
        elif command_strings[i] == '--hidden-layers': kwargs['hidden_layers'] = [int(x) for x in command_strings[i+1].split(',')]

        # if --t-value-threshold, read the minimum t_value_threshold for OLS.
        # Input is float.
        elif command_strings[i] == '--t-value-threshold': kwargs['t_value_threshold'] = float(command_strings[i+1])

        # if --n-estimators, read the n_estimators for TreeRegression.
        # Input is integer.
        elif command_strings[i] == '--n-estimators': kwargs['n_estimators'] = float(command_strings[i+1])

        # if --max-dpeth, read the max_depth for TreeRegression.
        # Input is integer.
        elif command_strings[i] == '--max-depth': kwargs['max_depth'] = float(command_strings[i+1])

        # if --min-samples-leaf, read the min_samples_leaf for TreeRegression.
        # Input is integer.
        elif command_strings[i] == '--min-samples-leaf': kwargs['min_samples_leaf'] = float(command_strings[i+1])

        i += 2

    # pass keywords into main function.
    main(**kwargs)
