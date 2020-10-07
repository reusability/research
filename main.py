"""
This module is the entrypoint for the project. It will handle the order of
task execution of training and analysing models.

Last updated: MB 12/08/2020 - created module
"""
# import local modules.
from model.least_squares import LeastSquares
from model.tree_classifier import TreeClassifier
from model.nerual_net import NN
from model.k_nearest_neigbors import KNearestNeighbors
from model.least_squares import LeastSquares

from utils import data_loader, pre_training_analysis_tools

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
This is the main function that is run.
"""
def main():
    # get data to use in this analysis.
    data = data_loader.load_2020_dataset(only_proposed=True)

    # determine which model we are using.
    model = TreeClassifier(data, normalize=True)
    #model.hyperparameter_tuning()
    model.train()
    model.describe()
    model.test()

if __name__ == '__main__':
    main()
