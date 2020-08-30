"""
This module is the entrypoint for the project. It will handle the order of
task execution of training and analysing models.

Last updated: MB 12/08/2020 - created module
"""
# import local modules.
from model.least_squares import LeastSquares
from model.tree_classifier import TreeClassifier
from model.nerual_net import NN
from utils import data_loader, pre_training_analysis_tools

"""
This function will display a correlation heatmap and scatterplots to assess
multicollinearity and performance of metrics against the reuse rate.
"""
def describe_data():
    data = data_loader.load_2019_dataset(sqaured=False, remove_multicollinearity=False)
    pre_training_analysis_tools.display_correlation_heatmap(data['test_x'])
    pre_training_analysis_tools.display_covariance_heatmap(data['test_x'])
    #pre_training_analysis_tools.display_correlation_scatterplots_xy(data['test_x'], data['test_y'])
    #pre_training_analysis_tools.display_correlation_scatterplots_x(data['test_x'])

"""
This is the main function that is run.
"""
def main():
    # get data to use in this analysis.
    data = data_loader.load_2019_dataset(only_proposed=True)

    # determine which model we are using.
    model = LeastSquares(data, normalize=False)
    model.train()
    model.describe()
    model.test()

if __name__ == '__main__':
    main()
