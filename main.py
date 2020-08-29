"""
This module is the entrypoint for the project. It will handle the order of
task execution of training and analysing models.

Last updated: MB 12/08/2020 - created module
"""
# import local modules.
from model.ols import OLS
from model.tree_classifier import TreeClassifier
from model.nerual_net import NN
from utils import data_loader, pre_training_analysis_tools

def main():
    # get data to use in this analysis.
    data = data_loader.load_2019_dataset(sqaured=True)

    #pre_training_analysis_tools.display_correlation_heatmap(data['test_x'])

    # determine which model we are using.
    model = NN(data, [64, 32, 16])
    model.train()
    model.describe()
    model.test()

    #pre_training_analysis_tools.display_correlation_scatterplots_xy(data['test_x'], data['test_y'])
    #pre_training_analysis_tools.display_correlation_scatterplots_x(data['test_x'])

if __name__ == '__main__':
    main()
