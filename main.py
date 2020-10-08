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

from utils import data_loader, pre_training_analysis_tools

"""
This function will display a correlation heatmap and scatterplots to assess
multicollinearity and performance of metrics against the reuse rate.
"""
def describe_data():
    data = data_loader.load_real_dataset(sqaured=False, remove_multicollinearity=False)
    #data = data_loader.load_real_dataset(sqaured=False, remove_multicollinearity=False)
    #pre_training_analysis_tools.display_correlation_heatmap(data['test_x'])
    #pre_training_analysis_tools.univariate_selection(data['test_x'], data['test_y'])
    #pre_training_analysis_tools.univariate_selection_chi(data['test_x'], data['test_y'])
    #re_training_analysis_tools.variance_threshold(data['test_x'])
    #pre_training_analysis_tools.feature_importance_ExtraTreesClassifier(data['test_x'], data['test_y'])
    #pre_training_analysis_tools.feature_importance_RandomForest(data['test_x'], data['test_y'])
    #pre_training_analysis_tools.recursive_feature_elimination(data['test_x'], data['test_y'])
    pre_training_analysis_tools.heatmap_with_dropped_highlycorrelated(data['test_x'], data['test_y'])
    #pre_training_analysis_tools.display_covariance_heatmap(data['test_x'])
    #pre_training_analysis_tools.display_correlation_scatterplots_xy(data['test_x'], data['test_y'])
    #pre_training_analysis_tools.display_correlation_scatterplots_x(data['test_x'])
    pre_training_analysis_tools.remove_collinear_features(data['test_x'], 'maven_reuse', 0.95, 'True')

"""
This is the main function that is run.
"""
def main():
    # get data to use in this analysis.
    data = data_loader.load_real_dataset(remove_multicollinearity=False, only_proposed=False)

    # determine which model we are using.
    model = LeastSquares(data, normalize=True)
    #model.hyperparameter_tuning()
    model.train()
    model.describe()
    model.test()

if __name__ == '__main__':
    main()
