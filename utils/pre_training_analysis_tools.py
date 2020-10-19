"""
This module contains tools to analyse the data before we perform any training.

Last updated: MB 29/08/2020 - moved functions from Ibrahim to this module.
"""
# import external libraries.
import seaborn as sns
import matplotlib.pyplot as plt

"""
This function will display a heatmap of correlations between each metrics. input
should be a pandas dataset.
"""
def display_correlation_heatmap(data):
    # Creating a heatmap - to find the correlation between each variable
    #get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    plt.title('Correlation')

    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

"""
This function will display a heatmap of correlations between each metrics. input
should be a pandas dataset.
"""
def display_covariance_heatmap(data):
    # Creating a heatmap - to find the correlation between each variable
    #get correlations of each features in dataset
    corrmat = data.cov()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    plt.title('Covariance')

    #plot heat map
    g=sns.heatmap(data[top_corr_features].cov(),annot=True,cmap="RdYlGn")

"""
This function will display a unique scatter plot for each pair of metrics to
visualise correlation between all metrics. input must be a pandas dataframe.
"""
def display_correlation_scatterplots_x(data):
    figure_count = 0

    # iterate over each pair of metrics.
    for i in range(0, len(data.keys())):
        for j in range(i+1, len(data.keys())):
            # setup plot.
            plt.figure(figure_count)
            plt.axes()
            plt.xlabel(data.columns[i])
            plt.ylabel(data.columns[j])

            # populate the plot.
            plt.scatter(data[data.columns[i]], data[data.columns[j]])

            # increment figure count.
            figure_count += 1

"""
This function will display a scatter plot of data_x and data_y for each x.
data_x should be a pandas dataframe and data_y should be a pandas series.
"""
def display_correlation_scatterplots_xy(data_x, data_y):
    figure_count = 0

    # dictionary with correlation values.
    correlations = dict()

    # iterate over each pair of metrics.
    for i in range(0, len(data_x.keys())):
        # setup plot.
        plt.figure(figure_count)
        plt.axes()
        plt.xlabel(data_x.columns[i])
        plt.ylabel(data_y.name)

        # populate the plot.
        plt.scatter(data_x[data_x.columns[i]], data_y)

        # increment figure count.
        figure_count += 1

        # update the correlations dict.
        correlations[data_x.columns[i]] = data_x[data_x.columns[i]].corr(data_y)

    # print the correlations dict.
    for metric in correlations:
        print(metric+': %.2f' % correlations[metric])
