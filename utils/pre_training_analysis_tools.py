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



# code sourced from link: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on#44674459
def remove_collinear_features(df_model, target_var, threshold, verbose):
    import numpy as np
    import pandas as pd

    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold and which have the least correlation with the target (dependent) variable. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        df_model: features dataframe
        target_var: target (dependent) variable
        threshold: features with correlations greater than this value are removed
        verbose: set to "True" for the log printing

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = df_model.drop(target_var, 1).corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    dropped_feature = ""
    drop_feature_name = ""

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1): 
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            #print("item:")
            #print(item)
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if (val >= threshold) and (col.values[0] in df_model.columns) and (row.values[0] in df_model.columns):
                # Print the correlated features and the correlation value
                if verbose:
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                col_value_corr = df_model[col.values[0]].corr(df_model[target_var])
                row_value_corr = df_model[row.values[0]].corr(df_model[target_var])
                if verbose:
                    print("{}: {}".format(col.values[0], np.round(col_value_corr, 3)))
                    print("{}: {}".format(row.values[0], np.round(row_value_corr, 3)))
                if abs(col_value_corr) < abs(row_value_corr):
                    drop_cols.append(col.values[0])
                    dropped_feature = "dropped: " + col.values[0]
                    drop_feature_name = col.values[0]
                else:
                    drop_cols.append(row.values[0])
                    dropped_feature = "dropped: " + row.values[0]
                    drop_feature_name = row.values[0]
                if verbose:
                    print(dropped_feature)
                    print("-----------------------------------------------------------------------------")
            
            if drop_feature_name in df_model.columns:
                # Drop one of each pair of correlated columns
                del df_model[drop_feature_name] # deleting the column from the dataset
                drop_feature_name = ""


    drops = set(drop_cols)
    print("dropped columns: ")
    print(list(drops))
    print("-----------------------------------------------------------------------------")
    print("used columns: ")
    print(df_model.columns.tolist())

    return df_model

# using scikit learn - sourced from article too
# input different statistical measures to rank data based on, below:
# Pearson’s Correlation Coefficient: f_regression()
# ANOVA: f_classif()
# Chi-Squared: chi2()
# Mutual Information: mutual_info_classif() and mutual_info_regression()

# Note: ANOVA should be the best metric to use although mutual info is apparently also powerful, if the independant data is numerical, and the 
# dependent data is categorical 
def univariate_selection(data_x,data_y):
    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import  mutual_info_classif

    #apply SelectKBest class to extract best features
    bestfeatures = SelectKBest(score_func= mutual_info_classif, k='all')
    fit = bestfeatures.fit(data_x,data_y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data_x.columns)
    #concat two dataframes for better visualization 
    pd.set_option('display.max_rows', None)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    print(featureScores.nlargest(20,'Score'))  #print best features

# using scikit learn , sourced from article (forget where)
# Note: not viable yet -> need to select a machine learning algo and probably tune it before this
def feature_importance_ExtraTreesClassifier(data_x,data_y,model):
    import pandas as pd
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model.fit(data_x,data_y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=data_x.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(feat_importances.nlargest(10))
    print("Feature Importance Ranking")
    print(df)

#using scikit learn - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
#After each iteration, it will remove step (set to 1) number of features that have the worst score based on the 
#the estimator used. It will then compare the mean score from each cross fold, and choose the number of features that
# had the best score. 
# Note: not viable yet -> need to select a machine learning algo and probably tune it before this
def recursive_feature_elimination(data_x, data_y, estimator):
    import pandas as pd 
    from sklearn.feature_selection import RFECV
    from sklearn.svm import SVR
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.ensemble import AdaBoostClassifier

    #estimator = SVR(kernel="linear")

    #estimator = AdaBoostClassifier(extra_tree, random_state=0)
    #estimator = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
     #         solver='lbfgs')
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    visualizer = RFECV(estimator, step=1, cv=cv, scoring='accuracy')
    visualizer = visualizer.fit(data_x, data_y)

    dfscores = pd.DataFrame(visualizer.ranking_)
    dfselected = pd.DataFrame(visualizer.support_)
    dfcolumns = pd.DataFrame(data_x.columns)
    dfscore = pd.DataFrame(visualizer.grid_scores_)
    pd.set_option('display.max_rows', None)
    featureScores = pd.concat([dfcolumns,dfscores,dfselected, dfscore],axis=1)
    featureScores.columns = ['Feature','Ranking', 'Selected', 'Score']  #naming the dataframe columns

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(visualizer.grid_scores_) + 1), visualizer.grid_scores_)
    plt.show()

    print('Optimal number of features :', visualizer.n_features_)
    print(featureScores.nsmallest(20,'Ranking'))  #print 10 best features

# code is sourced from here: https://stackoverflow.com/questions/29298973/removing-features-with-low-variance-using-scikit-learn
def variance_threshold(data_x):
    import pandas as pd
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold()
    print("Original feature shape:", data_x.shape)
    new_X = selector.fit_transform(data_x)
    print("Transformed feature shape:", new_X.shape)
    data_x = data_x.loc[:, selector.get_support()]
    print(data_x.columns)

# joins the x and y data into one pandas dataframe - called matrix
def join_dataxy(data_x,data_y):
    import pandas as pd
    # since the function below requires both the x and y in a dataframe, this goes through some code to achieve that 
    x = data_x
    y = data_y
    #concat two dataframes for better visualization 
    pd.set_option('display.max_rows', None)
    matrix = pd.concat([x,y],axis=1)
    return matrix 

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
