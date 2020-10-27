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
# df_model = needs to be pandas df with both x and y 
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
    print(featureScores.nlargest(50,'Score'))  #print best features

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
def recursive_feature_elimination(data_x, data_y):
    import pandas as pd 
    from sklearn.feature_selection import RFECV
    from sklearn.svm import SVR
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    #estimator = SVR(kernel="linear")

    #estimator = AdaBoostClassifier(extra_tree, random_state=0)
    #estimator = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
     #         solver='lbfgs')
    estimator = DecisionTreeClassifier()

    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=7, random_state=1)
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
def variance_threshold(data_x, data_y):
    import pandas as pd
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=0.0)
    print("Original feature shape:", data_x.shape)
    new_X = selector.fit_transform(data_x)
    print("Transformed feature shape:", new_X.shape)
    data_x = data_x.loc[:, selector.get_support()]

    # since the function below requires both the x and y in a dataframe, this goes through some code to achieve that 
    x = data_x
    y = data_y
    #concat two dataframes for better visualization 
    y.reset_index(drop=True)
    pd.set_option('display.max_rows', None)
    matrix = pd.concat([x,y],axis=1)
    matrix = matrix.reset_index()

    return matrix

#from here: https://github.com/erdogant/pca
def pca(data_x): 
    from pca import pca
    # Initialize
    model = pca()
    # Fit transform
    out = model.fit_transform(data_x)

    # Print the top features. The results show that f1 is best, followed by f2 etc
    print(out['topfeat'])

    #plotting a model to show the percentage of components that represent 95% variance of data
    model.plot()

    # need to fix it - since to many plots to see the name of the metrics
    # Create 3D scatter plots
    #model.biplot(legend=False, SPE=True, hotellingt2=True)
    #model.biplot3d(legend=False, SPE=True, hotellingt2=True)

    # Create only the scatter plots
    model.scatter(legend=False, SPE=True, hotellingt2=True)
    model.scatter3d(legend=False, SPE=True, hotellingt2=True)


# joins the x and y data into one pandas dataframe - called matrix
def join_dataxy(data_x,data_y):
    import pandas as pd
    # since the function below requires both the x and y in a dataframe, this goes through some code to achieve that 
    x = data_x
    y = data_y

    #concat two dataframes for better visualization 
    y.reset_index(drop=True)
    pd.set_option('display.max_rows', None)
    matrix = pd.concat([x,y],axis=1)
    return matrix 

#take in matrix xy and split it
def split_dataxy(data_xy):
    import pandas as pd
    x = data_xy
    y = data_xy.pop('maven_reuse')

    return {
        'train_x': x,
        'train_y': y
    }  

# take in matrix xy and split it into train and test 
def generate_train_test_xy(dataxy):
    # separate into train and test datasets.
    train_x = dataxy.sample(frac=0.8,random_state=0)
    test_x = dataxy.drop(train_x.index)   # remove all training observations.

    # split x and y values.
    train_y = train_x.pop('maven_reuse')
    test_y = test_x.pop('maven_reuse')
    return {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y,
    } 

# just to get an output score for all models with no parameter tuning
def all_model_score(data_x,data_y,test_x,test_y):
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from pylmnn import LargeMarginNearestNeighbor as LMNN
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    import numpy as np
    from sklearn.model_selection import cross_val_score

    #knn = KNeighborsClassifier(leaf_size=1, metric='minkowski', n_neighbors=7, p=4, weights='distance')
    #clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_features=8, min_impurity_decrease=0)
    #svc = SVC(C=50, decision_function_shape='ovo', gamma=1, kernel='rbf')
    #knn = KNeighborsClassifier(leaf_size=1, metric='manhattan', n_neighbors=6, p=1, weights='distance')
    #clf = DecisionTreeClassifier(criterion='gini', max_depth=10, max_features=12, min_samples_leaf=2, min_samples_split=2)
    
    knn = KNeighborsClassifier()

    def w_dist(x, y, **kwargs):
        return sum(kwargs["weights"]*((x-y)*(x-y)))

    wkn = knn = KNeighborsClassifier(metric=w_dist, p=2, 
                           metric_params={'weights': np.random.random(data_x.shape[1])})
    clf = DecisionTreeClassifier()
    svc = SVC()
    lmnn = LMNN()

    knn.fit(data_x, data_y)
    wkn.fit(data_x, data_y)
    clf.fit(data_x, data_y)
    svc.fit(data_x, data_y)

    knn_y_pred = knn.predict(test_x)
    wkn_y_pred = knn.predict(test_x)
    clf_y_pred = clf.predict(test_x)
    svc_y_pred = svc.predict(test_x)

    print('knn accuracy:')
    scores = cross_val_score(knn, data_x, data_y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(knn, test_x, test_y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(accuracy_score(data_y, knn.predict(data_x)))
    print(accuracy_score(test_y, knn_y_pred))
    ## Evaluating the confusion matrix results - includes precission, recall, f1-score, support
    print(classification_report(test_y, knn_y_pred))

    #print('wkn accuracy:')
    #print(accuracy_score(data_y, wkn.predict(data_x)))
    #print(accuracy_score(test_y, wkn_y_pred))
    ## Evaluating the confusion matrix results - includes precission, recall, f1-score, support
    #print(classification_report(test_y, knn_y_pred))

    print('decision tree accuracy:')
    scores = cross_val_score(clf, data_x, data_y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clf, test_x, test_y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(accuracy_score(data_y, clf.predict(data_x)))
    print(accuracy_score(test_y, clf_y_pred))
    ## Evaluating the confusion matrix results - includes precission, recall, f1-score, support
    print(classification_report(test_y, clf_y_pred))

    print('svm accuracy:')
    scores = cross_val_score(svc, data_x, data_y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(svc, test_x, test_y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(accuracy_score(data_y, svc.predict(data_x)))
    print(accuracy_score(test_y, svc_y_pred))
    ## Evaluating the confusion matrix results - includes precission, recall, f1-score, support
    print(classification_report(test_y, svc_y_pred))

    # Train the metric learner
    lmnn.fit(data_x, data_y)
    # Fit the nearest neighbors classifier
    knn = KNeighborsClassifier()
    knn.fit(lmnn.transform(data_x), data_y)

    lmnn_acc_train = knn.score(lmnn.transform(data_x), data_y)
    print('LMNN accuracy on train set of {} points: {:.4f}'.format(data_x.shape[0], lmnn_acc_train))

    # Compute the k-nearest neighbor test accuracy after applying the learned transformation
    lmnn_acc = knn.score(lmnn.transform(test_x), test_y)
    print('LMNN accuracy on test set of {} points: {:.4f}'.format(test_x.shape[0], lmnn_acc))

    ## Evaluating the confusion matrix results - includes precission, recall, f1-score, support
    #print(classification_report(test_y, lmnn_acc))

def model_tuning(data_x,data_y,test_x,test_y,model,param_space):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import GridSearchCV
    from skopt import BayesSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

    # defines the bayes search cv with parameters - refer to: https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
    # Bayesian optimisation is a type of sequential method in which it learns from each step what the optimal hyper-parameters are
    # (in contrast to grid or random search) - using some complicated maths model (im not sure about)       
    search = BayesSearchCV(estimator=model, search_spaces=param_space, n_jobs=-1, cv=cv, refit=True)

    # perform the search - i.e. it fits the model on the training data set for the different hyper-parameter settings
    search_result = search.fit(data_x, data_y)

    # Prints the results - optimal hyper-parameters and the accuracy score
    print("The best parameters are %s with a score of %0.2f"
        % (search_result.best_params_, search_result.best_score_))

    print("scores")
    search.score(data_x, data_y)
    search.score(test_x, test_y)

    # Displays all of the hyper-parameters combination in descending order of accuracy score
    #grid_results = pd.concat([pd.DataFrame(search_result.cv_results_["params"]),pd.DataFrame(search_result.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
    #grid_results.sort_values(by=['Accuracy'], inplace=True, ascending=False)
    #print(grid_results.head)


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

