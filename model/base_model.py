"""
This modules defines the base class for all machine learning models to analyse
reusability rate.

Last updated: MB 29/08/2020 - created module.
"""
# import external libraries.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns


# import local modules.
from utils import data_loader

"""
Base class that all models inherit from.
"""
class BaseModel:
    """
    store dataset. data is a dictionary.
    """
    def __init__(self, data, normalize=False):
        print(">> initialising model...")

        # if we are normalizing data, save the normalized x value.
        if normalize is True:
            self.normalization_params = data_loader.get_normalization_params(data['train_x'])
            self.train_x = self.normalize_x(data['train_x'])
            self.test_x = self.normalize_x(data['test_x'])

        # if we are not normalizing data, use regular x values.
        else:
            self.train_x = data['train_x']
            self.test_x = data['test_x']

        # save the y values and other attributes.
        self.train_y = data['train_y']
        self.test_y = data['test_y']
        self.test_predictions = pd.Series()    # placeholder for 'test' function.
        self.is_trained = False

    def hyperparameter_tuning(self, type, param_space):

        # definees the type and number of cross validation splits - refer to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html
        # Repeated Stratified K Fold -> This repeats a stratified k fold n number of times
        # Stratified k fold -> Shuffles the data once before splitting into n different parts,
        # where each part is used as a test set 

        # use this for regression
        cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=1)
        
        # use this one for classifiation - esp, when there is an inbalance in the classes (i.e. a lot are high reuse rate, 
        # but barely any are low)
        # requirement - that the number of different samples in each class (i.e. high reusability) is greater then n_splits
        # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

        if type == 'Grid':
            # Set all the variables for the grid search cross validation 
            search = GridSearchCV(estimator=self.model, param_grid=param_space, cv=cv, n_jobs=-1, scoring='accuracy')

        elif type == 'Bayesian':
            # defines the bayes search cv with parameters - refer to: https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
            # Bayesian optimisation is a type of sequential method in which it learns from each step what the optimal hyper-parameters are
            # (in contrast to grid or random search) - using some complicated maths model (im not sure about)       
            search = BayesSearchCV(estimator=self.model, search_spaces=param_space, n_jobs=-1, cv=cv)

        # perform the search - i.e. it fits the model on the training data set for the different hyper-parameter settings
        search_result = search.fit(self.train_x, self.train_y)

        # Prints the results - optimal hyper-parameters and the accuracy score
        print("The best parameters are %s with a score of %0.2f"
            % (search_result.best_params_, search_result.best_score_))

        # Displays all of the hyper-parameters combination in descending order of accuracy score
        grid_results = pd.concat([pd.DataFrame(search_result.cv_results_["params"]),pd.DataFrame(search_result.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
        grid_results.sort_values(by=['Accuracy'], inplace=True, ascending=False)
        print(grid_results.head)

    def find_hyperparams(self):
        print("tunning parameter.")

    """
    train the model with current train and test XY values saved as attributes.
    """
    def train(self):
        print(">> training model...")

    """
    output a description of the model.
    """
    def describe(self):
        print(">> describing model...")

        # throw an error if model is not trained yet.
        if self.is_trained is False:
            raise Exception('Train model before describing coefficients.')
            return

    """
    generate prdictions for the test_x data.
    """
    def test(self):
        print(">> predicting test data...")

        # throw an error if model is not trained yet.
        if self.is_trained is False:
            raise Exception('Train model before describing coefficients.')
            return

    """
    analyse the performance of the predictions.
    """
    def assess_performance(self):
        # if there is no 'test_predictions' data generated, throw error.
        if self.test_predictions is None:
            raise Exception('Run the `test` function to predict test data.')

        print(">> assessing prediction performance...")

        # configure plot.
        plt.figure(2)
        plt.axes(aspect='equal')
        plt.xlabel('True Value [ReuseRate]')
        plt.ylabel('Predicted Value [ReuseRate]')
        lims = [0, 600] # axis limits.
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)

        # Scatter plot of the predicted ReuseRate data set against the actual ReuseRate data set (as per mavern)
        plt.scatter(self.test_y, self.test_predictions)

        # calculate the R-squared value between the prediction and actuals.
        ESS = int(((self.test_predictions - int(self.test_y.mean())).pow(2)).sum())
        RSS = int(((self.test_y - self.test_predictions).pow(2)).sum())
        TSS = int(((self.test_y - int(self.test_y.mean())).pow(2)).sum())

        # the Rsquared value is only valid if TSS = ESS + RSS.
        # (i am using an epsilon of 10000).
        if abs(TSS - ESS - RSS) < 100000:
            print("ESS: "+str(ESS))
            print("RSS: "+str(RSS))
            print("TSS: "+str(TSS))
            r2 = 1 - RSS / TSS
            print("R-squared: %.2f" % r2)

        # calculate the correlation.
        print("correlation: %.2f" % self.test_y.corr(self.test_predictions))

        ## Printing out the confusion matrix as a heatmap - comparing the trained y variable
        ## with the actual y variable 

        conf_matrix = confusion_matrix(self.test_y, self.test_predictions)
        sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('Real Output')
        plt.ylabel('Predicted Output')

        ##Calculate the accuracy of the model 
        print(accuracy_score(self.test_y, self.test_predictions) )

        ## Evaluating the confusion matrix results - includes precission, recall, f1-score, support
        print(classification_report(self.test_y, self.test_predictions))


    """
    Convert a pandas dataframe of values into normalized values based on the
    normalized params attribute. x_values is a pandas dataframe.
    """
    def normalize_x(self, x_values):
        # throw an error if this model was not setup to use normalized values.
        if not self.normalization_params:
            raise Exception("This model was not setup to use normalized values.")
            return

        # copy the dataframe.
        normalized_values = pd.DataFrame()

        # iterate over each column and normalize.
        for column in x_values.columns:
            # retrieve normalization parameters.
            mean = self.normalization_params[column]['mean']
            std = self.normalization_params[column]['std']

            # save the normalized column.
            normalized_values[column] = (x_values[column] - mean) / std

        # return the normalized dataframe.
        return normalized_values
