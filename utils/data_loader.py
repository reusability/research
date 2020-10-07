"""
This module handles the loading of data from csv files into the tensorflow
ts.data.Dataset format.

Last update: MB 29/08/2020 - added the ability to square each datapoint.
"""
# import external libraries.
import pandas as pd
import numpy as np
import tensorflow as tf

# define constants.
DATASET_2019_FILEPATH = r'./data/dataset_2019.csv'  # dataset from https://www.sciencedirect.com/science/article/pii/S235234091931042X
DATASET_2020_FILEPATH = r'./data/aggregate_201007.csv'
DEFAULT_BATCH_SIZE = 32

"""
Return the 2019 dataset in the format of tf.data.Dataset.
"""
def load_2019_dataset(constant=True, sqaured=False, remove_multicollinearity=False, only_proposed=False):
    # define dataset information.
    filepath = DATASET_2019_FILEPATH
    y_column = 'ReuseRate'
    train_fraction = 0.8
    unnecessary_columns = ['id', 'Project', 'Name', 'LongName']
    only_proposed_columns = ['LCOM5', 'NII', 'TCD', 'PDA', 'DIT']
    multicollinear_columns = ['NL', 'WMC', 'CBO', 'CBOI', 'NOI', 'RFC', 'AD', 'CD', 'TNOS', \
        'CLOC', 'TCLOC', 'DLOC', 'LLOC', 'TLOC', 'LOC', 'TNG', 'TNPM', 'TNM', 'TLLOC']

    # call data loader function with the configuration specified above.
    return _load_dataset(filepath=filepath, unnecessary_columns=unnecessary_columns,
        y_column=y_column, multicollinear_columns=multicollinear_columns,
        only_proposed_columns=only_proposed_columns, train_fraction=train_fraction,
        constant=constant, sqaured=sqaured, remove_multicollinearity=remove_multicollinearity,
        only_proposed=only_proposed)

"""
Return the 2020 dataset in the format of tf.data.Dataset.
"""
def load_2020_dataset(constant=True, sqaured=False, remove_multicollinearity=False, only_proposed=False):
    # define dataset information.
    filepath = DATASET_2020_FILEPATH
    y_column = 'maven_reuse'
    train_fraction = 0.8
    unnecessary_columns = ['project', 'release', 'maven_release']
    only_proposed_columns = [
        'returnQty_median', \
        'maxNestedBlocksQty_average', \
        'numbersQty_stdev', \
        'tcc_stdev', \
        'lcc_stdev', \
        'numbersQty_max', \
        'nosi_max', \
        'visibleFieldsQty_stdev', \
        'visibleFieldsQty_max'
    ]
    multicollinear_columns = []

    # call data loader function with the configuration specified above.
    return _load_dataset(filepath=filepath, unnecessary_columns=unnecessary_columns,
        y_column=y_column, multicollinear_columns=multicollinear_columns,
        only_proposed_columns=only_proposed_columns, train_fraction=train_fraction,
        constant=constant, sqaured=sqaured, remove_multicollinearity=remove_multicollinearity,
        only_proposed=only_proposed)

"""
This module will add a new column for each existing column. Each column will
contain squared values of an existing column.
"""
def generate_squared_values(dataset):
    # names of each column we will sqaure. remove ReuseRate.
    headers = dataset.keys()

    # for each column name in headers, create a new column with squared values.
    # do not create new columns for constant or reuse rate.
    for column_name in [h for h in headers if h != 'ReuseRate' and h != 'constant']:
        dataset[column_name+'_squared'] = dataset[column_name].pow(2)

    # return the updated dataset.
    return dataset

"""
This is an internal function that performs common operations when loading
different datasets.
"""
def _load_dataset(filepath=None, unnecessary_columns=[], y_column=None, multicollinear_columns=[],
    only_proposed_columns = [], train_fraction=0.8, constant=True, sqaured=False,
    remove_multicollinearity=False, only_proposed=False):
    # throw an error if filepath or y_column is None.
    if filepath is None:
        raise Exception('filepath cannot be none...')
    if y_column is None:
        raise Exception('y_column cannot be none...')

    # read dataset from csv.
    complete_dataset = pd.read_csv(filepath)

    # strip any observation with incomplete data.
    complete_dataset = complete_dataset.dropna()

    # drop unnecessary columns.
    for column in unnecessary_columns:
        complete_dataset.pop(column)

    # if remove_multicollinearity is true, remove highly correlated x values.
    if remove_multicollinearity is True:
        for column in multicollinear_columns:
            complete_dataset.pop(column)

    # if we only want to use the proposed metrics, only select these columns.
    # ensure we retain the y_column.
    if only_proposed is True:
        complete_dataset = complete_dataset[only_proposed_columns + [y_column]]

    # if squared is true, add sqaured columns.
    if sqaured is True:
        complete_dataset = generate_squared_values(complete_dataset)

    # if we are adding a constant, add it to the complete dataset.
    if constant is True:
        complete_dataset['constant'] = 1

    # separate into train and test datasets.
    train_x = complete_dataset.sample(frac=train_fraction, random_state=0)
    test_x = complete_dataset.drop(train_x.index)   # remove all training observations.

    # split x and y values.
    train_y = train_x.pop(y_column)
    test_y = test_x.pop(y_column)

    # return the data split into test and training X and Y values.
    return {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y,
    }

"""
Normalize the data. required for neural networks. Dataset should be pandas.
"""
def get_normalization_params(dataset):
    # dictionary of parameters.
    normalized_params = dict()

    # iterate over each column and calculate the mean and standard deviation
    # for that column.
    for column in dataset.columns:
        # calculate mean and standard deviation.
        normalized_params[column] = {
            'mean': dataset[column].mean(),
            'std': dataset[column].std()
        }

    # override the mean and std of the constant. otherwise the constant would
    # not exist.
    normalized_params['constant'] = {
        'mean': 0,
        'std': 1,
    }

    # return dictionary.
    return normalized_params
