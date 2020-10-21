"""
This module handles the loading of data from csv files into the tensorflow
ts.data.Dataset format.

Last update: MB 29/08/2020 - added the ability to square each datapoint.
"""
# import external libraries.
import pandas as pd
import tensorflow as tf
import numpy as np

# define constants.
DATASET_2019_FILEPATH = r'./data/dataset_2019.csv'  # dataset from https://www.sciencedirect.com/science/article/pii/S235234091931042X
DATASET_REAL_FILEPATH = r'./data/aggregate_201019.csv'  # actual dataset 
#DATASET_REAL_FILEPATH = r'./data/dataset_reduced.csv'  # actual dataset 

DEFAULT_BATCH_SIZE = 32

"""
Return the 2019 dataset in the format of tf.data.Dataset.
"""
def load_2019_dataset(constant=True, sqaured=False, remove_multicollinearity=False, only_proposed=False):
    # read dataset from csv.
    complete_dataset = pd.read_csv(DATASET_2019_FILEPATH)

    # strip any observation with incomplete data.
    complete_dataset.dropna()

    # drop unnecessary values.
    complete_dataset.pop('Project')
    complete_dataset.pop('id')
    complete_dataset.pop('Name')
    complete_dataset.pop('LongName')

    # if remove_multicollinearity is true, remove highly correlated x values.
    if remove_multicollinearity is True:
        complete_dataset.pop('NL')
        complete_dataset.pop('WMC')
        complete_dataset.pop('CBO')
        complete_dataset.pop('CBOI')
        complete_dataset.pop('NOI')
        complete_dataset.pop('RFC')
        complete_dataset.pop('AD')
        complete_dataset.pop('CD')
        complete_dataset.pop('TNOS')
        complete_dataset.pop('CLOC')
        complete_dataset.pop('TCLOC')
        complete_dataset.pop('DLOC')
        complete_dataset.pop('LLOC')
        complete_dataset.pop('TLOC')
        complete_dataset.pop('LOC')
        complete_dataset.pop('TNG')
        complete_dataset.pop('TNPM')
        complete_dataset.pop('TNM')
        complete_dataset.pop('TLLOC')

    # if we are adding a constant, add it to the complete dataset.
    if constant is True:
        complete_dataset['constant'] = 1

    # if squared is true, add swaured columns.
    if sqaured is True:
        complete_dataset = generate_squared_values(complete_dataset)

    # if we only want to use the proposed metrics, throw out all other columns.
    if only_proposed is True:
        complete_dataset = complete_dataset[['LCOM5', 'NII', 'TCD', 'PDA', 'DIT', 'constant', 'ReuseRate']]

    # separate into train and test datasets.
    train_x = complete_dataset.sample(frac=0.8,random_state=0)
    test_x = complete_dataset.drop(train_x.index)   # remove all training observations.

    # split x and y values.
    train_y = train_x.pop('ReuseRate')
    test_y = test_x.pop('ReuseRate')

    # return the data split into test and training X and Y values.
    return {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y,
    }

def load_real_dataset(constant=True, sqaured=False, remove_multicollinearity=False, only_proposed=False):
    # read dataset from csv.
    complete_dataset = pd.read_csv(DATASET_REAL_FILEPATH)

    # strip any observation with incomplete data.
    complete_dataset.dropna(how='any', inplace=True)

    # drop unnecessary values.
    complete_dataset.pop('project')
    complete_dataset.pop('release')
    complete_dataset.pop('maven_release')
    complete_dataset.pop('class_count')

    # turn maven reuse into classification
    complete_dataset['maven_reuse'] = np.where(complete_dataset['maven_reuse'].between(0,44), 1, complete_dataset['maven_reuse'])
    complete_dataset['maven_reuse'] = np.where(complete_dataset['maven_reuse'].between(45,445), 2, complete_dataset['maven_reuse'])
    complete_dataset['maven_reuse'] = np.where(complete_dataset['maven_reuse'].between(446,100000), 3, complete_dataset['maven_reuse'])

    # if remove_multicollinearity is true, remove highly correlated x values.
    if remove_multicollinearity is True:
        complete_dataset.pop('wmc')
        complete_dataset.pop('rfc')
        complete_dataset.pop('cbo')
        complete_dataset.pop('lcc')

    # if we are adding a constant, add it to the complete dataset.
    if constant is True:
        complete_dataset['constant'] = 1

    # if squared is true, add swaured columns.
    if sqaured is True:
        complete_dataset = generate_squared_values(complete_dataset)

    # if we only want to use the proposed metrics, throw out all other columns.
    if only_proposed is True:
        #complete_dataset = complete_dataset[['cbo_max', 'staticMethodsQty_average', 'variablesQty_stdev', 'finalMethodsQty_stdev', 'stringLiteralsQty_stdev', 'visibleFieldsQty_max', 'maven_reuse']]
        complete_dataset = complete_dataset[[
            'privateMethodsQty_sum',
            'wmc_stdev',
            'lcc_stdev',
            'abstractMethodsQty_stdev',
            'modifiers_stdev',
            'privateMethodsQty_max',
            'innerClassesQty_average',
            'finalFieldsQty_average',
            'staticMethodsQty_max',
            'cbo_average',
            'parenthesizedExpsQty_max',
            'returnQty_stdev',
            'anonymousClassesQty_average',
            'dit_average',
            'protectedFieldsQty_stdev',
            'lambdasQty_max',
            'stringLiteralsQty_max',
            'returnQty_max',
            'maxNestedBlocksQty_average',
            'returnQty_median',
            'maven_reuse'
        ]]

        #

    # separate into train and test datasets.
    train_x = complete_dataset.sample(frac=0.8,random_state=0)
    test_x = complete_dataset.drop(train_x.index)   # remove all training observations.

    # split x and y values.
    train_y = train_x.pop('maven_reuse')
    test_y = test_x.pop('maven_reuse')

    # normalising the data - i assume we are meant to when feature analysing or not, either way i did this cause idk how else to
    # code sourced from here: https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
    from sklearn import preprocessing

    # for train x
    x = train_x.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    train_x = pd.DataFrame(x_scaled, columns=train_x.columns)

    # for test x
    x = test_x.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    test_x = pd.DataFrame(x_scaled, columns=test_x.columns)
    

    # return the data split into test and training X and Y values.
    return {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y
    }

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

    # override the mean and std of the constant. otherwise we would be diviving
    # by 0...
    normalized_params['constant'] = {
        'mean': 0,
        'std': 1,
    }

    # return dictionary.
    return normalized_params
