"""
This module handles the loading of data from csv files into the tensorflow
ts.data.Dataset format.

Last update: MB 29/08/2020 - added the ability to square each datapoint.
"""
# import external libraries.
import pandas as pd
import tensorflow as tf

# define constants.
DATASET_2019_FILEPATH = r'./data/dataset_2019.csv'  # dataset from https://www.sciencedirect.com/science/article/pii/S235234091931042X
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
    complete_dataset.pop('id')
    complete_dataset.pop('Project')
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
