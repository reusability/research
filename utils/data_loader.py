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
    train_x = complete_dataset.sample(frac=1.0,random_state=0)
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

def load_real_dataset(constant=False, sqaured=False, normalise=False, remove_multicollinearity=False, only_proposed=False, only_pca=False, only_dt=False, cfs_mi=False, cfs=False, wrapper_knn=False,
                    top20_mi=False, wrapper_svm=False, shap_dt=False):
    # read dataset from csv.
    complete_dataset = pd.read_csv(DATASET_REAL_FILEPATH)

    # strip any observation with incomplete data.
    complete_dataset.dropna(how='any', inplace=True)

    # drop unnecessary values.
    complete_dataset.pop('project')
    complete_dataset.pop('release')
    complete_dataset.pop('maven_release')
    complete_dataset.pop('class_count')

    if normalise is True:
        get_normalization_params(complete_dataset)

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

    if only_pca is True:
        complete_dataset = complete_dataset[[
        'privateMethodsQty_sum',		
        'maxNestedBlocksQty_median',		
        'lcc_min',		
        'abstractMethodsQty_stdev',	
        'parenthesizedExpsQty_max',	
        'modifiers_stdev',	
        'modifiers_min',	
        'dit_median',		
        'loc_stdev',		
        'modifiers_average',		
        'totalFieldsQty_stdev',	
        'defaultMethodsQty_max',		
        'lambdasQty_average',		
        'staticMethodsQty_max',	
        'anonymousClassesQty_average',		
        'loc_max',		
        'anonymousClassesQty_stdev',	
        'lcc_median',	
        'stringLiteralsQty_stdev',		
        'defaultMethodsQty_median',		
        'finalFieldsQty_median',		
        'modifiers_max',		
        'stringLiteralsQty_max',		
        'returnQty_median',		
        'maxNestedBlocksQty_max',
        'maven_reuse'
    ]]	

    if only_dt is True:
        complete_dataset = complete_dataset[[
        'parenthesizedExpsQty_stdev',		
        'uniqueWordsQty_stdev',		
        'returnQty_median',		
        'staticMethodsQty_stdev',	
        'finalFieldsQty_average',	
        'defaultFieldsQty_max',	
        'variablesQty_stdev',	
        'maven_reuse'
    ]]	

    if cfs_mi is True:
        complete_dataset = complete_dataset[[
        'wmc_sum','lcom_average','cbo_sum','tryCatchQty_stdev','loopQty_average','tryCatchQty_average','publicMethodsQty_stdev','comparisonsQty_sum','lcom_stdev',
        'lambdasQty_max','loopQty_stdev','staticFieldsQty_stdev','maven_reuse'
        ]]

    if cfs is True:
        complete_dataset = complete_dataset.iloc[:, [0, 114,  31, 178,  11, 156, 202,  80,  33,  12,  17,  42,  50,
       160,  51,  32,  54,  56,  78,  79, 115,263]]

       #'stringLiteralsQty_max','publicFieldsQty_sum','finalFieldsQty_median','parenthesizedExpsQty_min','protectedFieldsQty_max',
       #'defaultFieldsQty_median','defaultMethodsQty_average',''

    #if wrapper_knn is True:
    #    complete_dataset = complete_dataset[[
    #    'synchronizedMethodsQty_average', 'modifiers_median', 'lcc_sum', 'protectedFieldsQty_max', 'assignmentsQty_sum', 'totalFieldsQty_sum',
    #    'maven_reuse'
    #]]	

    if wrapper_knn is True:
        complete_dataset = complete_dataset[[
        'synchronizedMethodsQty_average', 'modifiers_median', 'lcc_sum', 'protectedFieldsQty_max', 'assignmentsQty_sum', 'totalFieldsQty_sum',
        'maven_reuse'
    ]]	

    if wrapper_svm is True:
        complete_dataset = complete_dataset[[
            'parenthesizedExpsQty_sum', 'publicFieldsQty_average', 'modifiers_median', 'maven_reuse'
        ]]

    if top20_mi is True:
        complete_dataset = complete_dataset[[
            'lcom_sum',  'numbersQty_sum', 'uniqueWordsQty_sum', 'publicFieldsQty_sum','staticFieldsQty_sum',
            'protectedFieldsQty_sum', 'maxNestedBlocksQty_average' ,'modifiers_sum' ,'lcc_sum','uniqueWordsQty_stdev',
            'uniqueWordsQty_median','maxNestedBlocksQty_sum','staticMethodsQty_sum','loc_sum','protectedFieldsQty_average',
            'finalMethodsQty_max','cbo_max','defaultMethodsQty_sum','staticFieldsQty_max' ,'parenthesizedExpsQty_average',
            'maven_reuse'
    ]]	

    if shap_dt is True:
        complete_dataset = complete_dataset [[
            'innerClassesQty_stdev','finalMethodsQty_stdev','nosi_stdev','assignmentsQty_average',
            'defaultFieldsQty_average','staticFieldsQty_max','variablesQty_average','stringLiteralsQty_average',
            'modifiers_max','comparisonsQty_max','tryCatchQty_sum','maxNestedBlocksQty_average',
            'defaultMethodsQty_max', 'maven_reuse'
        ]]


    # separate into train and test datasets.
    train_x = complete_dataset.sample(frac=0.8,random_state=0)
    test_x = complete_dataset.drop(train_x.index)   # remove all training observations.

    # split x and y values.
    train_y = train_x.pop('maven_reuse')
    test_y = test_x.pop('maven_reuse')    

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
