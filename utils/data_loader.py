"""
This module handles the loading of data from csv files into the tensorflow
ts.data.Dataset format.

Last update: MB 12/08/2020 - created module.
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
def load_2019_dataset():
    # read dataset from csv.
    complete_dataset = pd.read_csv(DATASET_2019_FILEPATH)

    # strip any observation with incomplete data.
    complete_dataset.dropna()

    # drop unnecessary values.
    complete_dataset.pop('id')
    complete_dataset.pop('Project')
    complete_dataset.pop('Name')
    complete_dataset.pop('LongName')

    # separate into train and test datasets.
    train_x = complete_dataset.sample(frac=0.8,random_state=0)
    test_x = complete_dataset.drop(train_x.index)   # remove all training observations.

    # split x and y values.
    train_y = train_x.pop('ReuseRate')
    test_y = test_x.pop('ReuseRate')

    # return the data split into test and training X and Y values.
    return (train_x, test_x, train_y, test_y)

"""
A utility method to create a tf.data dataset from a Pandas Dataframe.
https://www.tensorflow.org/tutorials/structured_data/feature_columns
"""
def df_to_dataset(dataframe, shuffle=True, batch_size=DEFAULT_BATCH_SIZE):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds
