import numpy as np
import pandas as pd

def load_data(filename, num):
    filepath = '{}.csv'.format(filename)
    try:
        df = pd.read_csv(filepath,header=None)
        if num >= df.shape[0]:
            return df
        else:
            return df.iloc[:num]
    except IOError:
        print('File {} not found! Returning...'.format(filepath))

def load_test_data_df(filename, num, test_size):
    filepath = '{}.csv'.format(filename)
    try:
        df = pd.read_csv(filepath,header=None)
        if num >= df.shape[0]:
            return df
        else:
            return df.iloc[num:num+test_size]
    except IOError:
        print('File {} not found! Returning...'.format(filepath))