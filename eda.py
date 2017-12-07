import pandas as pd
import numpy as np

def read_data():
    '''
    INPUT
        none

    OUTPUT
        - pandas dataframe with concatinated data

    takes csvs with same column names, reads them into pandas and concatinates them into on df
    '''

    path_month = '../data/public_wifi_usage_30_day.csv'
    path_historic = '../data/public_wifi_usage_historic.csv'

    df_month = pd.read_csv(filepath_or_buffer=path_month)
    df_historic = pd.read_csv(path_historic)
    dfs = [df_month, df_historic]
    df_total = pd.concat(dfs, ignore_index=True)

    # dict_value_counts = get_value_counts(df_total)
    # print_num_values(dict_value_counts)

    return df_total

def get_value_counts(df):
    '''
    INPUT
        - pandas dataframe

    OUTPUT
        - dictionary. key = column name of pandas df : value = list with first item number of unique entries in that column, second item seperate list of all those uniques items

    creates a dictionary to explore the unique items in each column of dataset
    '''

    dict_value_counts = {}
    for col in df:
        values_list = list(df[col].value_counts())
        dict_value_counts[col] = [len(values_list), values_list]
    return dict_value_counts

def print_num_values(dict_value_counts):
    '''
    INPUT
        - dictionary from get_value_counts function

    OUTPUT
        - none

    prints column names that are the keys of input dictionary and first element of the keys respective value which is the number of uniques items in the pandas column
    '''

    for k, v in dict_value_counts.iteritems():
        print 'column name: {} -> {} unique values'.format(k, v[0])
