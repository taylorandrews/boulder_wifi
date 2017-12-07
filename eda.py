import pandas as pd

def read_data(paths):
    '''
    INPUT
        - list of paths to csv files where data resides. column names in csvs must be the same!

    OUTPUT
        - pandas dataframe with concatinated data

    takes csvs with same column names, reads them into pandas and concatinates them into on df
    '''

    df_month = pd.read_csv(path_month)
    df_historic = pd.read_csv(path_historic)
    dfs = [df_month, df_historic]
    df_total = pd.concat(dfs)

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

if __name__ == '__main__':
    path_month = '../data/public_wifi_usage_30_day.csv'
    path_historic = '../data/public_wifi_usage_historic.csv'
    paths = [path_month, path_historic]

    df = read_data(paths)
    dict_value_counts = get_value_counts(df)
    print_num_values(dict_value_counts)
