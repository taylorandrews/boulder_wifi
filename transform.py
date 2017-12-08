import pandas as pd
from eda import read_data
from geopy.geocoders import Nominatim


def get_latlng(df, df_locations):
    '''

    '''

    df_locations['ADDRESS'] = df_locations['ADDRESS'].apply(lambda x: x + ' Boulder, CO USA')

    geolocator = Nominatim()

    df_locations['Latitude'] = df_locations['ADDRESS'].apply(lambda x: geolocator.geocode(x).latitude)
    df_locations['Longitude'] = df_locations['ADDRESS'].apply(lambda x: geolocator.geocode(x).longitude)

    lat_dict = pd.Series(df_locations['Latitude'].values, index = df_locations['WIFIGROUP']).to_dict()
    lng_dict = pd.Series(df_locations['Longitude'].values, index = df_locations['WIFIGROUP']).to_dict()

    df['Latitude'] = df['WiFiGroup'].apply(lambda x: lat_dict[x])
    df['Longitude'] = df['WiFiGroup'].apply(lambda x: lng_dict[x])

    return df

def get_date(df):
    '''
    INPUT
        - pandas dataframe to be transformed

    OUTPUT
        - pandas dataframe after transformation

    organizes dates into np.datetime64 format
    drops nans that exists through no listed DisconnectTime
    reindexes df
    creates Duration column of dtype timedelta64
    '''

    # reformat dates into corrent datatype
    cols_to_dt = ['ConnectTime', 'DisconnectTime']
    for col in cols_to_dt:
        df[col] = pd.to_datetime(df[col]
                                ,errors='coerce'
                                ,format="%m/%d/%Y %I:%M %p %Z")

    # drop nans
    df.dropna(axis=0, how='any', inplace=True)

    # reindex
    df.index = pd.RangeIndex(len(df.index))
    df.index = range(len(df.index))

    # adds duration column
    df['Duration'] = df['DisconnectTime'] - df['ConnectTime']

    return df

def load_transform():
    dft = read_data()
    df_dates = get_date(dft)
    df_locations = pd.read_csv('../data/public_wifi_locations.csv')
    df = get_latlng(df_dates, df_locations)
    df.drop_duplicates(inplace=True)
    # df.to_csv('~/Documents/mashey/fedex_day/data/data.csv')
    return df
