import warnings
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from time import time

from .location_filtering import is_within_m25
from .crime_category_mapper import CrimeMapper
from . import file_handler as fh

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(5)


def _load_csv_raw(path):
    """Loads a signle CSV file into a Pandas dataframe."""
    df = pd.read_csv(path, usecols=['Month', 'Crime type', 'Latitude', 'Longitude'])

    # filter out rows without a lat/lon
    df = df.loc[pd.notna(df['Latitude']) & pd.notna(df['Longitude'])]

    # filter out locations that aren't within the M25
    df = df[df.apply(lambda row: is_within_m25(row['Latitude'], row['Longitude']), axis=1)]

    return df


def get_raw_data(glob_path='*/*'):
    """
    Parses the glob path for CSV files and returns their dataframe representation.

    :param glob_path: A path accepted by the glob() function to search for CSV files in
    :return: Pandas dataframe object
    """
    start = time()

    path = fh.input_data_path(glob_path)
    data = pd.concat(map(_load_csv_raw, glob(path)))

    print('raw data loading finished in %.3f seconds' % (time() - start))

    return data


def get_normalised_data(glob_path='*/*'):
    """
    Parses the glob path for CSV files and returns their dataframe representation, after normalising the data.
    This includes:
     - mapping crime categories
     - dividing the year and month into separate columns
     - clustering the location or replacing it with one-hot encoding

     Additionally it attempts to use cached dataframe files, if allowed.

    :param glob_path: A path accepted by the glob() function to search for CSV files in
    :return: Numpy array
    """
    print('data loading started')
    start = time()

    # check cache first
    cached_data = fh.load_cached_data_file(glob_path)
    if cached_data is not None:
        print('cached data loading finished in %.3f seconds' % (time() - start))
        return cached_data[:, 1:], cached_data[:, 0]

    data_frame = get_raw_data(glob_path=glob_path).loc[:, ['Crime type', 'Month', 'Year', 'Latitude', 'Longitude']]

    clf = fh.load_model(fh.location_clusterer_path())

    data_frame['Crime type'] = data_frame['Crime type'].apply(CrimeMapper.map)
    data_frame['Year'] = data_frame['Month'].apply(lambda month: datetime.strptime(month, '%Y-%m').year)
    data_frame['Month'] = data_frame['Month'].apply(lambda month: datetime.strptime(month, '%Y-%m').month)

    # use kmeans to find out which cluster this coordinate belongs to
    data_frame['Location'] = data_frame.apply(lambda row: clf.predict([[row['Latitude'], row['Longitude']]]).item(0), axis=1)

    data_matrix = data_frame.as_matrix(columns=['Crime type', 'Month', 'Year', 'Location']).astype(np.float64)

    # cache data to disk for later reuse
    fh.save_data_to_cache(glob_path, data_matrix)

    print('data loading finished in %.3f seconds' % (time() - start))

    return data_matrix[:, 1:], data_matrix[:, 0]
