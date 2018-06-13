import os, json
import re
import pickle
from sklearn.externals import joblib


def project_root_path(*args):
    """
    Joins paths relative to the root directory of this project.

    E.g. project_root_path('app', 'classifier', 'file_handler.py') will return
    an absolute path to this file with the OS-correct separators.

    :param args: path segments (without a separator)
    :return: Resulting path
    """
    dir = os.path.dirname(os.path.dirname(__file__))
    dir = os.path.dirname(dir)
    return os.path.join(dir, *args)


def _pickle_path(file_name):
    """Returns an absolute path to the specified pickle file."""
    return project_root_path('pickles', file_name)


def cache_path(file_name):
    """Returns an absolute path to the specified cache file."""
    return project_root_path('cache', file_name)


def input_data_path(file_name):
    """Returns an absolute path to the specified pickle file."""
    return project_root_path('data', file_name)


def _generate_cache_data_path(glob_path):
    """
    Generates a "normalised" path to save a cache file to.

    :param glob_path: A path accepted by the glob() function to search for CSV files in
    :return: Absolute path
    """
    glob_path = re.sub('\W', '', glob_path) if glob_path != '*/*' else 'all'

    file_name = 'path=%s.pkl' % glob_path

    return cache_path(file_name)


def load_cached_data_file(glob_path):
    """
    Unpickles a cached file.

    :param glob_path: A path accepted by the glob() function to search for CSV files in
    :return: Unpickled object
    """
    path = _generate_cache_data_path(glob_path)

    if not os.path.isfile(path):
        return None

    with open(path, 'rb') as file:
        return pickle.load(file)


def save_data_to_cache(glob_path, data):
    """
    Pickles and saves a data cache file.

    :param glob_path: A path accepted by the glob() function to search for CSV files in
    :param data: The object to pickle
    """
    path = _generate_cache_data_path(glob_path)

    with open(path, 'wb') as file:
        pickle.dump(data, file)


def location_clusterer_path():
    """Returns the absolute path to the pickled K-means clusterer object."""
    return _pickle_path('location-clusterer.pkl')


def crime_classifier_path():
    """Returns the absolute path to the pickled crime classifier object."""
    return _pickle_path('crime-classifier.pkl')


def load_model(path):
    """Loads a scikit-learn model from the provided absolute path."""
    return joblib.load(path)


def save_model(clf, path):
    """Saves a scikit-learn model to the provided absolute path."""
    joblib.dump(clf, path)


def load_m25_coordinates():
    """Returns a list of the M25 coordinates."""
    with open(input_data_path('m25-coordinates.json')) as file:
        return json.load(file)
