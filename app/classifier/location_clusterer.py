from sklearn.cluster import KMeans
from time import time

from .data_loader import get_raw_data
from . import file_handler as fh

lat_lon = get_raw_data().loc[:, ['Latitude', 'Longitude']]

start = time()

clf = KMeans(n_clusters=20).fit(lat_lon)

print('Clustering finished in %.3f seconds' % (time() - start))

fh.save_model(clf, fh.location_clusterer_path())
