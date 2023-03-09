# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:58:12 2023

@author: zhuye
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances 
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial import distance
from sklearn.utils import parallel_backend
import joblib

## prepare data
actor_event = pd.read_csv('actor&event.csv')[['data_id', 'event_type', 'interaction']]
time_faltality = pd.read_csv('time&faltatliy.csv')[['data_id','#event_within_month','tranformed_fatality', 'days_from_19970101']]
geolocation = pd.read_csv('actor_event_geolocation_data.csv')[['data_id','region','country','latitude','longitude']]

final_data = pd.merge(actor_event, time_faltality, on = 'data_id')
final_data = pd.merge(final_data, geolocation, on = 'data_id')
final_data = final_data.drop(['data_id'], axis = 1)
final_data['interaction'] = final_data['interaction'].astype('str')


cat_vars = ['event_type', 'interaction',  'region', 'country']
num_vars = ['#event_within_month', 'tranformed_fatality', 'days_from_19970101','latitude', 'longitude']

X_numeric = final_data[num_vars].values
X_categorical = final_data[cat_vars].values

## standardize numerical varialbes
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

## onehot encode categorical variables
encoder = OneHotEncoder()
X_categorical_encoded = encoder.fit_transform(X_categorical)

## concatenate numerical and one_hot encoded variables
X = np.concatenate([X_numeric_scaled, X_categorical_encoded.toarray()], axis = 1)

## define a customized distance metric
def customized_dist(x, y, w1=0.5, w2=0.5):
    # the first 5 elements are numeric variables
    x_num = x[:5]
    y_num = y[:5]
    x_dummy = x[5:]
    y_dummy = y[5:]
    cosin_dist = distance.cosine(x_num, y_num)
    jaccard_dist = distance.jaccard(x_dummy, y_dummy)
    
    return w1*0.5*cosin_dist + w2*jaccard_dist



def gower_distances(X, Y=None, categorical_features=None, scale=True,
                    
                    
                    min_values=None, scale_factor=None):
    """Compute the distances between the observations in X and Y,
    that may contain mixed types of data, using an implementation
    of Gower formula.
    Parameters
    ----------
    X : {array-like, pandas.DataFrame} of shape (n_samples, n_features)
    Y : {array-like, pandas.DataFrame} of shape (n_samples, n_features), \
        default=None
    categorical_features : array-like of str, array-like of int, \
            array-like of bool, slice or callable, default=None
        Indexes the data on its second axis. Integers are interpreted as
        positional columns, while strings can reference DataFrame columns
        by name.
        A callable is passed the input data `X` and can return any of the
        above. To select multiple columns by name or dtype, you can use
        :obj:`~sklearn.compose.make_column_selector`.
        By default all non-numeric columns are considered categorical.
    scale : bool, default=True
        Indicates if the numerical columns should be scaled to [0, 1].
        If ``False``, the numerical columns are assumed to be already scaled.
        The scaling factors, _i.e._ ``min_values`` and ``scale_factor``, are
        taken from ``X``. If ``X`` and ``Y`` are both provided, ``min_values``
        and ``scale_factor`` have to be provided as well.
    min_values : ndarray of shape (n_features,), default=None
        Per feature adjustment for minimum. Equivalent to
        ``min_values - X.min(axis=0) * scale_factor``
        If provided, ``scale_factor`` should be provided as well.
        Only relevant if ``scale=True``.
    scale_factor : ndarray of shape (n_features,), default=None
        Per feature relative scaling of the data. Equivalent to
        ``(max_values - min_values) / (X.max(axis=0) - X.min(axis=0))``
        If provided, ``min_values`` should be provided as well.
        Only relevant if ``scale=True``.
    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
    References
    ----------
    Gower, J.C., 1971, A General Coefficient of Similarity and Some of Its
    Properties.
    Notes
    -----
    Categorical ordinal attributes should be treated as numeric for the purpose
    of Gower similarity.
    Current implementation does not support sparse matrices.
    This implementation modifies the Gower's original similarity measure in
    the sense that this implementation returns 1-S.
    """
    def _nanmanhatan(x, y):
        return np.nansum(np.abs(x - y))

    def _non_nans(x, y):
        return len(x) - np.sum(_object_dtype_isnan(x) | _object_dtype_isnan(y))

    def _nanhamming(x, y):
        return np.sum(x != y) - np.sum(
            _object_dtype_isnan(x) | _object_dtype_isnan(y))

    if issparse(X) or issparse(Y):
        raise TypeError("Gower distance does not support sparse matrices")

    if X is None or len(X) == 0:
        raise ValueError("X can not be None or empty")

    if scale:
        if (scale_factor is None) != (min_values is None):
            raise ValueError("min_value and scale_factor should be provided "
                             "together.")

    # scale_factor and min_values are either both None or not at this point
    if X is not Y and Y is not None and scale_factor is None and scale:
        raise ValueError("`scaling_factor` and `min_values` must be provided "
                         "when `Y` is provided and `scale=True`")

    if callable(categorical_features):
        cols = categorical_features(X)
    else:
        cols = categorical_features
    X_cat, X_num = _split_categorical_numerical(X, categorical_features=cols)
    Y_cat, Y_num = _split_categorical_numerical(Y, categorical_features=cols)

    if min_values is not None:
        min_values = np.asarray(min_values)
        scale_factor = np.asarray(scale_factor)
        check_consistent_length(min_values, scale_factor,
                                np.ndarray(shape=(X_num.shape[1], 0)))

    if X_num.shape[1]:
        X_num, Y_num = check_pairwise_arrays(X_num, Y_num, precomputed=False,
                                             dtype=float,
                                             force_all_finite=False)
        if scale:
            scale_data = X_num if Y_num is X_num else np.vstack((X_num, Y_num))
            if scale_factor is None:
                trs = MinMaxScaler().fit(scale_data)
            else:
                trs = MinMaxScaler()
                trs.scale_ = scale_factor
                trs.min_ = min_values
            X_num = trs.transform(X_num)
            Y_num = trs.transform(Y_num)

        nan_manhatan = distance.cdist(X_num, Y_num, _nanmanhatan)
        # nan_manhatan = np.nansum(np.abs(X_num - Y_num))
        valid_num = distance.cdist(X_num, Y_num, _non_nans)
    else:
        nan_manhatan = valid_num = None

    if X_cat.shape[1]:
        X_cat, Y_cat = check_pairwise_arrays(X_cat, Y_cat, precomputed=False,
                                             dtype=np.object,
                                             force_all_finite=False)
        nan_hamming = distance.cdist(X_cat, Y_cat, _nanhamming)
        valid_cat = distance.cdist(X_cat, Y_cat, _non_nans)
    else:
        nan_hamming = valid_cat = None

    # based on whether there are categorical and/or numerical data present,
    # we compute the distance metric
    # Division by zero and nans warnings are ignored since they are expected
    with np.errstate(divide='ignore', invalid='ignore'):
        if valid_num is not None and valid_cat is not None:
            D = (nan_manhatan + nan_hamming) / (valid_num + valid_cat)
        elif valid_num is not None:
            D = nan_manhatan / valid_num
        else:
            D = nan_hamming / valid_cat
    return D
# =============================================================================
# nn = NearestNeighbors(n_neighbors=5, metric=customized_dist, metric_params={'w1':0.5, 'w2':0.5})
# nn.fit(X)
# 
# distances = nn.radius_neighbors_graph(X)
# db = DBSCAN(metric='precomputed')
# db.fit(distances)
# =============================================================================




dbscan = DBSCAN(eps=0.01, min_samples=5, metric=customized_dist, metric_params={'w1':0.5, 'w2':0.5},
                    n_jobs = -1)
clusters = dbscan.fit_predict(X[:10000,])
print('done')

# =============================================================================
# mbk = MiniBatchKMeans(n_clusters=5, batch_size=5000, me)
# clusters = mbk.fit_predict(X)
# =============================================================================



# =============================================================================
# A = kneighbors_graph(X, 5)
# distances = pdist(X, metric='euclidean')
# 
# # Use Ward's method to build a hierarchical clustering tree
# linkage_matrix = linkage(distances, method='ward')
# 
# # Determine the number of clusters using the dendrogram and a threshold height
# threshold = 5.0
# clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
# 
# spectral = SpectralClustering(n_clusters=5, affinity='precomputed_nearest_neighbors', n_neighbors = 5)
# # Fit the estimator
# labels = spectral.fit_predict(A)
# =============================================================================
