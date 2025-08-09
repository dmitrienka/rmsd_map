#!/usr/bin/env python3

import numpy as np
from umap import UMAP
import seaborn as sns
import polars as pl

with np.load('distance_matrix.npz') as data:
    names = data['names']
    distances = data["distances"]

# fixing nans

max_dist = np.nanmax(distances)
nan_i, nan_j = np.where(np.isnan(distances))
distances[nan_i, nan_j] = max_dist
reducer = UMAP(metric='precomputed', n_neighbors=500)
fited = reducer.fit_transform(distances)
