#!/usr/bin/env python3

import numpy as np
from umap import UMAP
import polars as pl

import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import datashader as ds
from datashader.mpl_ext import dsshow

# reading data matrix

with np.load('../../../tmp/distance_matrix.npz') as data:
    names = data['names']
    distances = data["distances"]

# fixing nans

max_dist = np.nanmax(distances)
nan_i, nan_j = np.where(np.isnan(distances))
distances[nan_i, nan_j] = max_dist

# calculating UMAPs

N_NEIGHBORS_LIST = list(range(4, 10)) + list(range(10, 100, 10)) + list(range(100, 501, 50))

N_NEIGHBORS_LIST = [ N for N in  N_NEIGHBORS_LIST if N < distances.shape[0] / 2  ]

N_NEIGHBORS = len(N_NEIGHBORS_LIST)

dfs =[]
for N in N_NEIGHBORS_LIST:
    reducer = UMAP(metric='precomputed', n_neighbors=N)
    reducer.fit(distances)
    df = pl.DataFrame(reducer.embedding_, schema=["X", "Y"]).with_columns(pl.lit(N).alias("N"))
    dfs.append(df)
data = pl.concat(dfs, how = "vertical")

data.write_csv("../test.csv")


# plot

W = 2
H = 3
slots_per_page = W * H
output_file = "../test.pdf"

n_pages = math.ceil(N_NEIGHBORS / slots_per_page)

with PdfPages(output_file) as pdf:
    for p in range(n_pages):
        fig, axes = plt.subplots(nrows=H, ncols=W,
                                 figsize=(8.27, 11.69),
                                 squeeze=False)
        start = p * slots_per_page
        stop  = min(start + slots_per_page, N_NEIGHBORS)
        for i in range(slots_per_page):
            r, c = divmod(i, W)
            ax = axes[r][c]
            idx = start + i
            if idx < stop:
                NN = N_NEIGHBORS_LIST[idx]
                reducer = UMAP(metric='precomputed', n_neighbors=20)
                coords = data.filter(N = NN).select(["X","Y"]).to_pandas()
                ax.set_title(f'n_neighbors={NN}')
                _ = dsshow(coords, ds.Point("X", "Y"), ax = ax,
                           plot_width = 600, plot_height = 600, aspect = 'auto')
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
