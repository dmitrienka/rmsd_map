#!/usr/bin/env python3

import math

import datashader as ds
from datashader.mpl_ext import dsshow
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from umap import UMAP
import multiprocessing as mp
from functools import partial

import argparse
from pathlib import Path
# reading data matrix

def make_umap_df(neighbors, distances, densmap=False, seed=42):
    reducer = UMAP(metric='precomputed',
                   n_neighbors=neighbors,
                   densmap=densmap,
                   random_state=seed,
                   n_jobs=1)
    reducer.fit(distances)
    df = pl.DataFrame(reducer.embedding_, schema=["X", "Y"])
    df = df.with_columns(pl.lit(neighbors).alias("N"))
    return df

def read_distances_npz(npz_file):
    with np.load(npz_file) as data:
        names = data['names']
        distances = data["distances"]
    max_dist = np.nanmax(distances)
    nans = np.isnan(distances)
    nans_sum = np.sum(nans)
    if nans_sum > 0:
        print(f'There are {nans_sum} NaNs in the data matrix!')
        print(np.where(nans))
    nan_i, nan_j = np.where(nans)
    distances[nan_i, nan_j] = max_dist
    return names, distances


def generate_n_neighbors_list(max_neighbors: int = 500,
                              min_neighbors: int = 4):
    neighbors_list = []
    # Small values: < 10
    neighbors_list.extend(range(min_neighbors, min(10, max_neighbors)))
    # Medium values: 10-90 by 10s
    if max_neighbors >= 10:
        neighbors_list.extend(range(10, min(100, max_neighbors), 10))
    # Large values: 100-max_neighbors by 50s
    if max_neighbors >= 100:
        neighbors_list.extend(range(100, max_neighbors + 1, 50))
    return neighbors_list

def plot_umap_pages(df, output_file, grid_W = 2, grid_H = 3,
                    fig_W = 600, fig_H = 600,
                    page_W = 8.27, page_H = 11.69):
    slots_per_page = grid_W * grid_H
    N_NEIGHBORS_LIST = np.sort(np.unique(df["N"]))
    N_NEIGHBORS = len(N_NEIGHBORS_LIST)
    n_pages = math.ceil(N_NEIGHBORS / slots_per_page)

    with PdfPages(output_file) as pdf:
        for p in range(n_pages):
            fig, axes = plt.subplots(nrows=grid_H, ncols=grid_W,
                                 figsize=(page_W, page_H),
                                 squeeze=False)
            start = p * slots_per_page
            stop  = min(start + slots_per_page, N_NEIGHBORS)
            for i in range(slots_per_page):
                r, c = divmod(i, grid_W)
                ax = axes[r][c]
                idx = start + i
                if idx < stop:
                    NN = N_NEIGHBORS_LIST[idx]
                    coords = df.filter(N = NN).select(["X","Y"]).to_pandas()
                    ax.set_title(f'n_neighbors={NN}')
                    _ = dsshow(coords, ds.Point("X", "Y"), ax = ax,
                               plot_width = fig_W, plot_height = fig_H, aspect = 'auto')
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
       description='Calculate UMAP embeddings for multiple n_neighbors values',
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', type=Path,
                        help='Input NPZ file with distance matrix')
    parser.add_argument('-o', '--output-prefix', type=str, default='umap_results',
                        help='Output file prefix for CSV and PDF files')
    parser.add_argument('-d', '--densmap', action='store_false',
                        help='Use densmap variant of UMAP'),
    parser.add_argument('-s', '--seed', type = int, default=42,
                        help='Random seed for reproducible UMAP runs')
    group_neighbors = parser.add_mutually_exclusive_group()
    group_neighbors.add_argument('--neighbors', type=lambda s: [int(x) for x in s.split(',')],
                    help='Comma-separated list of n_neighbors values: "5,10,15,20,50,100"')
    group_neighbors.add_argument('--auto-neighbors', action='store_true',
                        help='Use automatic n_neighbors generation (default)')
    parser.add_argument('--min-neighbors', type=int, default=4,
                        help='Minimum n_neighbors value (for auto mode)')
    parser.add_argument('--max-neighbors', type=int, default=500,
                        help='Maximum n_neighbors value (for auto mode)')

    args = parser.parse_args()

    _, distances = read_distances_npz(args.input_file)
    N_MOLS = distances.shape[0]

    if args.neighbors:
        N_NEIGHBORS_LIST = args.neighbors
    else:
        N_NEIGHBORS_LIST = generate_n_neighbors_list(min(args.max_neighbors, N_MOLS // 2),
                                                     args.min_neighbors)

    dfs =[]
    with mp.Pool() as pool:
        runner = partial(make_umap_df, distances=distances, densmap=args.densmap, seed=args.seed)
        dfs = pool.map(runner, N_NEIGHBORS_LIST)
#    for N in N_NEIGHBORS_LIST:
#        dfs.append(make_umap_df(distances, N, densmap=args.densmap, seed=args.seed))
    data = pl.concat(dfs, how = "vertical")
    data.write_csv(f'{args.output_prefix}.csv')
    plot_umap_pages(data, f'{args.output_prefix}.pdf')

if __name__ == '__main__':
    main()
