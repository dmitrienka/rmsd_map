#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from rmsd_map.mol_io.fragment import Fragment
from rmsd_map.rmsd.kabsch_jax import calculate_pairs, kabsch_full_improp, kabsch_rmsd_improp, Reporter

def distance_matrix_stream(coords,
                           perm,
                           batch_size,
                           report_spacing):
    AB = jax.device_put(coords) # AB: (M, N, 3)
    perm_d  = jax.device_put(perm)
    AB = AB - AB.mean(axis=1, keepdims=True)
    M , N, _ = AB.shape
    T = M * (M-1) // 2
    pairs = triu_gen_batched(M, 1, batch_size)
    logger = Reporter(T, report_spacing)
    for jax_dict in calculate_pairs(AB, perm_d, pairs, kabsch_rmsd_improp, logger):
        np_dict = {k:np.asarray(v) for k,v in jax_dict.items()}
        yield np_dict

def distance_matrix_fragments(fragments, batch_size, report_spacing):
    NFrags = len(fragments)
    names = np.array([a.id for a in fragments])
    coords_np = np.stack([f.coords for f in fragments], axis=0)
    perms = fragments[0].perms()
    dfs = []
    mat = np.zeros((NFrags, NFrags), dtype=float)
    for batch in distance_matrix_stream(coords_np, perms, batch_size, report_spacing):
       i = batch["i"]
       j = batch["j"]
       rmsd  = batch["RMSD"]
       mat[i, j] = rmsd
       mat[j, i] = rmsd
       df =  pl.DataFrame(
           {
               'ID_A': names[i],
               'ID_B': names[j],
               'Dist': rmsd
           })
       dfs.append(df)
    df = pl.concat(dfs, how="vertical")
    return (mat, df)

def align_stream(coords,
                 perm,
                 center = 0,
                 batch_size = 4096):
    AB = jax.device_put(coords) # AB: (M, N, 3)
    perm_d  = jax.device_put(perm)
    AB = AB - AB.mean(axis=1, keepdims=True)
    M , _, _ = AB.shape
    pairs = align_gen_batched(M, center, batch_size)
    for jax_dict in calculate_pairs(AB, perm_d, pairs, kabsch_full_improp):
        np_dict = {k:np.asarray(v) for k,v in jax_dict.items()}
        yield np_dict

def align_fragments(fragments, n_center=0, batch_size=4096):
    perms = fragments[n_center].perms()
    coords_np = np.stack([f.coords for f in fragments], axis=0)
    fragments_aligned = []
    for batch in align_stream(coords_np, perms, center=n_center, batch_size=batch_size):
        for i in batch["i"]:
            fragment = fragments[i].replace(coords = batch["Transformed"][i])
            fragments_aligned.append(fragment)
    return np.asarray(fragments_aligned, dtype=object)

def triu_gen_batched(M, k, batch_size):
    rows = []
    cols = []
    for i in range(M):
        for j in range(i + k, M):
            rows.append(i)
            cols.append(j)

            if len(rows) == batch_size:
                yield (jnp.array(rows, dtype=jnp.int32), jnp.array(cols, dtype = jnp.int32))
                rows.clear()
                cols.clear()
    if rows:
        yield (jnp.array(rows, dtype=jnp.int32), jnp.array(cols, dtype=jnp.int32))

def align_gen_batched(M, center, batch_size):
    split_by = jnp.maximum(1, M // batch_size)
    ii = jnp.array_split(
        jnp.arange(M, dtype=jnp.int32),
        split_by)
    for i in ii:
        j = jnp.repeat(center, i.shape[0]).astype(jnp.int32)
        yield (i, j)
