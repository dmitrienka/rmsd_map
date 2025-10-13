#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from rmsd_map.mol_io.fragment import Fragment
from rmsd_map.rmsd.kabsch_jax import calculate_pairs, kabsch_full_proper, kabsch_full_improp, kabsch_rmsd_improp, Reporter

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
                 batch_size = 4096,
                 rmsd_kernel = kabsch_full_improp):
    AB = jax.device_put(coords) # AB: (M, N, 3)
    perm_d  = jax.device_put(perm)
    AB = AB - AB.mean(axis=1, keepdims=True)
    M , _, _ = AB.shape
    pairs = align_gen_batched(M, center, batch_size)
    for jax_dict in calculate_pairs(AB, perm_d, pairs, rmsd_kernel):
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


def partial_align_fragments(fragments, subset_idx, n_center=0, batch_size=4096):
    perms = np.expand_dims(np.arange(len(subset_idx)), axis=0)
    coords_np = np.stack([f.coords[subset_idx] for f in fragments], axis=0)
    mean_reduced = coords_np.mean(axis = 1)
    fragments_aligned = np.zeros_like(fragments)
    for batch in align_stream(coords_np, perms, center=n_center,
                              batch_size=batch_size, rmsd_kernel=kabsch_full_proper):
        for i in batch["i"]:
            rot = np.asarray(batch['Rotation'][i])
            coords_new = fragments[i].coords - mean_reduced[i]
            coords_new =  np.einsum("ji,nj->ni", rot, coords_new)
            fragment = fragments[i].replace(coords = coords_new)
            fragments_aligned[i] = fragment
    return fragments_aligned

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


def chain_fragments_naive(fragments):
    M = len(fragments)
    coords_np = np.stack([f.coords for f in fragments], axis=0)
    AB_old = jax.device_put(coords_np)
    AB_new = jnp.zeros(AB_old.shape, dtype = AB_old.dtype)
    AB_new = AB_new.at[0].set(AB_old[0])
    perms = jax.device_put(fragments[0].perms())
    new_fragments = np.empty(len(fragments), dtype = "object")
    new_fragments[0] = fragments[0]
    for i in jnp.arange(1, M):
        A = AB_new[i - 1:i]  # Add batch dimension: (1, N, 3)
        B = AB_old[i:i+1]    # Add batch dimension: (1, N, 3)
        r = kabsch_full_improp(A, B, perms)['Transformed']
        AB_new = AB_new.at[i].set(r[0])     # Remove batch dimension: (N, 3)
        new_fragments[i] = fragments[i].replace(coords = np.asarray(r[0]))
    return new_fragments


def chain_fragments(fragments):
    M = len(fragments)
    coords_np = np.stack([f.coords for f in fragments], axis=0)
    AB_old = jax.device_put(coords_np)
    perms = jax.device_put(fragments[0].perms())
    def next(carry, x):
        A = jnp.expand_dims(x, 0)
        B = jnp.expand_dims(carry, 0)
        r = kabsch_full_improp(A, B, perms)['Transformed'][0]
        return (r, r)
    _, AB_new = jax.lax.scan(next, AB_old[0], AB_old)
    new_fragments = np.empty(len(fragments), dtype = "object")
    for i in jnp.arange(M):
        new_fragments[i] = fragments[i].replace(coords = np.asarray(AB_new[i]))
    return new_fragments
