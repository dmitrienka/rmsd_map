#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from itertools import batched
from datetime import datetime
from flax.jax_utils import prefetch_to_device
jax.config.update('jax_enable_x64', True)

class Reporter:
    def __init__(self, expected_total, report_spacing):
        self.start_time = datetime.now()
        self.spacing = report_spacing
        self.total = expected_total
        self.log_storage = [{"Time": self.start_time, "Time_relative": 0.0, "Batches": 0, "Distances": 0 }]

    def log(self, batches, distances):
        now = datetime.now()
        time = now  - self.start_time
        log_item = {"Time": now, "Time_relative": time.total_seconds(), "Batches": batches, "Distances": distances}
        self.log_storage.append(log_item)
        self.report()

    def report(self):
        log_item = self.log_storage[-1]
        if log_item["Batches"] % self.spacing == 0:
            print( f"{log_item["Time_relative"]}s: {log_item["Distances"]}/{self.total}")

@jax.jit
def kabsch_rmsd_improp(A, # (B, N, 3), B - batch size, N - number of atoms
                       B, # (B, N, 3)
                       perms, # (P, N)
                       A_sq_sum = None, # (B,)
                       B_sq_sum = None): # (B,)
    N = A.shape[1]
    if A_sq_sum is None:
        A_sq_sum = jnp.sum(A**2, axis = [1,2])/N
    if B_sq_sum is None:
        B_sq_sum = jnp.sum(B**2, axis = [1,2])/N
    Ap = jnp.take(A, perms, axis = 1) # Ap: (B, P, N, 3)
    Rp = jnp.einsum("bpij,bik->bpjk", Ap, B) #Rp: (B, P, 3, 3)
    S = jnp.linalg.svd(Rp, compute_uv = False) #S: (B, P, 3)
    S_term = jnp.max(jnp.sum(S, axis = 2), axis = 1) * 2 / N
    MSD  = jnp.maximum(A_sq_sum + B_sq_sum - S_term, 0.0)
    RMSD = jnp.sqrt(MSD)
    return {"RMSD": RMSD}


@jax.jit
def kabsch_full_improp(A, # (B, N, 3)
                       B, # (B, N, 3)
                       perms, # (P, N)
                       A_sq_sum = None, # (B,)
                       B_sq_sum = None): # (B,)
    N = A.shape[1]
    if A_sq_sum is None:
        A_sq_sum = jnp.sum(A**2, axis = [1,2])/N
    if B_sq_sum is None:
        B_sq_sum = jnp.sum(B**2, axis = [1,2])/N
    Ap = jnp.take(A, perms, axis = 1) # Ap: (B, P, N, 3)
    Rp = jnp.einsum("bpij,bik->bpjk", Ap, B) #Rp: (B, P, 3, 3)
    U, S, tV = jnp.linalg.svd(Rp, compute_uv = True) #S: (B, P, 3), # U,tV: (B, P, 3, 3)
    permut_S_terms = jnp.sum(S, axis = 2) * 2 / N # (B,P)
    best_i = jnp.argmax(permut_S_terms, axis = 1) #(B)
    best_i_2d = jnp.expand_dims(best_i, 1)
    best_i_4d = jnp.expand_dims(best_i, (1,2,3))
    S_term = jnp.take_along_axis(permut_S_terms, best_i_2d, axis = 1) # (B,1)
    S_term = jnp.squeeze(S_term, axis = 1) # (B)
    MSD = jnp.maximum(A_sq_sum  + B_sq_sum - S_term, 0)
    RMSD = jnp.sqrt(MSD)
    Rot = jnp.einsum("bpij,bpjk->bpik", U,tV) # (B, P, 3, 3)
    Rot = jnp.take_along_axis(Rot, best_i_4d , axis = 1) # (B, 1, 3, 3)
    Rot = jnp.squeeze(Rot, axis = 1) # (B, 3, 3)
    A_best = jnp.take_along_axis(Ap, best_i_4d, axis = 1)
    A_best = jnp.squeeze(A_best, axis = 1)
    Trans = jnp.einsum("bij,bnj->bni", Rot, A_best) # (B, N, 3)
    return {'RMSD':RMSD, 'Permutation':best_i, 'Rotation':Rot, 'Transformed':Trans}


def calculate_pairs(AB,
                    permutations,
                    batched_index_gen,
                    rmsd_kernel,
                    logger = None):
    N = AB.shape[1]
    AB_terms = jnp.sum(AB**2, axis = [1,2]) / N
    buf = None
    batch_count = 0
    dist_count = 0
    for i, j in batched_index_gen:
        result = rmsd_kernel(AB[i], AB[j], permutations, AB_terms[i], AB_terms[j])
        result = {"i":i, "j":j, **result}
        batch_count += 1
        dist_count += len(i)
        if logger:
            logger.log(batches = batch_count, distances = dist_count)
        if buf is not None:
            yield jax.device_get(buf) # sync copy for previuos iteration, should be mostly free here
        for v in result.values():
            v.copy_to_host_async()  # async copy for current iteration
        buf = result
    if buf is not None:
        yield jax.device_get(buf) # yeild the last iteration


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

# def align_to_one_stream(coords_np, perm_np, center, batch_size):
#     AB = jax.device_put(coords_np) # AB: (B, N, 3)
#     perm_d  = jax.device_put(perm_np)
#     AB = AB - AB.mean(axis=1, keepdims=True)
#     M = AB.shape[0]
#     indices = np.vstack([np.arange(M), np.repeat(center, M)])
#     index_gen = np.array_split(np.expand_dims(indices, axis = 0), batch_size, axis = 2)
#     def transform(result, i,j, counter, start_time):
#         return(i, result["Transformed"])
#     return kabsch_run_batched(AB, perm_d, index_gen, kabsch_, transform)

# def run_distance_matrix_async(coords_np, perm_np, batch_size, report):
#     AB = jax.device_put(coords_np) # AB: (B, N, 3)
#     AB = AB - AB.mean(axis=1, keepdims=True)
#     M = AB.shape[0]
#     N = AB.shape[1]
#     T = M * (M-1) // 2
#     AB_terms = jnp.sum(AB**2, axis = [1,2])/N
#     perm_d  = jax.device_put(perm_np)
#     index_gen = triu_gen_batched(M, 1, batch_size)
#     def transform(result, i , j, counter, start_time):
#         if (counter % report == 0 ):
#             time = datetime.now() - start_time
#             print( f"{time.total_seconds()}s: {counter * batch_size}/{T}")
#         msd = jnp.maximum(AB_terms[i] + AB_terms[j] - result, 0)
#         rmsd = jnp.sqrt(msd)
#         return (i, j, rmsd)
#     return kabsch_run_batched(AB, perm_d, index_gen, kabsch_core_rmsd_improp, transform)

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




# without float64 sqrt errors broke
@partial(jax.jit, static_argnums=(2,))
def k_to_ij(k,M, dtype=jnp.int32):
    k = k.astype(dtype)
    i = (M - 2 - jnp.floor(
        (-1 + jnp.sqrt(
            ((2 * M - 1) ** 2 - 8 * (k + 1)).astype(jnp.float64)
        )
         ) / 2)
         ).astype(dtype)
    j = (k - i * (2 * M - i - 3) // 2 + 1).astype(dtype)
    return(i,j)


def run_distance_matrix_async_old(coords_np, perm_np, batch_size, report):
    AB = jax.device_put(coords_np) # AB: (B, N, 3)
    AB = AB - AB.mean(axis=1, keepdims=True)
    M = AB.shape[0]
    N = AB.shape[1]
    T = M * (M-1) // 2
    AB_terms = jnp.sum(AB**2, axis = [1,2])/N
    perm_d   = jax.device_put(perm_np)
    ij_ranges = triu_gen_batched(M, 1, batch_size)
    ij_ranges = prefetch_to_device(ij_ranges, 2)
    results = []
    prev_handles = None
    start_time = datetime.now()
    for counter, ij_range in enumerate(ij_ranges):
        if (counter % report == 0 ):
            time = datetime.now() - start_time
            print( f"{time.total_seconds()}s: {counter * batch_size}/{T}")
        i = ij_range[0][0]
        j = ij_range[0][1]
        A = AB[i]
        B = AB[j]
        rmsd = kabsch_rmsd_improp(A, B, perm_d, AB_terms[i], AB_terms[j])["RMSD"]

        rmsd.copy_to_host_async()
        i.copy_to_host_async()
        j.copy_to_host_async()
        current_handles = (i, j, rmsd)

        if prev_handles is not None:
            pi, pj, prmsd = prev_handles
            results.append((np.asarray(pi),
                            np.asarray(pj),
                            np.asarray(prmsd)))   # essentially free now

        prev_handles = current_handles
    if prev_handles is not None:
        pi, pj, prmsd = prev_handles
        results.append((np.asarray(pi),
                        np.asarray(pj),
                        np.asarray(prmsd)))
    return results
