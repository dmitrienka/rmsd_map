#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from itertools import batched
from datetime import datetime
from flax.jax_utils import prefetch_to_device
jax.config.update('jax_enable_x64', True)

@jax.jit
def kabsch_core_rmsd_improp(A, B, perms): # A:(B,N,3) B:(B,N,3) perm:(P,N)
    N = A.shape[1]
    Ap = jnp.take(A, perms, axis = 1) # Ap: (B, P, N, 3)
    Rp = jnp.einsum("bpij,bik->bpjk", Ap, B) #Rp: (B, P, 3, 3)
    S = jnp.linalg.svd(Rp, compute_uv = False) #S: (B, P, 3)
    S_term = jnp.max(jnp.sum(S, axis = 2), axis = 1) * 2 / N
    return S_term  # Rmsd = A_term + B_term - S_term; A_term = jnp.sum(Acs**2, axis = [1,2])/N

def triu_gen_batched(M, k, batch_size):
    rows = []
    cols = []
    for i in range(M):
        for j in range(i + k, M):
            rows.append(i)
            cols.append(j)

            if len(rows) == batch_size:
                batch_np = np.vstack((np.array(rows), np.array(cols)))
                yield np.expand_dims(batch_np, axis=0)
                rows.clear()
                cols.clear()
    if rows:
        batch_np = np.vstack((np.array(rows), np.array(cols)))
        yield np.expand_dims(batch_np, axis=0)
        

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


def run_distance_matrix_async(coords_np, perm_np, batch_size, report):
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
        S_term = kabsch_core_rmsd_improp(A, B, perm_d)
        rmsd = jnp.sqrt(AB_terms[i] + AB_terms[j] - S_term)

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
