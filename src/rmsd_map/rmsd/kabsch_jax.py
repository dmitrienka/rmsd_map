#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime

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
    best_i_2d = jnp.expand_dims(best_i, 1) # (B, 1)
    best_i_4d = jnp.expand_dims(best_i, (1,2,3)) # (B, 1, 1, 1)
    best_perms = jnp.take_along_axis(perms, best_i_2d, axis = 0) # (B, P)
    S_term = jnp.take_along_axis(permut_S_terms, best_i_2d, axis = 1) # (B,1)
    S_term = jnp.squeeze(S_term, axis = 1) # (B)
    MSD = jnp.maximum(A_sq_sum  + B_sq_sum - S_term, 0)
    RMSD = jnp.sqrt(MSD)
    Rot = jnp.einsum("bpij,bpjk->bpik", U,tV) # (B, P, 3, 3)
    Rot = jnp.take_along_axis(Rot, best_i_4d , axis = 1) # (B, 1, 3, 3)
    Rot = jnp.squeeze(Rot, axis = 1) # (B, 3, 3)
    A_best = jnp.take_along_axis(Ap, best_i_4d, axis = 1)
    A_best = jnp.squeeze(A_best, axis = 1) # (B, P, 3)
    Trans = jnp.einsum("bji,bnj->bni", Rot, A_best) # (B, N, 3)
    return {'RMSD':RMSD, 'Permutation':best_perms, 'Rotation':Rot, 'Transformed':Trans}


@jax.jit
def kabsch_full_proper(A, # (B, N, 3)
                       B, # (B, N, 3)
                       perms, # (P, N)
                       A_sq_sum = None, # (B,)
                       B_sq_sum = None): # (B,)
    M, N, _ = A.shape
    P, _ = perms.shape
    if A_sq_sum is None:
        A_sq_sum = jnp.sum(A**2, axis = [1,2])/N
    if B_sq_sum is None:
        B_sq_sum = jnp.sum(B**2, axis = [1,2])/N
    Ap = jnp.take(A, perms, axis = 1) # Ap: (B, P, N, 3)
    Rp = jnp.einsum("bpij,bik->bpjk", Ap, B) #Rp: (B, P, 3, 3)
    dets = jnp.linalg.det(Rp) # (B, P)
    detsigns = jnp.signbit(dets).astype(jnp.int32)  * -2 + 1 # (B, P)
    eye = jnp.tile(jnp.eye(3), (M, P, 1, 1))
    eye = eye.at[:, :, 2, 2].set(detsigns)
    U, S, tV = jnp.linalg.svd(Rp, compute_uv = True) #S: (B, P, 3), # U,tV: (B, P, 3, 3)
    S_new = S.at[:,:,2].set(S[:,:,2] * detsigns)
    permut_S_terms = jnp.sum(S_new, axis = 2) * 2 / N # (B,P)
    best_i = jnp.argmax(permut_S_terms, axis = 1) #(B)
    best_i_2d = jnp.expand_dims(best_i, 1) # (B, 1)
    best_i_4d = jnp.expand_dims(best_i, (1,2,3)) # (B, 1, 1, 1)
    best_perms = jnp.take_along_axis(perms, best_i_2d, axis = 0) # (B, P)
    S_term = jnp.take_along_axis(permut_S_terms, best_i_2d, axis = 1) # (B,1)
    S_term = jnp.squeeze(S_term, axis = 1) # (B)
    MSD = jnp.maximum(A_sq_sum  + B_sq_sum - S_term, 0)
    RMSD = jnp.sqrt(MSD)
    Rot = jnp.einsum("bpij,bpjl,bplk->bpik", U, eye, tV) # (B, P, 3, 3)
    Rot = jnp.take_along_axis(Rot, best_i_4d , axis = 1) # (B, 1, 3, 3)
    Rot = jnp.squeeze(Rot, axis = 1) # (B, 3, 3)
    A_best = jnp.take_along_axis(Ap, best_i_4d, axis = 1)
    A_best = jnp.squeeze(A_best, axis = 1) # (B, P, 3)
    Trans = jnp.einsum("bji,bnj->bni", Rot, A_best) # (B, N, 3)
    return {'RMSD':RMSD, 'Permutation':best_perms, 'Rotation':Rot, 'Transformed':Trans}


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


