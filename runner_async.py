#!/usr/bin/env python3

from datetime import datetime
import numpy as np
import polars as pl
import jax.numpy as jnp
from mol_io.cor_reader import read_cor_file
from graph import rdkit_symmetry as sym
from kabsch import kabsch_jax as kj

print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Reading and preparing data")

acids = read_cor_file("./acids_rwp5_all_noh.cor")
frags_np = jnp.stack([f.coords for f in acids], axis=0)
names = np.array([a.id for a in acids])
NFrags = names.shape[0]

mol = sym.fragment_to_mol(acids[1])
perms = sym.get_permutations(mol)

print(f"Set up a calculation with {NFrags} fragments, {perms.shape[1]} atoms and {perms.shape[0]} symmetries")
print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Starting distance calculations")

res = kj.run_distance_matrix_async(frags_np, perms, 200000, report = 5)

print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Saving data")
print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Creating DataFrame")

df = pl.concat([
    pl.DataFrame(
        {
            'ID_A': names[i],
            'ID_B': names[j],
            'Dist': rmsd
        }
    ) for (i, j, rmsd) in res
], how="vertical")

print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Saving DataFrame")

df.write_csv("distance_matrix_rwp5.csv")


print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Creating matrix")

mat = np.zeros((NFrags, NFrags), dtype=float)

for i, j, rmsd in res:
    mat[i, j] = rmsd
    mat[j, i] = rmsd

print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Saving matrix")

np.savez("distance_matrix_rwp5.npz", names = names, distances = mat)

print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Finished!")

