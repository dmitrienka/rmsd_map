#!/usr/bin/env python3

import numpy as np
from rdkit import Chem, Geometry
from rdkit.Chem import rdDetermineBonds

def fragment_to_mol(fragment, covalent_factor = 1.3):
    mol = Chem.RWMol()
    for element in fragment.elements:
        atom = Chem.Atom(element)
        mol.AddAtom(atom)
    conf = Chem.Conformer(len(fragment.elements))
    for i, coord in enumerate(fragment.coords):
        point = Geometry.Point3D(*[float(x) for x in coord])
        conf.SetAtomPosition(i, point)
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(mol, covFactor = covalent_factor)
    return mol

def get_permutations(mol):
    # need some guardrails for combinatorial explosion here
  ts =  mol.GetSubstructMatches(mol, uniquify = False)
  return np.array(ts)
