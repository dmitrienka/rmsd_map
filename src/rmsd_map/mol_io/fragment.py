import dataclasses
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, Geometry
from rdkit.Chem import rdDetermineBonds
import py3Dmol

@dataclass
class Fragment:
    """Container for molecular fragment data"""
    id: str
    elements: np.ndarray
    coords: np.ndarray

    def __post_init__(self):
        self._mol = None
        self._perms = None

    def mol(self, covalent_factor = 1.3):
        if self._mol:
            return self._mol
        mol = Chem.RWMol()
        for element in self.elements:
            atom = Chem.Atom(element)
            mol.AddAtom(atom)
        conf = Chem.Conformer(len(self.elements))
        for i, coord in enumerate(self.coords):
            point = Geometry.Point3D(*[float(x) for x in coord])
            conf.SetAtomPosition(i, point)
        mol.AddConformer(conf)
        rdDetermineBonds.DetermineConnectivity(mol, covFactor = covalent_factor)
        self._mol = mol
        return self._mol

    def perms(self):
        mol = self.mol()
        ts =  mol.GetSubstructMatches(mol, uniquify = False)
        self._perms = np.array(ts)
        return self._perms

    def replace(self, /, **changes):
        return dataclasses.replace(self, **changes)

    def plot(self, view):
        mol = self.mol()
        molblock = Chem.MolToMolBlock(mol)
        view.addModel(molblock, 'sdf')

    @classmethod
    def create_view(cls, width=1000, height=380):
        view = py3Dmol.view(width = width, height=height)
        view.setStyle({"line": {} })
        view.setBackgroundColor('black')
        return view

    @classmethod
    def plot_fragments(cls, fragments, view = None, **args):
        if view is None:
            view = fragments[0].create_view(**args)
        for fragment in fragments:
            fragment.plot(view)
        view.zoomTo()
        return view
