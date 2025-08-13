import numpy as np
from dataclasses import dataclass
from typing import List
import re

@dataclass
class Fragment:
    """Container for molecular fragment data"""
    id: str
    elements: np.ndarray  # (N,) str
    coords: np.ndarray    # (N,3) float32

ELEMENT_PATTERN = re.compile(r'^([A-Z][a-z]?)')

def extract_element(atom_name: str) -> str:
    """Extract element symbol from atom name like 'C21', 'O1%', etc."""
    match = ELEMENT_PATTERN.match(atom_name)
    if match:
        return match.group(1)
    # Fallback for weird cases
    return atom_name[:2].strip()

def read_cor_file(filename: str) -> List[Fragment]:
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    fragments = []
    i = 0

    while i < len(lines):
        if '**FRAG**' in lines[i]:
            # Parse header
            p = r'[ \*]+'
            head = re.split(p, lines[i])
            frag_id = head[0] + "_" + head[2]
            i += 1

            # Find extent of this fragment
            start = i
            while i < len(lines) and '**FRAG**' not in lines[i]:
                i += 1

            # Parse atom block in one go
            atom_lines = lines[start:i]
            if atom_lines:
                # Pre-allocate arrays
                n_atoms = len(atom_lines)
                elements = np.empty(n_atoms, dtype='U2')
                coords = np.empty((n_atoms, 3), dtype=np.float32)
                codes = np.empty(n_atoms, dtype=np.int32)

                # Parse all at once
                for j, line in enumerate(atom_lines):
                    parts = line.split()
                    elements[j] = extract_element(parts[0])
                    coords[j] = parts[1:4]
                
                fragments.append(Fragment(
                    id=frag_id,
                    elements=elements,
                    coords=coords,
                ))
        else:
            i += 1

    return fragments
