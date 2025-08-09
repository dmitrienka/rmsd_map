#!/usr/bin/env python3

import pytest
import numpy as np
import jax
from pathlib import Path
from typing import List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mol_io.cor_reader import Fragment

# Configure JAX for testing
jax.config.update('jax_enable_x64', False)  # Use float32 for speed
jax.config.update('jax_platform_name', 'cpu')  # CPU for deterministic tests

@pytest.fixture
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def simple_fragment():
    """Create a simple water molecule fragment"""
    return Fragment(
        id="WATER",
        elements=np.array(['O', 'H', 'H'], dtype='U2'),
        coords=np.array([
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0]
        ], dtype=np.float32),
        codes=np.array([1, 2, 3], dtype=np.int32)
    )

@pytest.fixture
def benzene_fragment():
    """Create a benzene fragment with D6h symmetry"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    coords = np.column_stack([
        1.4 * np.cos(angles),
        1.4 * np.sin(angles),
        np.zeros(6)
    ])
    return Fragment(
        id="BENZENE",
        elements=np.array(['C']*6, dtype='U2'),
        coords=coords.astype(np.float32),
        codes=np.arange(1, 7, dtype=np.int32)
    )

@pytest.fixture
def sample_cor_content():
    """Sample COR file content for testing"""
    return """DAMWOH  **FRAG**        1
C21          0.65694   1.49897   7.03071     1555042
C1           0.10962   2.91199   7.10263     1555008
O1          -0.78510   3.32435   6.39515     1555002
O3           0.72956   3.64629   7.98917     1555004
ECAKIG  **FRAG**        1
C2          -0.89009   7.67558   4.90279     1555002
C1%         -1.04172   6.34101   5.46970     1555001
O2          -1.14015   6.25790   6.76068     1555010
O1          -1.08433   5.36047   4.75329     1555009
"""

@pytest.fixture
def create_test_cor_file(tmp_path, sample_cor_content):
    """Create a temporary COR file for testing"""
    cor_file = tmp_path / "test.cor"
    cor_file.write_text(sample_cor_content)
    return cor_file
