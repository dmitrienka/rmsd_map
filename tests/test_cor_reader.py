#!/usr/bin/env python3

import pytest
import numpy as np
from mol_io.cor_reader import read_cor_file, extract_element, Fragment

class TestElementExtraction:
    """Test element symbol extraction from atom names"""

    @pytest.mark.parametrize("atom_name,expected", [
        ("C21", "C"),
        ("O1", "O"),
        ("C33%", "C"),
        ("O14%", "O"),
        ("Cl1", "Cl"),
        ("Ca2+", "Ca"),
        ("Fe3", "Fe"),
        ("H", "H"),
        ("N12", "N"),
    ])
    def test_extract_element(self, atom_name, expected):
        assert extract_element(atom_name) == expected

    def test_extract_element_edge_cases(self):
        assert extract_element("") == ""
        assert extract_element("123") == ""  # No element
        assert extract_element("c1") == "c1"[:2].strip()  # Fallback

class TestCORReader:
    """Test COR file reading functionality"""

    def test_read_simple_cor(self, create_test_cor_file):
        fragments = read_cor_file(str(create_test_cor_file))

        assert len(fragments) == 2
        assert fragments[0].id == "DAMWOH"
        assert fragments[1].id == "ECAKIG"

        # Check first fragment
        assert len(fragments[0].elements) == 4
        assert list(fragments[0].elements) == ['C', 'C', 'O', 'O']

        # Check coordinates
        expected_coords = np.array([
            [0.65694, 1.49897, 7.03071],
            [0.10962, 2.91199, 7.10263],
            [-0.78510, 3.32435, 6.39515],
            [0.72956, 3.64629, 7.98917]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            fragments[0].coords, expected_coords, decimal=5
        )

        # Check element extraction worked on C1%
        assert fragments[1].elements[1] == 'C'

    def test_read_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.cor"
        empty_file.write_text("")
        fragments = read_cor_file(str(empty_file))
        assert len(fragments) == 0

    def test_read_malformed_cor(self, tmp_path):
        malformed = tmp_path / "malformed.cor"
        malformed.write_text("""BROKEN  **FRAG**  1
Not enough columns
C1 1.0 2.0  # Missing coordinate
""")
        fragments = read_cor_file(str(malformed))
        # Should handle gracefully, return what it can parse
        assert len(fragments) == 1
        assert len(fragments[0].elements) == 0  # No valid atoms
