import numpy as np
import pytest
from scipy.spatial import cKDTree

from turbogamma.geometry import shell_offsets

RADIUS = 5.3
STEP = 0.5
shell_1d = shell_offsets(RADIUS, 0.5, 1)
shell_2d = shell_offsets(RADIUS, 0.5, 2)
shell_3d = shell_offsets(RADIUS, 0.5, 3)


class TestShell1d:

    @staticmethod
    def test_1d_shell_radius():
        assert np.allclose(shell_1d, RADIUS)

    @staticmethod
    def test_1d_shell_gaps():
        assert abs(shell_1d[-1] - shell_1d[0]) <= STEP

    @staticmethod
    def test_1d_shell_npoints():
        assert len(shell_1d) == 2


class TestShell2d:
    @staticmethod
    def test_2d_shell_radius():
        assert np.allclose(np.linalg.norm(shell_2d, axis=1), RADIUS)

    @staticmethod
    def test_2d_shell_gaps():
        diffs = np.abs(shell_2d[:, 1] - shell_2d[-1, :])
        assert np.max(diffs) <= STEP

    @staticmethod
    def test_2d_shell_npoints():
        expected_npoints = 2 * np.pi * RADIUS / STEP
        assert len(shell_2d) == pytest.approx(expected_npoints, rel=1e-1)


class TestShell3d:
    @staticmethod
    def test_3d_shell_radius():
        assert np.allclose(np.linalg.norm(shell_3d, axis=1), RADIUS)

    @staticmethod
    def test_3d_shell_gaps():
        tree = cKDTree(shell_3d)  # spatial index
        result = tree.query(shell_3d, k=2)  # nearest neighbors (k=1 is itself)
        distances: np.ndarray = result[0]  # all the distances to their 2 nearest neighbors
        distances_filtered = distances[:, 1]  # removing the point itself from candidates
        assert distances_filtered.max() <= STEP

    @staticmethod
    def test_3d_shell_npoints():
        expected_npoints = 4 * np.pi * RADIUS ** 2 / STEP ** 2
        assert len(shell_3d) == pytest.approx(expected_npoints, rel=1e-1)
