import numpy as np
import pytest
from scipy.spatial import cKDTree

from turbogamma.geometry import shell_offsets


@pytest.fixture(params=[1, 2, 3], ids=lambda d: f"{d}d")
def n_dims(request) -> int:
    return request.param


@pytest.fixture(params=[0.05, 1.0, 5.3], ids=lambda r: f"r{r}")
def radius(request) -> float:
    """Non-zero radii, 0 has its dedicated test"""
    return request.param


@pytest.fixture(params=[0.2, 0.5], ids=lambda s: f"s{s}")
def step(request) -> float:
    return request.param


@pytest.fixture
def shell(radius: float, step: float, n_dims: int) -> np.ndarray:
    return shell_offsets(radius, step, n_dims)


def max_neighbour_gap(points: np.ndarray) -> float:
    """Largest distance from any point to its closest neighbor"""
    if len(points) < 2:
        return 0.0
    distances, _indices = cKDTree(points).query(points, k=2)
    return distances[:, 1].max()  # column 0 is the point itself, at distance 0


class TestShellAllDims:
    """Properties that hold in every dimension."""

    def test_points_lie_on_the_sphere(self, shell, radius):
        assert np.allclose(np.linalg.norm(shell, axis=1), radius)

    def test_shape_is_points_by_dims(self, shell, n_dims):
        assert shell.ndim == 2
        assert shell.shape[1] == n_dims

    def test_zero_radius_is_a_single_origin_point(self, step, n_dims):
        shell = shell_offsets(0.0, step, n_dims)
        assert shell.shape == (1, n_dims)
        assert np.all(shell == 0.0)


class TestShell1d:
    @pytest.fixture
    def n_dims(self) -> int:
        return 1

    def test_exactly_two_points(self, shell):
        assert len(shell) == 2

    def test_points_are_plus_and_minus_radius(self, shell, radius):
        assert sorted(shell[:, 0]) == pytest.approx([-radius, radius])


class TestShell2d:
    @pytest.fixture
    def n_dims(self) -> int:
        return 2

    def test_no_gap_exceeds_step(self, shell, step):
        assert max_neighbour_gap(shell) <= step

    @pytest.mark.parametrize("radius", [5.0, 20.0, 50.0])
    def test_point_count_scales_with_circumference(self, shell, radius, step):
        assert len(shell) == pytest.approx(2 * np.pi * radius / step, rel=0.1)


class TestShell3d:
    @pytest.fixture
    def n_dims(self) -> int:
        return 3

    def test_no_gap_exceeds_step(self, shell, step):
        assert max_neighbour_gap(shell) <= step

    @pytest.mark.parametrize("radius", [5.0, 20.0, 50.0])
    def test_point_count_scales_with_area(self, shell, radius, step):
        assert len(shell) == pytest.approx(4 * np.pi * radius ** 2 / step ** 2, rel=0.1)

    def test_both_poles_are_present(self, shell, radius):
        poles = [(0.0, 0.0, radius), (0.0, 0.0, -radius)]
        distances, _indices = cKDTree(shell).query(poles)
        assert distances.max() < 1e-9
