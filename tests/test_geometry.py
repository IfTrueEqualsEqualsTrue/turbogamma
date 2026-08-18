import numpy as np
import pytest
from scipy.spatial import cKDTree

from turbogamma.geometry import shell_offsets, radii_schedule

SCHEDULE_ATOL = 1e-6


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


@pytest.fixture(params=[2.0, 3.0], ids=lambda d: f"dta{d}")
def dta(request) -> float:
    return request.param


@pytest.fixture(params=[2, 10], ids=lambda i: f"i{i}")
def interp_fraction(request) -> float:
    return request.param


@pytest.fixture(params=[1.0, 10.0], ids=lambda d: f"dmax{d}")
def d_max(request) -> float:
    return request.param


@pytest.fixture
def schedule(dta: float, interp_fraction: int, d_max: float) -> np.ndarray:
    return radii_schedule(dta, interp_fraction, d_max)


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


class TestRadiiSchedule:

    def test_starts_at_zero(self, schedule):
        assert schedule[0] == pytest.approx(0.0, abs=SCHEDULE_ATOL)

    def test_raises_on_null_interp_factor(self):
        with pytest.raises(ValueError):
            radii_schedule(3.0, 0, 10.0)

    def test_raises_on_uninteger_interp_factor(self):
        with pytest.raises(TypeError):
            radii_schedule(3.0, 1.5, 10.0)

    def test_sctricly_increasing(self, schedule):
        assert np.all(np.diff(schedule) > 0)

    def test_dmax_bound(self, schedule, d_max):
        assert schedule[-1] <= d_max

    def test_gap_smaller_than_step(self, schedule, dta, interp_fraction):
        step = dta / interp_fraction
        assert np.all(np.diff(schedule) <= step + SCHEDULE_ATOL)

    @pytest.mark.parametrize("d_max", [0.0])
    def test_null_dmax(self, schedule):
        assert len(schedule) == 1
        assert schedule[0] == [0.0]

    def test_mutliples_of_dta_are_present(self, schedule, dta, d_max):
        n_dta = int(np.floor(d_max / dta)) + 1
        targets_dta = np.arange(n_dta) * dta
        diffs = np.abs(schedule[:, None] - targets_dta).min(axis=0)  # Distances of the target to their closest match
        assert np.all(diffs < SCHEDULE_ATOL)

    @pytest.mark.parametrize("interp_fraction", [1])
    def test_unity_interp_fraction(self, schedule, dta, d_max):
        n_dta = int(np.floor(d_max / dta)) + 1
        assert len(schedule) == n_dta
        assert np.allclose(schedule, dta * np.arange(n_dta), atol=SCHEDULE_ATOL)
