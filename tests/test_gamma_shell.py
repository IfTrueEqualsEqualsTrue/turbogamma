import dataclasses

import numpy as np
import pytest

from turbogamma.classes import DoseGrid, Protocol
from turbogamma.gamma_bruteforce import gamma_bruteforce
from turbogamma.gamma_shell import find_best_gamma, gamma_shell

GRID_SIZE = 10
SPACING = 2.0
REF_DOSE = 100.0
DOSE_TOLERANCE_ABS = 3.0
DOSE_OFFSET = 3.0

RAMP_CASES = [
    (1, False),
    (2, False), (2, True),
    (3, False), (3, True),
]


def make_uniform_grid(size, dose, spacing):
    coords = np.arange(size) * spacing
    return DoseGrid(coordinates=(coords,), dose=np.full(size, dose))


def unit_direction(ndim, diagonal):
    """Axis-aligned (1, 0, ...) or normalized diagonal (1,1,...)/sqrt(ndim)."""
    d = np.ones(ndim) if diagonal else np.eye(ndim)[0]
    return d / np.linalg.norm(d)


def make_ramp_grid(ndim, size, spacing, dose, a, s, direction):
    i = np.arange(size)
    coords = i * spacing
    mesh = np.meshgrid(*([coords] * ndim), indexing="ij")
    position_along_ramp = sum(g * d for g, d in zip(mesh, direction))
    dose_field = dose + a * (position_along_ramp - s)
    return DoseGrid(coordinates=(coords,) * ndim, dose=dose_field)


@pytest.fixture
def protocol():
    return Protocol(dose_difference=3.0, dta=3.0, dose_threshold=0, local=False, interp_fraction=10, max_gamma=3.0,
                    dose_tolerance_abs=3.0)


@pytest.fixture
def r_ref():
    return np.array([10.0])


class TestFindBestGammaShell:

    @staticmethod
    def test_uniform_offset(r_ref, protocol):
        """Constant ref/eval doses with no spatial variation: moving never improves the dose match, so gamma is a pure dose
        term, gamma = delta(D) / dose_tolerance_abs"""
        eval_grid = make_uniform_grid(GRID_SIZE, REF_DOSE + DOSE_OFFSET, SPACING)

        result = find_best_gamma(r_ref, REF_DOSE, DOSE_TOLERANCE_ABS, eval_grid, protocol)

        assert np.isclose(result.gamma, 1.0)  # |REF_DOSE - DOSE_OFFSET| / DOSE_TOLERANCE_ABS = 3/3 = 1

    @staticmethod
    def test_perfect_match(r_ref, protocol):
        """Identical ref/eval dose fields: gamma is exactly 0 and the search must terminate at the first shell (d=0)"""
        eval_grid = make_uniform_grid(GRID_SIZE, REF_DOSE, SPACING)

        result = find_best_gamma(r_ref, REF_DOSE, DOSE_TOLERANCE_ABS, eval_grid, protocol)

        assert np.isclose(result.gamma, 0.0)
        assert result.shells_visited == 1

    @staticmethod
    def test_out_of_bounds(protocol):
        """Reference point far outside the evaluation grid: every shell returns +inf, and the search must still terminate
        (via d_max) rather than spin"""
        eval_grid = make_uniform_grid(GRID_SIZE, REF_DOSE, SPACING)
        r_ref_far = np.array([10_000.0])  # way outside the grid extent

        result = find_best_gamma(r_ref_far, REF_DOSE, DOSE_TOLERANCE_ABS, eval_grid, protocol)

        assert np.isinf(result.gamma)

    @staticmethod
    @pytest.mark.parametrize("ndim,diagonal", RAMP_CASES)
    def test_shifted_ramp(protocol, ndim, diagonal):
        """Linear ramp shifted by s: the true optimum is a distance/dose trade-off. Checks that the search actually
           terminates near the true optimal offset t*"""
        a, s, spacing, size = 1.0, 2.0, 1.0, 40
        direction = unit_direction(ndim, diagonal)
        centre = (size // 2) * spacing
        r_ref = np.full(ndim, centre)
        x_ref = np.dot(r_ref, direction)
        dose_ref = 1.0 + a * x_ref  # ref uses s=0

        eval_grid = make_ramp_grid(ndim, size, spacing, dose=1.0, a=a, s=s, direction=direction)

        result = find_best_gamma(r_ref, dose_ref, protocol.dose_tolerance_abs, eval_grid, protocol)

        k = abs(a) * protocol.dta / protocol.dose_tolerance_abs
        expected_gamma = (abs(s) / protocol.dta) * k / np.sqrt(1 + k ** 2)
        t_star = s * k ** 2 / (1 + k ** 2)

        assert np.isclose(result.gamma, expected_gamma, rtol=0.02)
        assert np.isclose(result.terminating_radius, t_star, atol=protocol.dta / protocol.interp_fraction)

    @staticmethod
    @pytest.mark.parametrize("ndim,diagonal", RAMP_CASES)
    def test_ramp_convergence_with_interp_fraction(protocol, ndim, diagonal):
        """Running the same ramp point at increasing interp_fraction must give non-increasing gammas"""
        a, s, spacing, size = 1.0, 2.0, 1.0, 40
        direction = unit_direction(ndim, diagonal)
        centre = (size // 2) * spacing
        r_ref = np.full(ndim, centre)
        dose_ref = 1.0 + a * np.dot(r_ref, direction)

        eval_grid = make_ramp_grid(ndim, size, spacing, dose=1.0, a=a, s=s, direction=direction)

        gammas = []
        for interp_fraction in [5, 10, 20, 40]:
            fine_protocol = dataclasses.replace(protocol, interp_fraction=interp_fraction)
            result = find_best_gamma(r_ref, dose_ref, protocol.dose_tolerance_abs, eval_grid, fine_protocol)
            gammas.append(result.gamma)

        assert all(g2 <= g1 + 1e-9 for g1, g2 in zip(gammas, gammas[1:]))


class TestGammaShell:
    @staticmethod
    def test_single_point_grid_matches_find_best_gamma(protocol):
        """A ref grid with exactly one point should give the same gamma as
        calling find_best_gamma directly — proves the loop wiring is correct
        with nothing else (normalization, cutoff) interfering."""
        ref_dose = 100.0
        ref_grid = DoseGrid(coordinates=(np.array([10.0]),), dose=np.array([ref_dose]))
        eval_grid = make_uniform_grid(GRID_SIZE, ref_dose + 3.0, SPACING)

        result = gamma_shell(ref_grid, eval_grid, protocol)

        dose_tolerance_abs = protocol.dose_difference / 100 * ref_dose  # global == local, only one point
        expected = find_best_gamma(
            np.array([10.0]), ref_dose, dose_tolerance_abs, eval_grid, protocol
        ).gamma

        assert np.isclose(result.gamma[0], expected)

    @staticmethod
    def test_uniform_offset_full_grid(protocol):
        """Constant ref/eval doses, run through the full grid path. Every point
        should give the same gamma (no spatial dependence) — reuses M4's
        uniform-offset case as a whole-grid regression."""
        ref_grid = make_uniform_grid(GRID_SIZE, REF_DOSE, SPACING)
        eval_grid = make_uniform_grid(GRID_SIZE, REF_DOSE + DOSE_OFFSET, SPACING)

        result = gamma_shell(ref_grid, eval_grid, protocol)

        expected = 1.0  # |REF_DOSE - DOSE_OFFSET| / DOSE_TOLERANCE_ABS = 3/3 = 1
        assert np.allclose(result.gamma, expected)

    @staticmethod
    def test_global_normalization_matches_manual_computation(protocol):
        """dose_tolerance_abs used should equal dose_difference% * max(ref_grid.dose),
        computed once — verify by comparing against a hand-computed value."""
        ref_dose = np.array([80.0, 100.0, 90.0, 85.0])  # non-uniform: global != local per point
        ref_grid = DoseGrid(coordinates=(np.arange(4) * SPACING,), dose=ref_dose)
        eval_grid = make_uniform_grid(4, REF_DOSE + DOSE_OFFSET, SPACING)

        dose_difference_abs_theory = ref_grid.dose.max() * protocol.dose_difference / 100

        protocol_global = dataclasses.replace(protocol, dose_tolerance_abs=None, local=False)  # let global do its thing
        protocol_enforced = dataclasses.replace(protocol, dose_tolerance_abs=dose_difference_abs_theory)
        result_global = gamma_shell(ref_grid, eval_grid, protocol_global)
        result_enforced = gamma_shell(ref_grid, eval_grid, protocol_enforced)

        assert np.allclose(result_global.gamma, result_enforced.gamma)

    @staticmethod
    def test_local_normalization_varies_per_point(protocol):
        """With protocol.local=True, dose_tolerance_abs should be computed per point from that point's own ref dose,
        not a single global value."""
        ref_dose = np.array([80.0, 85.0])
        ref_grid = DoseGrid(coordinates=(np.arange(2) * SPACING,), dose=ref_dose)
        eval_dose_value = 100.0
        eval_grid = make_uniform_grid(2, eval_dose_value, SPACING)

        dose_tolerance_abs_theory = ref_dose * protocol.dose_difference / 100  # per-point array

        protocol_local = dataclasses.replace(protocol, dose_tolerance_abs=None, local=True)
        result_local = gamma_shell(ref_grid, eval_grid, protocol_local)

        expected_gamma = np.abs(eval_dose_value - ref_dose) / dose_tolerance_abs_theory

        assert np.allclose(result_local.gamma, expected_gamma)

    @staticmethod
    def test_low_dose_cutoff_produces_nan(protocol):
        """Ref points below lower_percent_dose_cutoff are never computed and
        appear as NaN in the result, not 0 or inf."""
        ref_dose = np.array([10.0, 100.0])  # 10 is 10% of max=100, well below a 20% cutoff
        ref_grid = DoseGrid(coordinates=(np.arange(2) * SPACING,), dose=ref_dose)
        eval_grid = make_uniform_grid(2, REF_DOSE, SPACING)

        protocol_cutoff = dataclasses.replace(protocol, dose_threshold=20)

        result = gamma_shell(ref_grid, eval_grid, protocol_cutoff)

        assert np.isnan(result.gamma[0])  # 10 < 20% of 100: should be excluded
        assert not np.isnan(result.gamma[1])  # 100 is the max itself: should be computed

    @staticmethod
    def test_out_of_bounds_ref_point_is_nan_not_inf(protocol):
        """A ref point outside eval_grid's extent (expanded by d_max) should be NaN"""
        ref_grid = DoseGrid(coordinates=(np.array([10000.0]),), dose=np.array([REF_DOSE]))
        eval_grid = make_uniform_grid(GRID_SIZE, REF_DOSE, SPACING)

        result = gamma_shell(ref_grid, eval_grid, protocol)

        assert np.isnan(result.gamma[0])

    @staticmethod
    def test_nan_count_matches_cutoff_plus_out_of_bounds(protocol):
        """Exact accounting: NaN count == (points below cutoff) + (points outside eval extent), with no overlap
        double-counted or missed points."""
        #   indices 0,1 -> below cutoff, in-bounds (low dose)
        #   indices 2,3 -> above cutoff, in-bounds (should compute fine)
        #   indices 4,5 -> above cutoff, out of bounds (far from eval grid)
        ref_coords = np.array([0.0, 2.0, 4.0, 6.0, 10_000.0, 10_002.0])
        ref_dose = np.array([1.0, 1.0, 100.0, 100.0, 100.0, 100.0])
        ref_grid = DoseGrid(coordinates=(ref_coords,), dose=ref_dose)

        eval_grid = make_uniform_grid(GRID_SIZE, REF_DOSE, SPACING)

        protocol_cutoff = dataclasses.replace(protocol, dose_threshold=20)

        result = gamma_shell(ref_grid, eval_grid, protocol_cutoff)

        n_below_cutoff = 2
        n_out_of_bounds = 2
        expected_nan_count = n_below_cutoff + n_out_of_bounds

        assert np.sum(np.isnan(result.gamma)) == expected_nan_count

        assert np.isnan(result.gamma[0]) and np.isnan(result.gamma[1])
        assert np.isnan(result.gamma[4]) and np.isnan(result.gamma[5])

        assert not np.isnan(result.gamma[2]) and not np.isnan(result.gamma[3])

    @staticmethod
    def test_shell_search_ge_bruteforce_inequality(protocol):
        """bruteforce gamma must be >= shell-interpolated gamma, pointwise, everywhere"""
        size, spacing = 15, 3.0  # more coarse spacing so grid bias shows up on bruteforce
        direction = unit_direction(1, diagonal=False)
        a, s = 1.0, 1.5
        ref_grid = make_ramp_grid(1, size, spacing, dose=100.0, a=a, s=0.0, direction=direction)
        eval_grid = make_ramp_grid(1, size, spacing, dose=100.0, a=a, s=s, direction=direction)

        result_shell = gamma_shell(ref_grid, eval_grid, protocol)
        result_bruteforce = gamma_bruteforce(ref_grid, eval_grid, protocol)

        interior = slice(2, -2)  # avoid edge effects
        assert np.all(result_bruteforce.gamma[interior] >= result_shell.gamma[interior] - 1e-9)

    # @staticmethod
    # def test_matches_pymedphys_golden_fixtures(protocol):
    #     """Compare gamma_shell against pre-computed pymedphys fixtures across
    #     real dose maps. Different discretization, so compare within tolerance,
    #     not exact — both per-point distribution and pass-rate."""
    #
    #     protocol_golden = dataclasses.replace(protocol, dose_tolerance_abs=None, dose_threshold=20)
    #     fixtures = load_fixtures(protocol_golden)
    #
    #     for prefix, fixture in fixtures.items():
    #         result = gamma_shell(fixture.ref_grid, fixture.eval_grid, protocol_golden)
    #
    #         valid = ~np.isnan(fixture.gamma) & ~np.isnan(result.gamma)
    #         diff = np.abs(result.gamma[valid] - fixture.gamma[valid])
    #
    #         assert np.median(diff) < 0.05, f"{prefix}: median diff too high"
    #         assert np.percentile(diff, 95) < 0.2, f"{prefix}: 95th pct diff too high"
    #
    #         pass_rate_shell = np.mean(result.gamma[valid] < 1)
    #         pass_rate_pymedphys = np.mean(fixture.gamma[valid] < 1)
    #         assert abs(pass_rate_shell - pass_rate_pymedphys) < 1, f"{prefix}: pass rate mismatch"
