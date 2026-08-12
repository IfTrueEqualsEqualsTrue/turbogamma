import numpy as np
import pytest

from turbogamma.gamma import gamma_1d, DTA, DOSE_TOLERANCE, DoseGrid

ATOL = 1e-4
ARRAY_SIZE = 16
TEST_CONSTANT_DOSE = 5.0


def ramp(i: int) -> float:
    return TEST_CONSTANT_DOSE * i


@pytest.fixture
def zeros_points() -> DoseGrid:
    return DoseGrid(coordinates=(np.zeros(ARRAY_SIZE),),
                    dose=np.zeros(ARRAY_SIZE))


@pytest.fixture
def constant_dose_offset_points() -> DoseGrid:
    return DoseGrid(coordinates=(np.zeros(ARRAY_SIZE),),
                    dose=np.full(ARRAY_SIZE, TEST_CONSTANT_DOSE))


@pytest.fixture
def dta_ramp_points():
    def _make(shift: int) -> DoseGrid:
        i = np.arange(ARRAY_SIZE)
        pos = DTA * (i + shift)
        dose = ramp(i)
        return DoseGrid((pos,), dose)

    return _make


class TestGamma1d:

    def test_zeros(self, zeros_points):
        np.testing.assert_allclose(gamma_1d(zeros_points, zeros_points).gamma,
                                   np.zeros(ARRAY_SIZE), atol=ATOL)

    def test_constant_dose_offset(self, zeros_points, constant_dose_offset_points):
        np.testing.assert_allclose(gamma_1d(zeros_points, constant_dose_offset_points).gamma,
                                   np.full(ARRAY_SIZE, TEST_CONSTANT_DOSE / DOSE_TOLERANCE), atol=ATOL)

    @pytest.mark.parametrize("shift", [1, 3])
    def test_dta_ramp(self, dta_ramp_points, shift: int):
        ref = dta_ramp_points(0)
        evaluation = dta_ramp_points(shift)
        np.testing.assert_allclose(gamma_1d(ref, evaluation).gamma,
                                   np.full(ARRAY_SIZE, float(shift)), atol=ATOL)

    def test_tradeoff(self):
        ref_pos = (np.array([0.0]),)
        ref_dose = np.array([0.0])
        ref_grid = DoseGrid(ref_pos, ref_dose)
        eval_pos = (np.array([DTA, DTA*5, DTA]),)
        eval_dose = np.array([DOSE_TOLERANCE * 5, DOSE_TOLERANCE, DOSE_TOLERANCE])
        eval_grid = DoseGrid(eval_pos, eval_dose)
        best_gamma: float = 1.4
        np.testing.assert_allclose(gamma_1d(ref_grid, eval_grid).gamma, [best_gamma], atol=1)
