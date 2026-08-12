from typing import Callable

import numpy as np
import pytest

from turbogamma.gamma import gamma_1d, DTA, DOSE_TOLERANCE, DoseGrid
from turbogamma.scenarios import build_zeros, build_constant_dose, \
    build_dose_ramp

ATOL = 1e-4
ARRAY_SIZE = 16
TEST_CONSTANT_DOSE = 5.0


@pytest.fixture
def zeros_grid() -> DoseGrid:
    return build_zeros(ARRAY_SIZE)


@pytest.fixture
def constant_dose_grid() -> DoseGrid:
    return build_constant_dose(ARRAY_SIZE, TEST_CONSTANT_DOSE)


@pytest.fixture
def dta_ramp_grid() -> Callable:
    return build_dose_ramp(ARRAY_SIZE, TEST_CONSTANT_DOSE)


class TestGamma1d:

    def test_zeros(self, zeros_grid):
        np.testing.assert_allclose(gamma_1d(zeros_grid, zeros_grid).gamma,
                                   np.zeros(ARRAY_SIZE), atol=ATOL)

    def test_constant_dose_offset(self, zeros_grid, constant_dose_grid):
        np.testing.assert_allclose(gamma_1d(zeros_grid, constant_dose_grid).gamma,
                                   np.full(ARRAY_SIZE, TEST_CONSTANT_DOSE / DOSE_TOLERANCE), atol=ATOL)

    @pytest.mark.parametrize("shift", [1, 3])
    def test_dta_ramp(self, dta_ramp_grid, shift: int):
        ref = dta_ramp_grid(0)
        evaluation = dta_ramp_grid(shift)
        np.testing.assert_allclose(gamma_1d(ref, evaluation).gamma,
                                   np.full(ARRAY_SIZE, float(shift)), atol=ATOL)

    def test_tradeoff(self):
        ref_pos = (np.array([0.0]),)
        ref_dose = np.array([0.0])
        ref_grid = DoseGrid(ref_pos, ref_dose)
        eval_pos = (np.array([DTA, DTA * 5, DTA]),)
        eval_dose = np.array([DOSE_TOLERANCE * 5, DOSE_TOLERANCE, DOSE_TOLERANCE])
        eval_grid = DoseGrid(eval_pos, eval_dose)
        best_gamma: float = 1.4
        np.testing.assert_allclose(gamma_1d(ref_grid, eval_grid).gamma, [best_gamma], atol=1)
