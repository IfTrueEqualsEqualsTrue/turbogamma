from collections.abc import Callable

import numpy as np
import pytest

from turbogamma.gamma import gamma_1d, gamma_2d, gamma_3d, DoseGrid
from turbogamma.scenarios import Scenarios1d, Scenarios2d, Scenarios3d

ATOL = 1e-4
ARRAY_SIZE = 16
BASE_CONSTANT_DOSE = 2.0
CONSTANT_DOSE_OFFSET = 1.0
DOSE_TOLERANCE_ABS = 2.0
GAMMA_TRADEOFF = 1.41


@pytest.fixture
def constant_dose_line() -> Callable:
    return Scenarios1d.build_constant_dose(ARRAY_SIZE)


@pytest.fixture
def constant_dose_grid() -> Callable:
    return Scenarios2d.build_constant_dose(ARRAY_SIZE)


@pytest.fixture
def constant_dose_volume() -> Callable:
    return Scenarios3d.build_constant_dose(ARRAY_SIZE)


class TestGamma1d:

    def test_uniform_dose_offset(self, constant_dose_line):
        base_dose = constant_dose_line(BASE_CONSTANT_DOSE)
        offset_dose = constant_dose_line(BASE_CONSTANT_DOSE + CONSTANT_DOSE_OFFSET)
        np.testing.assert_allclose(gamma_1d(base_dose, offset_dose, DOSE_TOLERANCE_ABS).gamma,
                                   np.full(ARRAY_SIZE, CONSTANT_DOSE_OFFSET / DOSE_TOLERANCE_ABS), atol=ATOL)

    def test_symmetry(self, constant_dose_line):
        base_dose = constant_dose_line(BASE_CONSTANT_DOSE)
        offset_dose = constant_dose_line(BASE_CONSTANT_DOSE + CONSTANT_DOSE_OFFSET)
        gamma_fw = gamma_1d(base_dose, offset_dose, DOSE_TOLERANCE_ABS)
        gamma_bw = gamma_1d(offset_dose, base_dose, DOSE_TOLERANCE_ABS)
        np.testing.assert_allclose(gamma_bw.gamma, gamma_fw.gamma, atol=ATOL)

    def test_tradeoff(self):
        """ Reference point, good position bad dose, good position and dose, bad position good dose,"""
        ref_grid = DoseGrid((np.array([0.0]),), np.array([0.0]))
        eval_grid = DoseGrid((np.array([0.5, 2, 10]),), np.array([10, 2, 0.5]))
        gamma_result = gamma_1d(ref_grid, eval_grid, dose_tolerance_abs=2.0)
        np.testing.assert_allclose(gamma_result.gamma, np.array([GAMMA_TRADEOFF]), atol=1e-1)


class TestGamma2d:

    def test_uniform_dose_offset(self, constant_dose_grid):
        base_dose = constant_dose_grid(BASE_CONSTANT_DOSE)
        offset_dose = constant_dose_grid(BASE_CONSTANT_DOSE + CONSTANT_DOSE_OFFSET)
        np.testing.assert_allclose(gamma_2d(base_dose, offset_dose, DOSE_TOLERANCE_ABS).gamma,
                                   np.full((ARRAY_SIZE, ARRAY_SIZE), CONSTANT_DOSE_OFFSET / DOSE_TOLERANCE_ABS),
                                   atol=ATOL)

    def test_symmetry(self, constant_dose_grid):
        base_dose = constant_dose_grid(BASE_CONSTANT_DOSE)
        offset_dose = constant_dose_grid(BASE_CONSTANT_DOSE + CONSTANT_DOSE_OFFSET)
        gamma_fw = gamma_2d(base_dose, offset_dose, DOSE_TOLERANCE_ABS)
        gamma_bw = gamma_2d(offset_dose, base_dose, DOSE_TOLERANCE_ABS)
        np.testing.assert_allclose(gamma_bw.gamma, gamma_fw.gamma, atol=ATOL)


class TestGamma3d:

    def test_uniform_dose_offset(self, constant_dose_volume):
        base_dose = constant_dose_volume(BASE_CONSTANT_DOSE)
        offset_dose = constant_dose_volume(BASE_CONSTANT_DOSE + CONSTANT_DOSE_OFFSET)
        np.testing.assert_allclose(gamma_3d(base_dose, offset_dose, DOSE_TOLERANCE_ABS).gamma,
                                   np.full((ARRAY_SIZE, ARRAY_SIZE, ARRAY_SIZE),
                                           CONSTANT_DOSE_OFFSET / DOSE_TOLERANCE_ABS), atol=ATOL)

    def test_symmetry(self, constant_dose_volume):
        base_dose = constant_dose_volume(BASE_CONSTANT_DOSE)
        offset_dose = constant_dose_volume(BASE_CONSTANT_DOSE + CONSTANT_DOSE_OFFSET)
        gamma_fw = gamma_3d(base_dose, offset_dose, DOSE_TOLERANCE_ABS)
        gamma_bw = gamma_3d(offset_dose, base_dose, DOSE_TOLERANCE_ABS)
        np.testing.assert_allclose(gamma_bw.gamma, gamma_fw.gamma, atol=ATOL)
