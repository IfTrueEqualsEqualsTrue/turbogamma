from collections.abc import Callable

import numpy as np
import pytest

from turbogamma.gamma import gamma_1d, gamma_2d, gamma_3d
from turbogamma.scenarios import Scenarios1d, Scenarios2d, Scenarios3d

ATOL = 1e-4
ARRAY_SIZE = 16
BASE_CONSTANT_DOSE = 2.0
CONSTANT_DOSE_OFFSET = 1.0
DOSE_TOLERANCE_ABS = 2.0


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
