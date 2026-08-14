import pytest
from pytest_benchmark.plugin import benchmark

from turbogamma.gamma_bruteforce import gamma_bruteforce_1d, gamma_bruteforce_2d, gamma_bruteforce_3d
from turbogamma.scenarios import Scenarios1d, Scenarios3d, Scenarios2d

DOSE_REF = 1.0
DOSE_EVAL = 2.0


@pytest.fixture(params=[8, 32])
def uniform_lines(request):
    size = request.param
    builder = Scenarios1d.build_constant_dose(size)
    ref_grid = builder(DOSE_REF)
    eval_grid = builder(DOSE_EVAL)
    return ref_grid, eval_grid


@pytest.fixture(params=[8, 32])
def uniform_grids(request):
    size = request.param
    builder = Scenarios2d.build_constant_dose(size)
    ref_grid = builder(DOSE_REF)
    eval_grid = builder(DOSE_EVAL)
    return ref_grid, eval_grid


@pytest.fixture(params=[8, 16])
def uniform_volumes(request):
    size = request.param
    builder = Scenarios3d.build_constant_dose(size)
    ref_grid = builder(DOSE_REF)
    eval_grid = builder(DOSE_EVAL)
    return ref_grid, eval_grid


class TestGamma1dBench:

    def test_bench_gamma_1d_uniform(self, benchmark, uniform_lines):
        ref_grid, eval_grid = uniform_lines
        benchmark(gamma_bruteforce_1d, ref_grid, eval_grid)


class TestGamma2dBench:

    def test_bench_gamma_2d_uniform(self, benchmark, uniform_grids):
        ref_grid, eval_grid = uniform_grids
        benchmark(gamma_bruteforce_2d, ref_grid, eval_grid)


class TestGamma3dBench:

    def test_bench_gamma_3d_uniform(self, benchmark, uniform_volumes):
        ref_grid, eval_grid = uniform_volumes
        benchmark(gamma_bruteforce_3d, ref_grid, eval_grid)
