from turbogamma.gamma import gamma_2d, GammaResult, gamma_1d
from turbogamma.scenarios import Scenarios2d, Scenarios1d


class GammaProvider1d:

    @staticmethod
    def get_uniform_offset(size: int, dose_ref: float, dose_eval: float) -> GammaResult:
        uniform_dose_builder = Scenarios1d.build_constant_dose(size)
        ref_grid = uniform_dose_builder(dose_ref)
        offset_grid = uniform_dose_builder(dose_eval)
        gamma = gamma_1d(ref_grid, offset_grid)
        return gamma

    @staticmethod
    def get_square_feature(size: int, dose: float, shift: int) -> GammaResult:
        square_feature_builder = Scenarios1d.build_square_feature(size, dose)
        ref_grid = square_feature_builder(0)
        eval_grid = square_feature_builder(shift)
        return gamma_1d(ref_grid, eval_grid)


class GammaProvider2d:

    @staticmethod
    def get_uniform_offset(size, dose_ref, dose_eval) -> GammaResult:
        uniform_dose_builder = Scenarios2d.build_constant_dose(size)
        ref_grid = uniform_dose_builder(dose_ref)
        offset_grid = uniform_dose_builder(dose_eval)
        gamma = gamma_2d(ref_grid, offset_grid)
        return gamma

    @staticmethod
    def get_square_feature(size: int, dose: float, shift: int) -> GammaResult:
        square_feature_builder = Scenarios2d.build_square_feature(size, dose)
        ref_grid = square_feature_builder(0)
        eval_grid = square_feature_builder(shift)
        return gamma_2d(ref_grid, eval_grid)
