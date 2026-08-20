from dataclasses import replace
from typing import Callable

from turbogamma.classes import DoseGrid, GammaResult, Protocol, protocol_regular, ABSOLUTE_DOSE_TOLERANCE
from turbogamma.gamma_bruteforce import gamma_bruteforce_1d, gamma_bruteforce_2d, gamma_bruteforce_3d
from turbogamma.scenarios import Scenarios1d, Scenarios2d, Scenarios3d


class GammaProvider:
    builder = None
    gamma_func: Callable[[DoseGrid, DoseGrid, Protocol], GammaResult]
    protocol: Protocol = protocol_regular

    @classmethod
    def _gamma(cls, ref_grid, eval_grid) -> GammaResult:
        return cls.gamma_func(ref_grid, eval_grid, cls.protocol)

    @classmethod
    def get_uniform_offset(cls, size: int, dose_ref: float, dose_eval: float) -> GammaResult:
        build = cls.builder.build_constant_dose(size)
        return cls._gamma(build(dose_ref), build(dose_eval))

    @classmethod
    def get_square_feature(cls, size: int, dose: float, shift: int) -> GammaResult:
        build = cls.builder.build_square_feature(size, dose)
        return cls._gamma(build(0), build(shift))

    @classmethod
    def get_ramp(cls, size: int, slope: float, shift: int, dose_offset: float = 1.0) -> GammaResult:
        build = cls.builder.build_ramp(size, slope, dose_offset)
        return cls._gamma(build(0), build(shift))

    @classmethod
    def set_context(cls, protocol: Protocol = protocol_regular, abs_dose_tolerance: float = ABSOLUTE_DOSE_TOLERANCE):
        cls.protocol = replace(protocol, dose_tolerance_abs=abs_dose_tolerance)


class GammaProvider1d(GammaProvider):
    builder = Scenarios1d()
    gamma_func = staticmethod(gamma_bruteforce_1d)


class GammaProvider2d(GammaProvider):
    builder = Scenarios2d()
    gamma_func = staticmethod(gamma_bruteforce_2d)


class GammaProvider3d(GammaProvider):
    builder = Scenarios3d()
    gamma_func = staticmethod(gamma_bruteforce_3d)
