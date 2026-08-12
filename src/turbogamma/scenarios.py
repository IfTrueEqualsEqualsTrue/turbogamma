from typing import Callable

import numpy as np

from turbogamma.gamma import DTA, DoseGrid


def dose_ramp(i: int, dose: float) -> float:
    return dose * i


def build_zeros(size: int) -> DoseGrid:
    return DoseGrid(coordinates=(np.zeros(size),), dose=np.zeros(size))


def build_constant_dose(size: int, dose: float) -> DoseGrid:
    return DoseGrid(coordinates=(np.zeros(size),),
                    dose=np.full(size, dose))


def build_dose_ramp(size: int, constant_dose: float) -> Callable:
    def _make(shift: int) -> DoseGrid:
        i = np.arange(size)
        pos = DTA * (i + shift)
        dose = dose_ramp(i, constant_dose)
        return DoseGrid((pos,), dose)

    return _make
