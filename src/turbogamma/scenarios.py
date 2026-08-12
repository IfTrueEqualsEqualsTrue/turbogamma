from typing import Callable

import numpy as np

from turbogamma.gamma import DTA, DoseGrid


class Scenarios1d:

    @staticmethod
    def build_zeros(size: int) -> DoseGrid:
        return DoseGrid(coordinates=(np.zeros(size),), dose=np.zeros(size))

    @staticmethod
    def build_constant_dose(size: int, dose: float) -> DoseGrid:
        return DoseGrid(coordinates=(np.zeros(size),),
                        dose=np.full(size, dose))

    @staticmethod
    def build_uniform_feature(size: int, constant_dose: float) -> Callable:
        def _make(shift: int) -> DoseGrid:
            i = np.arange(size)
            pos = DTA * i

            feature_width = size // 2
            start = (size - feature_width) // 2
            mask = (i >= start + shift) & (i < start + shift + feature_width)
            dose = np.where(mask, constant_dose, 0.0)

            return DoseGrid((pos,), dose)

        return _make


class Scenarios2d:

    @staticmethod
    def build_zeros(size: int) -> DoseGrid:
        return DoseGrid(coordinates=(np.zeros(size), np.zeros(size)), dose=np.zeros((size, size)))

    @staticmethod
    def build_constant_dose(size: int, dose: float) -> DoseGrid:
        return DoseGrid(coordinates=(np.zeros(size), np.zeros(size)),
                        dose=np.full((size, size), dose))

    @staticmethod
    def build_uniform_feature(size: int, constant_dose: float) -> Callable:
        def _make(shift: int) -> DoseGrid:
            i = np.arange(size)
            ix, iy = np.meshgrid(i, i, indexing="ij")
            pos = DTA * i

            feature_width = size // 2
            start = (size - feature_width) // 2
            mask = (ix >= start + shift) & (iy >= start + shift) & (ix < start + shift + feature_width) & (
                        iy < start + shift + feature_width)
            dose = np.where(mask, constant_dose, 0.0)
            return DoseGrid((pos, pos,), dose)

        return _make
