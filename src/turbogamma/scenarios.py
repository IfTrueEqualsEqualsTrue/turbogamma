from typing import Callable

import numpy as np

from turbogamma.gamma import DTA, DoseGrid


class Scenarios1d:

    @staticmethod
    def build_constant_dose(size: int, ) -> Callable:
        def _make(dose: float):
            i = np.arange(size)
            pos = DTA * i
            return DoseGrid(coordinates=(pos,), dose=np.full(size, dose))

        return _make

    @staticmethod
    def build_square_feature(size: int, constant_dose: float) -> Callable:
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
    def build_constant_dose(size: int, ) -> Callable:
        def _make(dose: float):
            i = np.arange(size)
            pos = DTA * i
            return DoseGrid(coordinates=(pos, pos,), dose=np.full((size, size), dose))

        return _make

    @staticmethod
    def build_square_feature(size: int, constant_dose: float) -> Callable:
        def _make(shift: int) -> DoseGrid:
            i = np.arange(size)
            ix, iy = np.meshgrid(i, i, indexing="ij")
            pos = DTA * i

            feature_width = size // 2
            start = (size - feature_width) // 2
            mask = (ix >= start + shift) & (iy >= start + shift) & (ix < start + shift + feature_width) & (
                    iy < start + shift + feature_width)
            dose = np.where(mask, constant_dose, 1.0)
            return DoseGrid((pos, pos,), dose)

        return _make


class Scenarios3d:

    @staticmethod
    def build_constant_dose(size: int, ) -> Callable:
        def _make(dose: float):
            i = np.arange(size)
            pos = DTA * i
            return DoseGrid(coordinates=(pos, pos, pos,), dose=np.full((size, size, size), dose))

        return _make

    @staticmethod
    def build_square_feature(size: int, constant_dose: float) -> Callable:
        def _make(shift: int) -> DoseGrid:
            i = np.arange(size)
            ix, iy, iz = np.meshgrid(i, i, i, indexing="ij")
            pos = DTA * i

            feature_width = size // 2
            start = (size - feature_width) // 2
            mask = ((ix >= start + shift) &
                    (iy >= start + shift) &
                    (iz >= start + shift) &
                    (ix < start + shift + feature_width) &
                    (iy < start + shift + feature_width) &
                    (iz < start + shift + feature_width))
            dose = np.where(mask, constant_dose, 1.0)
            return DoseGrid((pos, pos, pos), dose)

        return _make
