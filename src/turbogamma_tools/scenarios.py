from typing import Callable

import numpy as np

from turbogamma.classes import DoseGrid, SPACING


class Scenarios1d:

    def build_constant_dose(self, size: int, ) -> Callable:
        def _make(dose: float):
            i, pos = self.build_1d_axes(size)
            return DoseGrid(coordinates=(pos,), dose=np.full(size, dose))

        return _make

    def build_square_feature(self, size: int, constant_dose: float) -> Callable:
        def _make(shift: int) -> DoseGrid:
            i, pos = self.build_1d_axes(size)

            feature_width = size // 2
            start = (size - feature_width) // 2
            mask = (i >= start + shift) & (i < start + shift + feature_width)
            dose = np.where(mask, constant_dose, 0.0)

            return DoseGrid((pos,), dose)

        return _make

    def build_ramp(self, size: int, a: float, dose_offset=1.0):
        """ Build a 1d dose ramp of length size, shifted by s with a gradient of a"""

        def _make(s: int):
            i, pos = self.build_1d_axes(size)

            dose = (i - s) * a + np.full(size, dose_offset)
            return DoseGrid((pos,), dose)

        return _make

    @staticmethod
    def build_1d_axes(size: int) -> tuple[np.ndarray]:
        i = np.arange(size)
        pos = SPACING * i
        return i, pos


class Scenarios2d:

    @staticmethod
    def build_constant_dose(size: int, ) -> Callable:
        def _make(dose: float):
            i = np.arange(size)
            pos = SPACING * i
            return DoseGrid(coordinates=(pos, pos,), dose=np.full((size, size), dose))

        return _make

    def build_square_feature(self, size: int, constant_dose: float) -> Callable:
        def _make(shift: int) -> DoseGrid:
            i, ix, iy, pos = self.build_2d_axes(size)

            feature_width = size // 2
            start = (size - feature_width) // 2
            mask = (ix >= start + shift) & (iy >= start + shift) & (ix < start + shift + feature_width) & (
                    iy < start + shift + feature_width)
            dose = np.where(mask, constant_dose, 0.0)
            return DoseGrid((pos, pos,), dose)

        return _make

    def build_ramp(self, size: int, a: float, dose_offset=1.0):
        """ Build a 2d diagonnal dose ramp of length size, shifted by s with a gradient of a"""

        def _make(s: int):
            i, ix, iy, pos = self.build_2d_axes(size)

            dose = (ix + iy - s) * a + np.full((size, size), dose_offset)
            return DoseGrid((pos, pos), dose)

        return _make

    @staticmethod
    def build_2d_axes(size: int) -> tuple[np.ndarray]:
        i = np.arange(size)
        ix, iy = np.meshgrid(i, i, indexing="ij")
        pos = SPACING * i
        return i, ix, iy, pos


class Scenarios3d:

    @staticmethod
    def build_constant_dose(size: int, ) -> Callable:
        def _make(dose: float):
            i = np.arange(size)
            pos = SPACING * i
            return DoseGrid(coordinates=(pos, pos, pos,), dose=np.full((size, size, size), dose))

        return _make

    @staticmethod
    def build_square_feature(size: int, constant_dose: float) -> Callable:
        def _make(shift: int) -> DoseGrid:
            i = np.arange(size)
            ix, iy, iz = np.meshgrid(i, i, i, indexing="ij")
            pos = SPACING * i

            feature_width = size // 2
            start = (size - feature_width) // 2
            mask = ((ix >= start + shift) &
                    (iy >= start + shift) &
                    (iz >= start + shift) &
                    (ix < start + shift + feature_width) &
                    (iy < start + shift + feature_width) &
                    (iz < start + shift + feature_width))
            dose = np.where(mask, constant_dose, 0.0)
            return DoseGrid((pos, pos, pos), dose)

        return _make
