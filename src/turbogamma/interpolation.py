from dataclasses import dataclass
from typing import Protocol as TypingProtocol

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from turbogamma.classes import DoseGrid

CONSTANT_SPACING_TOLERANCE = 1e-4

Coords = tuple[NDArray[np.float64], ...]
Bounds = tuple[tuple[float, float], ...]


class DoseQuery(TypingProtocol):
    """Samples an evaluation dose field at arbitrary physical coordinates."""

    n_dims: int
    bounds: Bounds

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class ScipyDoseQuery:
    n_dims: int
    bounds: Bounds
    axes: Coords
    _interp: RegularGridInterpolator

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._interp(points)


def _validate_grids(eval_grid: DoseGrid):
    """Raise ValueError if axes and dose array are inconsistent."""
    if len(eval_grid.coordinates) != eval_grid.dose.ndim:
        raise ValueError("Number of axes doesn't match grid shape")
    for i, ax in enumerate(eval_grid.coordinates):
        if len(ax) != eval_grid.dose.shape[i]:
            raise ValueError(f"Axis {i} doesn't match the size of the dosemap")
        if len(ax) < 2:
            raise ValueError(f"Axis {i} needs at least 2 points")
        if np.any(ax[1:] <= ax[:-1]):
            raise ValueError(f"Coordinates on axis {i} aren't sctrictly increasing")


def _get_axes_spacings(eval_axes: Coords):
    """Uniform spacing per axis. Raises if any axis is non-uniform. Assumes axes are already validated as strictly
    increasing."""
    spacings = ()
    for i, ax in enumerate(eval_axes):
        diffs = np.diff(ax)
        normalized = np.abs(diffs - diffs[0]) / diffs[0]
        if np.any(normalized > CONSTANT_SPACING_TOLERANCE):
            raise ValueError(f"Axis {i} does not have constant spacing")
        spacings += (diffs[0],)
    return spacings


def _axis_bounds(axes: Coords) -> Bounds:
    return tuple((ax[0], ax[-1]) for ax in axes)


def build_query(eval_grid: DoseGrid) -> tuple[DoseQuery, tuple[float, ...]]:
    """ Validates the grid and returns a queryable dose field"""
    _validate_grids(eval_grid)
    spacings = _get_axes_spacings(eval_grid.coordinates)
    interpolator = RegularGridInterpolator(points=eval_grid.coordinates, values=eval_grid.dose, bounds_error=False,
                                           fill_value=np.inf)
    bounds = _axis_bounds(eval_grid.coordinates)
    scipy_query = ScipyDoseQuery(len(eval_grid.coordinates), bounds, eval_grid.coordinates, interpolator)
    return scipy_query, spacings
