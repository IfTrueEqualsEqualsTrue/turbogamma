import numpy as np

from turbogamma.classes import DoseGrid, Protocol
from turbogamma.geometry import radii_schedule, shell_offsets
from turbogamma.interpolation import build_query

max_gamma = 3.0


def find_best_gamma(ref_grid: DoseGrid, eval_grid: DoseGrid, protocol: Protocol) -> float:
    gamma_sq_best: np.float64 = float("inf")
    ndim = len(eval_grid.coordinates)
    d_max = max_gamma * protocol.dta
    interp_query, spacings = build_query(eval_grid)
    schedule = radii_schedule(protocol.dta, protocol.interp_fraction, d_max)
    for d in schedule:
        if gamma_sq_best < (d / protocol.dta) ** 2:
            break
        else:
            cached_offsets = shell_offsets(d, protocol.dta / protocol.interp_fraction, ndim)
            interpolated_eval_points = interp_query(cached_offsets)
            dose_value =

    return gamma_sq_best


ref_grid = DoseGrid((np.array([0.0]), (np.array([0.0])), np.array([2.0])))
eval_grid = DoseGrid((np.array([0.5, 2, 10]),), np.array([10, 2, 0.5]))
