import numpy as np

from turbogamma.classes import DoseGrid, Protocol, GammaSearchResult, GammaResult
from turbogamma.geometry import radii_schedule, shell_offsets
from turbogamma.interpolation import build_query


def find_best_gamma(r_ref: np.ndarray, dose_ref: float, dose_tolerance_abs: float, eval_grid: DoseGrid,
                    protocol: Protocol) -> GammaSearchResult:
    gamma_sq_best: np.float64 = float("inf")
    ndim = len(eval_grid.coordinates)
    d_max = protocol.max_gamma * protocol.dta
    interp_query, _ = build_query(eval_grid)
    schedule, step = radii_schedule(protocol.dta, protocol.interp_fraction, d_max)
    shells_visited = 0
    last_d = None
    for d in schedule:
        if gamma_sq_best <= (d / protocol.dta) ** 2:
            break

        cached_offsets = shell_offsets(d, step, ndim)
        interpolated_eval_points = interp_query(r_ref + cached_offsets)
        dose_terms = np.abs(interpolated_eval_points - dose_ref)
        min_dose_term = np.min(dose_terms, axis=-1)
        gamma_sq = (d / protocol.dta) ** 2 + (min_dose_term / dose_tolerance_abs) ** 2
        gamma_sq_best = min(gamma_sq, gamma_sq_best)
        shells_visited += 1
        last_d = d

    return GammaSearchResult(np.sqrt(gamma_sq_best), shells_visited, last_d)


def gamma_shell(ref_grid: DoseGrid, eval_grid: DoseGrid, protocol: Protocol) -> GammaResult:
    raise NotImplementedError
