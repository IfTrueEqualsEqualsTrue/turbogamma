import numpy as np

from turbogamma.classes import DoseGrid, Protocol, GammaSearchResult, GammaResult, resolve_dose_tolerance
from turbogamma.geometry import radii_schedule, shell_offsets
from turbogamma.interpolation import build_query


def _distance_term_sq(d: float, dta: float) -> float:
    return (d / dta) ** 2


def _shell_min_dose_diff(
        r_ref: np.ndarray,
        dose_ref: float,
        d: float,
        step: float,
        ndim: int,
        interp_query,
) -> float:
    """min |D_eval - D_ref| over the shell of radius d."""
    offsets = shell_offsets(d, step, ndim)
    interpolated = interp_query(r_ref + offsets)
    return np.min(np.abs(interpolated - dose_ref))


def _should_stop(gamma_sq_best: float, d: float, dta: float) -> bool:
    """Exact pruning bound: once gamma_sq_best <= (d/dta)^2, no larger
    radius can ever improve it."""
    return gamma_sq_best <= _distance_term_sq(d, dta)


def find_best_gamma(
        r_ref: np.ndarray,
        dose_ref: float,
        dose_tolerance_abs: float,
        eval_grid: DoseGrid,
        protocol: Protocol,
) -> GammaSearchResult:
    """Shell-search gamma for a single reference point.

    Marches an outward radial schedule, terminating as soon as no larger shell can improve the result. """
    ndim = len(eval_grid.coordinates)
    d_max = protocol.max_gamma * protocol.dta
    interp_query, _ = build_query(eval_grid)
    schedule, step = radii_schedule(protocol.dta, protocol.interp_fraction, d_max)

    gamma_sq_best = float("inf")
    shells_visited = 0
    last_d = None

    for d in schedule:
        if _should_stop(gamma_sq_best, d, protocol.dta):
            break

        min_dose_diff = _shell_min_dose_diff(r_ref, dose_ref, d, step, ndim, interp_query)
        gamma_sq = _distance_term_sq(d, protocol.dta) + (min_dose_diff / dose_tolerance_abs) ** 2
        gamma_sq_best = min(gamma_sq, gamma_sq_best)

        shells_visited += 1
        last_d = d

    return GammaSearchResult(np.sqrt(gamma_sq_best), shells_visited, last_d)


def _flatten_ref_positions(ref_grid: DoseGrid) -> np.ndarray:
    """Physical coordinates of every reference point, shape (N, ndim), in
    the grid's natural flattening order (matches ref_grid.dose.ravel())."""
    axes = np.meshgrid(*ref_grid.coordinates, indexing="ij")
    return np.stack([axis.ravel() for axis in axes], axis=-1)


def _above_dose_cutoff(ref_grid: DoseGrid, protocol: Protocol) -> np.ndarray:
    """Boolean mask (flat, matching ref_grid.dose.ravel()): True for points
    at or above lower_percent_dose_cutoff, expressed as a % of the ref
    grid's global max dose."""
    threshold = protocol.dose_threshold / 100 * ref_grid.dose.max()
    return ref_grid.dose.ravel() >= threshold


def _within_eval_bounds(pos_ref: np.ndarray, eval_grid: DoseGrid, protocol: Protocol) -> np.ndarray:
    """Boolean mask: True for reference points whose distance to the eval
    grid's bounding box is small enough that a valid gamma candidate could
    exist within d_max = max_gamma * dta."""
    d_max = protocol.max_gamma * protocol.dta
    in_bounds = np.ones(pos_ref.shape[0], dtype=bool)
    for axis_i, axis in enumerate(eval_grid.coordinates):
        lo, hi = axis.min() - d_max, axis.max() + d_max
        coord = pos_ref[:, axis_i]
        in_bounds &= (coord >= lo) & (coord <= hi)
    return in_bounds


def _search_kept_points(
        pos_ref_kept: np.ndarray,
        ref_dose_kept: np.ndarray,
        dose_tolerance_abs: float | np.ndarray,
        eval_grid: DoseGrid,
        protocol: Protocol,
) -> np.ndarray:
    """Run find_best_gamma for every kept point. dose_tolerance_abs may be
    a single scalar (global/override) or a per-point array (local) — same
    length as pos_ref_kept. Returns gamma per point, inf converted to NaN."""
    is_scalar_tolerance = np.isscalar(dose_tolerance_abs)
    gamma_kept = np.empty(pos_ref_kept.shape[0])

    for i in range(pos_ref_kept.shape[0]):
        tol = dose_tolerance_abs if is_scalar_tolerance else dose_tolerance_abs[i]
        result = find_best_gamma(pos_ref_kept[i], ref_dose_kept[i], tol, eval_grid, protocol)
        gamma_kept[i] = result.gamma if np.isfinite(result.gamma) else np.nan

    return gamma_kept


def gamma_shell(ref_grid: DoseGrid, eval_grid: DoseGrid, protocol: Protocol) -> GammaResult:
    """Shell-search gamma index over a full reference grid.

    Points below the low-dose cutoff or outside the evaluation grid's
    reachable bounds are excluded and reported as NaN, without running
    the shell search on them.
    """
    pos_ref = _flatten_ref_positions(ref_grid)
    keep_mask = _above_dose_cutoff(ref_grid, protocol) & _within_eval_bounds(pos_ref, eval_grid, protocol)

    pos_ref_kept = pos_ref[keep_mask]
    ref_dose_kept = ref_grid.dose.ravel()[keep_mask]
    dose_tolerance_abs = resolve_dose_tolerance(protocol, ref_dose_kept, ref_grid.dose.max())

    gamma_flat = np.full(ref_grid.dose.size, np.nan)
    gamma_flat[keep_mask] = _search_kept_points(pos_ref_kept, ref_dose_kept, dose_tolerance_abs, eval_grid, protocol)

    return GammaResult(ref_grid=ref_grid, eval_grid=eval_grid, gamma=gamma_flat.reshape(ref_grid.dose.shape))
