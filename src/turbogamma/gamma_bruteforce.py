import numpy as np

from turbogamma.classes import Protocol, DoseGrid, GammaResult, protocol_regular


def _dose_tolerance_abs(protocol: Protocol, ref_dose_kept: np.ndarray, max_dose_ref_grid: float):
    if protocol.dose_tolerance_abs is not None:
        return protocol.dose_tolerance_abs
    if protocol.local:
        return protocol.dose_difference / 100 * ref_dose_kept.reshape((-1, 1))
    return protocol.dose_difference / 100 * max_dose_ref_grid  # single scalar


def gamma_bruteforce(ref_grid: DoseGrid, eval_grid: DoseGrid,
                     protocol: Protocol = protocol_regular) -> GammaResult:
    """Brute-force gamma index for grids of any dimensionality."""
    max_dose_ref_grid = ref_grid.dose.max()

    cutoff_threshold = protocol.dose_threshold * max_dose_ref_grid / 100
    mask = ref_grid.dose >= cutoff_threshold  # points to KEEP

    # Position: one row per kept reference point, one column per evaluation point
    ref_axes = np.meshgrid(*ref_grid.coordinates, indexing='ij')
    pos_ref_kept = np.stack([axis[mask] for axis in ref_axes], axis=-1)

    eval_axes = np.meshgrid(*eval_grid.coordinates, indexing='ij')
    pos_eval_flatten = np.stack([axis.flatten() for axis in eval_axes], axis=-1)

    sq_dist = ((pos_ref_kept[:, None, :] - pos_eval_flatten) ** 2).sum(axis=-1)

    # Dose
    ref_dose_kept = ref_grid.dose[mask]
    dose_diff = ref_dose_kept[:, None] - eval_grid.dose.flatten()

    dose_tolerance_abs = _dose_tolerance_abs(protocol, ref_dose_kept, max_dose_ref_grid)

    gammas = np.sqrt((dose_diff / dose_tolerance_abs) ** 2 + (np.sqrt(sq_dist) / protocol.dta) ** 2)
    gamma = gammas.min(axis=1)
    gamma_full = np.full(ref_grid.dose.shape, np.nan)
    gamma_full[mask] = gamma

    return GammaResult(ref_grid=ref_grid, eval_grid=eval_grid, gamma=gamma_full)


gamma_bruteforce_1d = gamma_bruteforce
gamma_bruteforce_2d = gamma_bruteforce
gamma_bruteforce_3d = gamma_bruteforce
