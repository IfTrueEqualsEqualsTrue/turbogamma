import numpy as np

from turbogamma.classes import Protocol, DoseGrid, GammaResult, DOSE_TOLERANCE_PERCENT, DTA, protocol_regular


def gamma_bruteforce_1d(ref_grid: DoseGrid, eval_grid: DoseGrid, protocol: Protocol = protocol_regular,
                        dose_tolerance_abs: float = None) -> GammaResult:
    max_dose_ref_grid = ref_grid.dose.max()

    cutoff_threshold = protocol.dose_threshold * max_dose_ref_grid / 100
    mask = ref_grid.dose >= cutoff_threshold  # points to KEEP

    ref_pos_row = ref_grid.coordinates[0]
    ref_pos_kept = ref_pos_row[mask]
    ref_dose_kept = ref_grid.dose[mask]
    ref_pos_col = ref_pos_kept.reshape((len(ref_pos_kept), 1))
    eval_pos_row = eval_grid.coordinates[0]
    ref_dose_col = ref_dose_kept.reshape((len(ref_pos_kept), 1))
    dose_diff = ref_dose_col - eval_grid.dose
    pos_diff = ref_pos_col - eval_pos_row

    percent = protocol.dose_difference / 100
    if dose_tolerance_abs is None:
        if protocol.local:
            dose_tolerance_abs = percent * ref_dose_kept.reshape((-1, 1))
        else:
            dose_tolerance_abs = percent * max_dose_ref_grid  # single scalar

    gammas: np.ndarray = np.sqrt((dose_diff / dose_tolerance_abs) ** 2 + (pos_diff / protocol.dta) ** 2)
    gamma = gammas.min(axis=1)
    gamma_full = np.full(ref_grid.dose.shape, np.nan)
    gamma_full[mask] = gamma
    return GammaResult(ref_grid=ref_grid, eval_grid=eval_grid, gamma=gamma_full)


def gamma_bruteforce_2d(ref_grid: DoseGrid, eval_grid: DoseGrid, protocol: Protocol = protocol_regular,
                        dose_tolerance_abs: float = None) -> GammaResult:
    max_dose_ref_grid = ref_grid.dose.max()

    cutoff_threshold = protocol.dose_threshold * max_dose_ref_grid / 100
    mask = ref_grid.dose >= cutoff_threshold  # points to KEEP

    # Position first
    xx_pos_ref, yy_pos_ref = np.meshgrid(ref_grid.coordinates[0], ref_grid.coordinates[1], indexing='ij')
    pos_ref_kept = np.stack([xx_pos_ref[mask], yy_pos_ref[mask]], axis=-1)

    xx_pos_eval, yy_pos_eval = np.meshgrid(eval_grid.coordinates[0], eval_grid.coordinates[1], indexing='ij')
    pos_eval_flatten = np.stack([xx_pos_eval.flatten(), yy_pos_eval.flatten()], axis=-1)

    dist_diff = pos_ref_kept[:, None, :] - pos_eval_flatten
    sq_dist = (dist_diff ** 2).sum(axis=-1)

    # Dose then
    ref_dose_kept = ref_grid.dose[mask]
    dose_eval_flatten = eval_grid.dose.flatten()
    dose_diff = ref_dose_kept[:, None] - dose_eval_flatten

    percent = protocol.dose_difference / 100
    if dose_tolerance_abs is None:
        if protocol.local:
            dose_tolerance_abs = percent * ref_dose_kept.reshape((-1, 1))
        else:
            dose_tolerance_abs = percent * max_dose_ref_grid  # single scalar

    gammas = np.sqrt((dose_diff / dose_tolerance_abs) ** 2 + (np.sqrt(sq_dist) / protocol.dta) ** 2)
    gamma = gammas.min(axis=1)
    gamma_full = np.full(ref_grid.dose.shape, np.nan)
    gamma_full[mask] = gamma

    return GammaResult(ref_grid=ref_grid, eval_grid=eval_grid, gamma=gamma_full)


def gamma_bruteforce_3d(ref_grid: DoseGrid, eval_grid: DoseGrid, dose_tolerance_abs: float = None) -> GammaResult:
    if dose_tolerance_abs is None:
        dose_tolerance_abs = DOSE_TOLERANCE_PERCENT * ref_grid.dose.max()

    # Position first
    xx_pos_ref, yy_pos_ref, zz_pos_ref = np.meshgrid(*ref_grid.coordinates, indexing='ij')
    pos_ref_flatten = np.stack([xx_pos_ref.flatten(), yy_pos_ref.flatten(), zz_pos_ref.flatten()], axis=-1)

    xx_pos_eval, yy_pos_eval, zz_pos_eval = np.meshgrid(*eval_grid.coordinates, indexing='ij')
    pos_eval_flatten = np.stack([xx_pos_eval.flatten(), yy_pos_eval.flatten(), zz_pos_eval.flatten()], axis=-1)

    dist_diff = pos_ref_flatten[:, None, :] - pos_eval_flatten
    sq_dist = (dist_diff ** 2).sum(axis=-1)

    # Dose then
    dose_ref_flatten = ref_grid.dose.flatten()
    dose_eval_flatten = eval_grid.dose.flatten()
    dose_diff = dose_ref_flatten[:, None] - dose_eval_flatten

    gammas = np.sqrt((dose_diff / dose_tolerance_abs) ** 2 + (np.sqrt(sq_dist) / DTA) ** 2)
    gamma = gammas.min(axis=1).reshape(ref_grid.dose.shape)

    return GammaResult(ref_grid=ref_grid, eval_grid=eval_grid, gamma=gamma)
