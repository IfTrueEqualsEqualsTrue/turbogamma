from dataclasses import dataclass

import numpy as np

DOSE_TOLERANCE_PERCENT = 2
DTA = 2


@dataclass
class Protocol:
    dose_difference: float
    dta: float
    dose_threshold: float
    local: bool = False
    interp_fraction: float = 10
    max_gamma: float = 3

    def folder_name(self) -> str:
        return (
            f"dd{self.dose_difference:g}"
            f"_dta{self.dta:g}"
            f"_dt{self.dose_threshold:g}"
            f"_local{int(self.local)}"
            f"_if{self.interp_fraction:g}"
            f"_g{self.max_gamma:g}"
        )


@dataclass
class DoseGrid:
    coordinates: tuple[np.ndarray]
    dose: np.ndarray


@dataclass
class GammaResult:
    ref_grid: DoseGrid = None
    eval_grid: DoseGrid = None
    gamma: np.ndarray = None
    protocol: Protocol = None
    id: str = ""


def gamma_1d(ref_grid: DoseGrid, eval_grid: DoseGrid, protocol: Protocol,
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


def gamma_2d(ref_grid: DoseGrid, eval_grid: DoseGrid, dose_tolerance_abs: float = None) -> GammaResult:
    if dose_tolerance_abs is None:
        dose_tolerance_abs = DOSE_TOLERANCE_PERCENT * ref_grid.dose.max()

    # Position first
    xx_pos_ref, yy_pos_ref = np.meshgrid(ref_grid.coordinates[0], ref_grid.coordinates[1], indexing='ij')
    pos_ref_flatten = np.stack([xx_pos_ref.flatten(), yy_pos_ref.flatten()], axis=-1)

    xx_pos_eval, yy_pos_eval = np.meshgrid(eval_grid.coordinates[0], eval_grid.coordinates[1], indexing='ij')
    pos_eval_flatten = np.stack([xx_pos_eval.flatten(), yy_pos_eval.flatten()], axis=-1)

    dist_diff = pos_ref_flatten[:, None, :] - pos_eval_flatten
    sq_dist = (dist_diff ** 2).sum(axis=-1)

    # Dose then
    dose_ref_flatten = ref_grid.dose.flatten()
    dose_eval_flatten = eval_grid.dose.flatten()
    dose_diff = dose_ref_flatten[:, None] - dose_eval_flatten

    gammas = np.sqrt((dose_diff / dose_tolerance_abs) ** 2 + (np.sqrt(sq_dist) / DTA) ** 2)
    gamma = gammas.min(axis=1).reshape(ref_grid.dose.shape)

    return GammaResult(ref_grid=ref_grid, eval_grid=eval_grid, gamma=gamma)


def gamma_3d(ref_grid: DoseGrid, eval_grid: DoseGrid, dose_tolerance_abs: float = None) -> GammaResult:
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
