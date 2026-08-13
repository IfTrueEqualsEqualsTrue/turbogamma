from dataclasses import dataclass

import numpy as np

DOSE_TOLERANCE_PERCENT = 0.02
DTA = 2


@dataclass
class DoseGrid:
    coordinates: tuple[np.ndarray]
    dose: np.ndarray


@dataclass
class GammaResult:
    ref_grid: DoseGrid
    eval_grid: DoseGrid
    gamma: np.ndarray


def gamma_1d(ref_grid: DoseGrid, eval_grid: DoseGrid, dose_tolerance_abs: float = None) -> GammaResult:
    if dose_tolerance_abs is None:
        dose_tolerance_abs = DOSE_TOLERANCE_PERCENT * ref_grid.dose.max()

    ref_pos_row = ref_grid.coordinates[0]
    ref_pos_col = ref_pos_row.reshape((len(ref_pos_row), 1))
    eval_pos_row = eval_grid.coordinates[0]
    ref_dose_col = ref_grid.dose.reshape((len(ref_pos_row), 1))
    dose_diff = ref_dose_col - eval_grid.dose
    pos_diff = ref_pos_col - eval_pos_row
    gammas: np.ndarray = np.sqrt((dose_diff / dose_tolerance_abs) ** 2 + (pos_diff / DTA) ** 2)
    gamma = gammas.min(axis=1)
    return GammaResult(ref_grid, eval_grid, gamma)


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

    return GammaResult(ref_grid, eval_grid, gamma)


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

    return GammaResult(ref_grid, eval_grid, gamma)
