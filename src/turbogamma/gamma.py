from dataclasses import dataclass

import numpy as np

DOSE_TOLERANCE = 0.02
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


def gamma_1d(ref_grid: DoseGrid, eval_grid: DoseGrid) -> GammaResult:
    ref_pos_row = ref_grid.coordinates[0]
    ref_pos_col = ref_pos_row.reshape((len(ref_pos_row), 1))
    eval_pos_row = eval_grid.coordinates[0]
    ref_dose_col = ref_grid.dose.reshape((len(ref_pos_row), 1))
    dose_diff = ref_dose_col - eval_grid.dose
    pos_diff = ref_pos_col - eval_pos_row
    gammas: np.ndarray = np.sqrt((dose_diff / DOSE_TOLERANCE) ** 2 + (pos_diff / DTA) ** 2)
    gamma = gammas.min(axis=1)
    return GammaResult(ref_grid, eval_grid, gamma)
