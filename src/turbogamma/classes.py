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


protocol_regular = Protocol(dose_difference=3, dta=3, dose_threshold=20)


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
