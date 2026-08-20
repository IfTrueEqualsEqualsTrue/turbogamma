from math import sqrt
from dataclasses import dataclass

import numpy as np

ABSOLUTE_DOSE_TOLERANCE = 4 * sqrt(2)
SPACING = 0.5


@dataclass
class Protocol:
    dose_difference: float
    dta: float
    dose_threshold: float
    local: bool = False
    interp_fraction: float = 10
    max_gamma: float = 3
    dose_tolerance_abs: float | None = None

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


@dataclass(frozen=True)
class GammaSearchResult:
    gamma: float
    shells_visited: int
    terminating_radius: float

def resolve_dose_tolerance(protocol: Protocol, ref_dose: np.ndarray, max_dose_ref_grid: float):
    if protocol.dose_tolerance_abs is not None:
        return protocol.dose_tolerance_abs
    if protocol.local:
        return protocol.dose_difference / 100 * ref_dose
    return protocol.dose_difference / 100 * max_dose_ref_grid  # single scalar