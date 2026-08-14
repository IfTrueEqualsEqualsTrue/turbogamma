import json
import os
import re
import struct
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pymedphys as pmp

from scripts.epi import read_epi_content_file
from turbogamma.gamma import DoseGrid, GammaResult

banned_fixtures = [5, 7, 9, 14, 16, ]

DEFAULT_RAM = 12
DEFAULT_MAX_GAMMA = 3
FIXTURE_DIR = os.path.join("..", "tests", "golden_fixtures")
INPUT_DIR = os.path.join(FIXTURE_DIR, "input")
OUTPUT_DIR = os.path.join(FIXTURE_DIR, "output")


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


# pip install pymedphys[tests]==0.41.0
def compute_gamma_from_arrays(ref_grid: DoseGrid,
                              eval_grid: DoseGrid,
                              dose_difference: float,
                              dta: float,
                              dose_threshold: float,
                              local: bool,
                              interp_fraction: float = 10,
                              ramGB: int = DEFAULT_RAM,
                              max_gamma: object = 3,
                              ) -> np.ndarray:
    """
    Compute the gamma index from two arrays coming from a fluence map. Returns the avg gamma.
    :param ref_grid: reference dosemap & axes
    :param eval_grid: evaluated dosemap & axes
    :param dose_difference: dose difference criteria
    :param dta: distance-to-agreement criteria
    :param dose_threshold: dose threshold criteria
    :param local: normalization method
    :param interp_fraction: The fraction by which gamma distance threshold is divided into for interpolation. Defaults
           to 10 as recommended within <https://dx.doi.org/10.1118/1.2721657>. If a 3 mm distance threshold is chosen,
            this default value would mean that the evaluation grid is interpolated at a step size of 0.3 mm.
    :param ramGB: the ram in GB to be allowed for the computation
    :param max_gamma: the gamma cap value
    :return: the computed gamma index as a NumPy array
    """
    dose_reference = ref_grid.dose
    dose_evaluation = eval_grid.dose

    axes_reference = ref_grid.coordinates
    axes_evaluation = eval_grid.coordinates

    gamma_options = {
        'dose_percent_threshold': dose_difference,
        'distance_mm_threshold': dta,
        'lower_percent_dose_cutoff': dose_threshold,
        'interp_fraction': interp_fraction,
        'max_gamma': max_gamma,
        'random_subset': None,
        'local_gamma': local,
        'ram_available': ramGB * (2 ** 30),
    }

    gamma = pmp.gamma(
        axes_reference, dose_reference,
        axes_evaluation, dose_evaluation,
        **gamma_options)

    return gamma


def get_centered_grids(shape, x_min, y_min, x_max, y_max) -> tuple:
    """ Generates the axes for a 2D grid centered at 0 given the pixel spacing and the shape of the grid. """
    height, width = shape

    y = np.linspace(y_min, y_max, height)
    x = np.linspace(x_min, x_max, width)

    return x, y,


def get_dose_maps(ref_dose_path, eval_dose_path) -> tuple:
    """ Extract pixel data of the fluences for a given beam """
    try:
        ref_dose, ref_dim = read_epi_content_file(ref_dose_path)
        eval_dose, eval_dim = read_epi_content_file(eval_dose_path)
        return ref_dose, eval_dose, ref_dim, eval_dim
    except struct.error:
        raise FileNotFoundError("The file is not a valid EPI content file, or is None.")


def build_dose_grids(ref_dose, eval_dose, ref_dim, eval_dim) -> tuple[DoseGrid]:
    ref_axes = get_centered_grids(ref_dose.shape, *ref_dim)
    eval_axes = get_centered_grids(eval_dose.shape, *eval_dim)

    return DoseGrid(ref_axes, ref_dose), DoseGrid(eval_axes, eval_dose)


def ensure_folder(folder_name):
    target = Path(os.path.join(OUTPUT_DIR, folder_name))
    if target.exists() and target.is_dir():
        return
    else:
        os.mkdir(target)


def extract_epi_prefix(s):
    match = re.match(r"^(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)", s)

    if match:
        result = match.group()
        return result
    return None


class GammaComputation:

    def __init__(self, ref_dose_path, eval_dose_path, protocol: Protocol, prefix: str):
        self.ref_dose_path = ref_dose_path
        self.eval_dose_path = eval_dose_path
        self.protocol: Protocol = protocol
        self.ref_grid: DoseGrid = None
        self.eval_grid: DoseGrid = None
        self.gamma_result: GammaResult = None
        self.prefix: int = prefix
        self.target_folder = self.protocol.folder_name()
        ensure_folder(self.target_folder)
        self.filename = os.path.join(OUTPUT_DIR, self.target_folder, self.prefix)

    def build_dose_maps(self):
        ref_dose, eval_dose, ref_dim, eval_dim = get_dose_maps(self.ref_dose_path, self.eval_dose_path)
        self.ref_grid, self.eval_grid = build_dose_grids(ref_dose, eval_dose, ref_dim, eval_dim)

    def compute_gamma(self):
        gamma = compute_gamma_from_arrays(self.ref_grid, self.eval_grid, **asdict(self.protocol))
        self.gamma_result = GammaResult(self.ref_grid, self.eval_grid, gamma)

    def save_results(self):
        ref_axes_dict = {f"ref_axis_{i}": axis for i, axis in enumerate(self.ref_grid.coordinates)}
        eval_axes_dict = {f"eval_axis_{i}": axis for i, axis in enumerate(self.eval_grid.coordinates)}
        np.savez(f"{self.filename}.npz", ref_dose=self.ref_grid.dose, eval_dose=self.eval_grid.dose,
                 gamma=self.gamma_result.gamma, **ref_axes_dict, **eval_axes_dict)
        with open(f"{self.filename}_meta.json", "w") as f:
            meta = asdict(self.protocol)
            meta["prefix"] = self.prefix
            json.dump(meta, f)

    def run(self):
        self.build_dose_maps()
        self.compute_gamma()
        self.save_results()


def find_fixture_pairs(input_dir: Path) -> dict[str, dict[str, Path]]:
    """Group files by prefix (e.g. '0_1') into {'ref': path, 'eval': path}."""
    pairs = {}
    for f in Path(input_dir).glob("*.epi.content"):
        stem = f.name.removesuffix(".epi.content")
        prefix, role = stem.rsplit("_", 1)
        pairs.setdefault(prefix, {})[role] = f
    return pairs


if __name__ == "__main__":
    protocol = Protocol(dose_difference=3, dta=3, dose_threshold=20)

    for prefix, files in find_fixture_pairs(INPUT_DIR).items():
        if "ref" not in files or "eval" not in files:
            print(f"skipping {prefix}: incomplete pair ({files})")
            continue
        elif int(prefix.split("_")[0]) in banned_fixtures:
            print(f"skipping banned prefix :{prefix}")

        computation = GammaComputation(files["ref"], files["eval"], protocol, prefix)
        computation.run()
        print(f"done: {prefix}")
