import os
from pathlib import Path

import numpy as np

from pmp_gamma import OUTPUT_DIR, compute_gammas
from turbogamma.classes import Protocol, DoseGrid, GammaResult


def load_fixtures(protocol: Protocol) -> dict[str, GammaResult]:
    protocol_dir = protocol.folder_name()
    if not protocol_dir in os.listdir(OUTPUT_DIR):
        compute_gammas(protocol)
    gamma_results = {}

    npz_files = Path(os.path.join(OUTPUT_DIR, protocol_dir)).glob("*.npz")

    for npz in npz_files:
        prefix = str(npz.name).removesuffix(".npz")

        arrays: dict = np.load(npz)

        ref_grid = DoseGrid(tuple(arr for n, arr in arrays.items() if n.startswith("ref_axis")),
                            dose=arrays["ref_dose"])
        eval_grid = DoseGrid(tuple(arr for n, arr in arrays.items() if n.startswith("eval_axis")),
                             dose=arrays["eval_dose"])

        gamma_results[prefix] = GammaResult(gamma=arrays["gamma"], ref_grid=ref_grid, eval_grid=eval_grid, id=prefix)

    return gamma_results
