import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from turbogamma_tools.pmp_gamma import OUTPUT_DIR, compute_gammas
from turbogamma.classes import Protocol, DoseGrid, GammaResult, protocol_regular


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


def compute_fixture_metrics(protocol: Protocol) -> dict[str, dict]:
    """Summarize each precomputed pymedphys golden fixture (no gamma_shell run).

    Reports the gamma distribution and pass rate of the stored fixture so you
    can tell which maps are trivial (everything passes) and which are
    discriminating. Run once to decide which fixtures to keep in the test."""
    protocol_golden = dataclasses.replace(protocol, dose_tolerance_abs=None, dose_threshold=20)
    fixtures = load_fixtures(protocol_golden)

    metrics: dict[str, dict] = {}
    for prefix, fixture in fixtures.items():
        gamma = fixture.gamma
        valid = ~np.isnan(gamma)
        gamma_valid = gamma[valid]

        metrics[prefix] = {
            "shape": list(gamma.shape),
            "n_total": int(gamma.size),
            "n_valid": int(valid.sum()),
            "frac_nan": float(np.mean(~valid)),
            "pass_rate": float(np.mean(gamma_valid < 1)),
            "gamma_median": float(np.median(gamma_valid)),
            "gamma_mean": float(np.mean(gamma_valid)),
            "gamma_p95": float(np.percentile(gamma_valid, 95)),
            "gamma_max": float(np.max(gamma_valid)),
        }
        print(f"Fixture {prefix} done")

    return metrics


def write_fixture_metrics(protocol: Protocol = protocol_regular,
                          path: str | os.PathLike = None) -> str:
    """Write per-fixture summary metrics to a JSON file and return its path."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "fixture_metrics.json")

    metrics = compute_fixture_metrics(protocol)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    return str(path)


if __name__ == "__main__":
    out = write_fixture_metrics()
    print(f"Wrote fixture metrics to {out}")
