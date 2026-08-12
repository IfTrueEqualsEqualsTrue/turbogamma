import matplotlib.pyplot as plt

from turbogamma.gamma import DoseGrid, GammaResult, gamma_1d
from turbogamma.scenarios import build_dose_ramp


def plot_dose_1d(ax: plt.Axes, grid: DoseGrid, title: str = "") -> None:
    ax.plot(grid.coordinates[0], grid.dose, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Distance in mm")
    ax.set_ylabel("Dose in Gy")


def plot_gamma_1d(ax: plt.Axes, gamma_result: GammaResult, title: str = "") -> None:
    ax.plot(gamma_result.ref_grid.coordinates[0], gamma_result.gamma, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Distance in mm")
    ax.set_ylabel("Gamma index")


def plot_gamma_test(gamma_result: GammaResult) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    if len(gamma_result.ref_grid.coordinates) == 1:
        plot_dose_1d(axes[0, 0], gamma_result.ref_grid, "Reference dose")
        plot_dose_1d(axes[0, 1], gamma_result.ref_grid, "Evaluation dose")
        plot_gamma_1d(axes[1, 0], gamma_result, "Gamma map")
    plt.tight_layout()
    plt.show()


def main_1d() -> None:
    size = 24
    constant_dose = 5.0
    ref_func = build_dose_ramp(size, constant_dose)
    evaluated_func = build_dose_ramp(size, constant_dose)
    ref_grid = ref_func(0)
    eval_grid = evaluated_func(5)
    gamma = gamma_1d(ref_grid, eval_grid)
    plot_gamma_test(gamma)
