import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh

from scripts.providers import GammaProvider2d, GammaProvider1d
from turbogamma.gamma import DoseGrid, GammaResult, gamma_1d, gamma_2d
from turbogamma.scenarios import Scenarios1d, Scenarios2d


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


def plot_dose_2d(ax: plt.Axes, grid: DoseGrid, title: str = "") -> QuadMesh:
    mesh = ax.pcolormesh(*grid.coordinates, grid.dose)
    ax.set_title(title)
    ax.set_xlabel("Distance in mm")
    ax.set_ylabel("Distance in mm")
    return mesh


def plot_gamma_2d(ax: plt.Axes, gamma_result: GammaResult, title: str = "") -> QuadMesh:
    mesh = ax.pcolormesh(*gamma_result.ref_grid.coordinates, gamma_result.gamma)
    ax.set_title(title)
    ax.set_xlabel("Distance in mm")
    ax.set_ylabel("Distance in mm")
    return mesh


def plot_gamma_test(gamma_result: GammaResult) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    dimension = len(gamma_result.ref_grid.coordinates)
    if dimension == 1:
        plot_dose_1d(axes[0, 0], gamma_result.ref_grid, "Reference dose")
        plot_dose_1d(axes[0, 1], gamma_result.eval_grid, "Evaluation dose")
        plot_gamma_1d(axes[1, 0], gamma_result, "Gamma map")
    elif dimension == 2:
        mesh_dose = plot_dose_2d(axes[0, 0], gamma_result.ref_grid, "Reference dose")
        plot_dose_2d(axes[0, 1], gamma_result.eval_grid, "Evaluation dose")
        mesh_gamma = plot_gamma_2d(axes[1, 0], gamma_result, "Gamma map")
        fig.colorbar(mesh_dose, ax=[axes[0, 0], axes[0, 1]])
        fig.colorbar(mesh_gamma, ax=[axes[1, 0], axes[1, 1]])
    # plt.tight_layout()
    plt.show()


def main_1d() -> None:
    gamma = GammaProvider1d.get_square_feature(16, 2.0, 5)
    plot_gamma_test(gamma)


def main_2d() -> None:
    gamma = GammaProvider2d.get_square_feature(16, 2.0, 5)
    plot_gamma_test(gamma)


main_2d()
