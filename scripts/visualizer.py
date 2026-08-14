import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from mpl_toolkits.mplot3d.art3d import Path3DCollection

from scripts.golden_fixtures import load_fixtures
from scripts.pmp_gamma import protocol_regular
from scripts.providers import GammaProvider3d
from turbogamma.gamma_bruteforce import gamma_bruteforce_2d, gamma_bruteforce_1d
from turbogamma.classes import DoseGrid, GammaResult

POINT_SIZE = 100


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


def _scatter_volume(ax: plt.Axes, coordinates: tuple[np.ndarray], values: np.ndarray,
                    title: str) -> Path3DCollection:
    xx, yy, zz = np.meshgrid(*coordinates, indexing="ij")
    scatter = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(),
                         c=values.flatten(), s=POINT_SIZE, marker=".")
    ax.set_title(title)
    ax.set_xlabel("Distance in mm")
    ax.set_ylabel("Distance in mm")
    ax.set_zlabel("Distance in mm")
    return scatter


def plot_dose_3d(ax: plt.Axes, grid: DoseGrid, title: str = "") -> Path3DCollection:
    return _scatter_volume(ax, grid.coordinates, grid.dose, title)


def plot_gamma_3d(ax: plt.Axes, gamma_result: GammaResult, title: str = "") -> Path3DCollection:
    return _scatter_volume(ax, gamma_result.ref_grid.coordinates, gamma_result.gamma, title)


def plot_gamma_test(gamma_result: GammaResult, gamma_reference: GammaResult | None = None) -> None:
    dimension = len(gamma_result.ref_grid.coordinates)
    subplot_kw = {"projection": "3d"} if dimension == 3 else {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, subplot_kw=subplot_kw)
    if dimension == 1:
        plot_dose_1d(axes[0, 0], gamma_result.ref_grid, "Reference dose")
        plot_dose_1d(axes[0, 1], gamma_result.eval_grid, "Evaluation dose")
        plot_gamma_1d(axes[1, 0], gamma_result, "Gamma map")
        if gamma_reference is not None:
            plot_gamma_1d(axes[1, 1], gamma_reference, "Gamma reference")
    elif dimension == 2:
        mesh_dose = plot_dose_2d(axes[0, 0], gamma_result.ref_grid, "Reference dose")
        plot_dose_2d(axes[0, 1], gamma_result.eval_grid, "Evaluation dose")
        mesh_gamma = plot_gamma_2d(axes[1, 0], gamma_result, "Gamma map")
        if gamma_reference is not None:
            plot_gamma_2d(axes[1, 1], gamma_reference, "Gamma reference")
        fig.colorbar(mesh_dose, ax=[axes[0, 0], axes[0, 1]])
        fig.colorbar(mesh_gamma, ax=[axes[1, 0], axes[1, 1]])
    elif dimension == 3:
        mesh_dose = plot_dose_3d(axes[0, 0], gamma_result.ref_grid, "Reference dose")
        plot_dose_3d(axes[0, 1], gamma_result.eval_grid, "Evaluation dose")
        mesh_gamma = plot_gamma_3d(axes[1, 0], gamma_result, "Gamma map")
        if gamma_reference is not None:
            plot_gamma_3d(axes[1, 1], gamma_reference, "Gamma reference")
        fig.colorbar(mesh_dose, ax=[axes[0, 0], axes[0, 1]])
        fig.colorbar(mesh_gamma, ax=[axes[1, 0], axes[1, 1]])
    # plt.tight_layout()
    plt.show()


def main_1d() -> None:
    gammas = load_fixtures(protocol_regular)
    target = gammas["11_1"]
    size = target.ref_grid.dose.shape[0] // 2
    ref_grid = DoseGrid((target.ref_grid.coordinates[0],), target.ref_grid.dose[:, size])
    eval_grid = DoseGrid((target.eval_grid.coordinates[0],), target.eval_grid.dose[:, size])
    gamma = gamma_bruteforce_1d(ref_grid, eval_grid, protocol_regular)
    reference = GammaResult(gamma=target.gamma[:, size], ref_grid=ref_grid)
    plot_gamma_test(gamma_result=gamma, gamma_reference=reference)


def main_2d() -> None:
    # gamma = GammaProvider2d.get_square_feature(16, 2.0, 5)
    gammas = load_fixtures(protocol_regular)
    target = gammas["11_1"]
    gamma = gamma_bruteforce_2d(target.ref_grid, target.eval_grid, protocol_regular)
    reference = GammaResult(gamma=target.gamma, ref_grid=target.ref_grid)
    plot_gamma_test(gamma_result=gamma, gamma_reference=reference)


def main_3d() -> None:
    gamma = GammaProvider3d.get_square_feature(16, 2.0, 5)
    plot_gamma_test(gamma)


main_2d()
