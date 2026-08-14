import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from mpl_toolkits.mplot3d.art3d import Path3DCollection

from scripts.golden_fixtures import load_2d_fixtures
from scripts.pmp_gamma import protocol_regular
from scripts.providers import GammaProvider2d, GammaProvider1d, GammaProvider3d
from turbogamma.gamma import DoseGrid, GammaResult, gamma_2d

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


def plot_gamma_test(gamma_result: GammaResult) -> None:
    dimension = len(gamma_result.ref_grid.coordinates)
    subplot_kw = {"projection": "3d"} if dimension == 3 else {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, subplot_kw=subplot_kw)
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
    elif dimension == 3:
        mesh_dose = plot_dose_3d(axes[0, 0], gamma_result.ref_grid, "Reference dose")
        plot_dose_3d(axes[0, 1], gamma_result.eval_grid, "Evaluation dose")
        mesh_gamma = plot_gamma_3d(axes[1, 0], gamma_result, "Gamma map")
        fig.colorbar(mesh_dose, ax=[axes[0, 0], axes[0, 1]])
        fig.colorbar(mesh_gamma, ax=[axes[1, 0], axes[1, 1]])
    # plt.tight_layout()
    plt.show()


def main_1d() -> None:
    gamma = GammaProvider1d.get_square_feature(16, 2.0, 5)
    plot_gamma_test(gamma)


def main_2d() -> None:
    # gamma = GammaProvider2d.get_square_feature(16, 2.0, 5)
    gammas = load_2d_fixtures(protocol_regular)
    target = gammas["11_1"]
    gamma = gamma_2d(target.ref_grid, target.eval_grid)
    plot_gamma_test(gamma)


def main_3d() -> None:
    gamma = GammaProvider3d.get_square_feature(16, 2.0, 5)
    plot_gamma_test(gamma)



main_2d()
