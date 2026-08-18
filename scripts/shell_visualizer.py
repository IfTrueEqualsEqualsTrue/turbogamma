import numpy as np
from matplotlib import pyplot as plt

from turbogamma.geometry import shell_offsets


def plot_shells(radius: float = 1.0, step: float = 0.2) -> None:
    """Plot the generated shell points for 1D, 2D, and 3D. Vibe coded."""

    fig = plt.figure(figsize=(15, 5))

    # ------------------------------------------------------------------
    # 1D
    # ------------------------------------------------------------------
    offsets_1d = shell_offsets(radius, step, 1)

    ax1 = fig.add_subplot(1, 3, 1)

    ax1.scatter(
        offsets_1d[:, 0],
        np.zeros(len(offsets_1d)),
        s=100,
        color="red",
    )

    ax1.axhline(0, linewidth=1)
    ax1.axvline(0, linewidth=1, linestyle="--")

    ax1.set_title(
        f"1D shell ({len(offsets_1d)} points)"
    )
    ax1.set_xlabel("x")
    ax1.set_yticks([])

    ax1.set_xlim(-radius * 1.3, radius * 1.3)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_aspect("equal")

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------
    offsets_2d = shell_offsets(radius, step, 2)

    ax2 = fig.add_subplot(1, 3, 2)

    ax2.scatter(
        offsets_2d[:, 0],
        offsets_2d[:, 1],
        s=15,
        color="red"
    )

    # Draw the ideal shell for reference.
    circle = plt.Circle(
        (0, 0),
        radius,
        fill=False,
        linestyle="--",
        linewidth=1,
    )
    ax2.add_patch(circle)

    ax2.axhline(0, linewidth=1)
    ax2.axvline(0, linewidth=1)

    ax2.set_title(
        f"2D shell ({len(offsets_2d)} points)"
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax2.set_xlim(-radius * 1.2, radius * 1.2)
    ax2.set_ylim(-radius * 1.2, radius * 1.2)
    ax2.set_aspect("equal")

    # ------------------------------------------------------------------
    # 3D
    # ------------------------------------------------------------------
    offsets_3d = shell_offsets(radius, step, 3)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    ax3.scatter(
        offsets_3d[:, 0],
        offsets_3d[:, 1],
        offsets_3d[:, 2],
        s=8,
        color="red"
    )

    # Draw a wireframe sphere for reference.
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))

    ax3.plot_wireframe(
        x,
        y,
        z,
        linewidth=0.3,
        alpha=0.25,
    )

    ax3.set_title(
        f"3D shell ({len(offsets_3d)} points)"
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    ax3.set_xlim(-radius * 1.2, radius * 1.2)
    ax3.set_ylim(-radius * 1.2, radius * 1.2)
    ax3.set_zlim(-radius * 1.2, radius * 1.2)

    ax3.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_shells(
        radius=1.0,
        step=1.0,
    )
