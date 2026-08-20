from math import ceil

import numpy as np
from numpy import pi


def shell_offsets(radius: float, step: float, n_dims: int) -> np.ndarray:  # (M, n_dims)
    if radius == 0.0:
        return np.zeros((1, n_dims))
    if n_dims == 1:
        return np.array([[radius], [-radius]])
    elif n_dims == 2:
        n_jumps = ceil(2 * pi * radius / step)
        angles = np.linspace(0, 2 * pi, n_jumps, endpoint=False)
        vectors = radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        return vectors
    elif n_dims == 3:
        n_theta = ceil(pi * radius / step) + 1  # polar angle, adding 1 because endpoints=True
        theta = np.linspace(0, pi, n_theta, endpoint=True)  # angles of the rings from origin
        ring_radii = radius * np.sin(theta)
        vectors = []
        for th, r in zip(theta, ring_radii):
            n_phi = max(1, ceil(2 * pi * r / step))  # angles of the points within the center of the ring
            phi = np.linspace(0, 2 * pi, n_phi, endpoint=False)  # same as 2d within the ring
            vectors.append(np.stack([r * np.cos(phi),
                                     r * np.sin(phi),
                                     np.full(phi.shape, radius * np.cos(th))], axis=-1))
        return np.concatenate(vectors)
    else:
        raise ValueError("n_dims should be 1, 2, or 3")


def radii_schedule(dta: float, interp_fraction: int, d_max: float):
    """ Generates the radii to visit to explore all the shells within d_max range, with a step of dta/interp_fraction"""
    if interp_fraction < 1:
        raise ValueError("Interpolation fraction should at least be 1")
    elif type(interp_fraction) != int:
        raise TypeError("Interpolation fraction should be an integer")
    step = dta / interp_fraction
    n = int(np.floor(d_max / step))
    radii = np.arange(n + 1) * step
    return radii, step
