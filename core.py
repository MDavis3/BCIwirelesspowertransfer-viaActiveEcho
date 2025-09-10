"""
Core physics utilities: dipole B-field, PTE proxy, grid generation.
Pure, vectorized NumPy. No hidden state.
"""

import numpy as np


MU0 = 4 * np.pi * 1e-7  # vacuum permeability


def unit_axis_from_angle(degrees: float) -> np.ndarray:
    """Return unit vector in x–z plane for given angle in degrees (0°=+z)."""
    theta = np.deg2rad(degrees)
    # angle measured from +z toward +x in x–z plane
    ux = np.sin(theta)
    uz = np.cos(theta)
    return np.array([ux, 0.0, uz], dtype=float)


def dipole_field_at_points(coil_pos: np.ndarray, coil_moment: np.ndarray, eval_pts: np.ndarray) -> np.ndarray:
    """Dipole B field at eval_pts from dipole at coil_pos with magnetic moment m.
    B = mu0/(4*pi*r^3) * (3 n (n·m) - m). Shapes:
    - coil_pos: (3,), coil_moment: (3,), eval_pts: (..., 3) -> returns (..., 3)
    """
    r_vec = eval_pts - coil_pos  # (..., 3)
    r2 = np.sum(r_vec * r_vec, axis=-1)
    r = np.sqrt(r2)
    # Avoid division by zero for points at the dipole location
    eps = 1e-12
    r_safe = np.maximum(r, eps)
    n_hat = r_vec / r_safe[..., None]
    n_dot_m = np.sum(n_hat * coil_moment, axis=-1)[..., None]
    term = 3.0 * n_hat * n_dot_m - coil_moment
    coeff = (MU0 / (4.0 * np.pi)) * (1.0 / (r_safe ** 3))[..., None]
    return coeff * term


def pte_proxy(B_vec: np.ndarray, axis_u: np.ndarray) -> np.ndarray:
    """Relative PTE proxy: |B · u| for B_vec (...,3) and unit axis_u (3,)."""
    return np.abs(np.sum(B_vec * axis_u, axis=-1))


def make_grid_xy(n: int, span_cm: float, depth_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an x–y grid (meters) over +/- span_cm/2 at fixed z=depth_m.
    Returns (grid_pts (n*n,3), X (n,n), Y (n,n)) for plotting.
    """
    half = (span_cm * 1e-2) / 2.0
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.full_like(X, depth_m)
    pts = np.stack([X, np.zeros_like(X), Z], axis=-1)
    pts[..., 1] = Y  # set y into second component
    return pts.reshape(-1, 3), X, Y


