"""
Active Echo control rules: coupling computation and steering weights.
Pure NumPy, vectorized. Binary phase (+/-1) with amplitudes ∝ |c_k|.
"""

import numpy as np


def coupling_for_implant(coil_positions: np.ndarray,
                         coil_moments: np.ndarray,
                         implant_pos: np.ndarray,
                         implant_axis: np.ndarray,
                         field_fn) -> np.ndarray:
    """Compute AE couplings c_k = B_k(p)·u for each coil.
    field_fn: callable(pos, moment, eval_pts)->B_vecs, using one point here.
    Returns (N_coils,) real couplings.
    """
    N = coil_positions.shape[0]
    eval_pt = implant_pos.reshape(1, 3)
    c_list = []
    for k in range(N):
        Bk = field_fn(coil_positions[k], coil_moments[k], eval_pt)[0]
        c_list.append(float(np.dot(Bk, implant_axis)))
    return np.array(c_list, dtype=float)


def steering_weights_from_c(c: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """Single-implant steering: w_k = sign(c_k)|c_k| / sum|c_k| scaled to budget.
    Returns weights summing to L1 = budget (unless all zero).
    """
    mags = np.abs(c)
    s = np.sum(mags)
    if s <= 1e-12:
        return np.zeros_like(c)
    w = np.sign(c) * (mags / s)
    return w * budget


def steer_weights_fair(c_matrix: np.ndarray, budget: float = 1.0, iters: int = 50) -> np.ndarray:
    """Fairness heuristic: maximize min_i |Σ_k w_k c_k^(i)| with L1 budget.
    c_matrix: (N_implants, N_coils)
    Returns weights with L1 norm ~ budget and binary phases.
    """
    n_impl, n_coils = c_matrix.shape
    w = np.zeros(n_coils, dtype=float)
    for _ in range(iters):
        gains = c_matrix @ w  # (N_implants,)
        worst = int(np.argmin(np.abs(gains)))
        c = c_matrix[worst]
        step = np.sign(c) * np.abs(c)
        s = np.sum(np.abs(step)) + 1e-12
        step = step / s
        w = w + 0.1 * step
        # project onto L1 ball with soft scaling
        l1 = np.sum(np.abs(w)) + 1e-12
        if l1 > budget:
            w *= (budget / l1)
    # snap to binary phase with proportional amplitudes
    l1 = np.sum(np.abs(w)) + 1e-12
    return np.sign(w) * (np.abs(w) / l1) * budget


