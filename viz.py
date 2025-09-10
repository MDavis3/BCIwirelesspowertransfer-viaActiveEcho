"""
Visualization helpers: heatmaps and rotation sweep.
"""

import numpy as np
import matplotlib.pyplot as plt


def _format_axes_cm(ax):
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")


def plot_heatmap(X_m: np.ndarray, Y_m: np.ndarray, Z: np.ndarray, title: str, outfile: str):
    """Plot heatmap with colorbar titled 'Relative PTE proxy'. X_m,Y_m are (n,n) in meters."""
    X_cm = X_m * 100.0
    Y_cm = Y_m * 100.0
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(Z, extent=[X_cm.min(), X_cm.max(), Y_cm.min(), Y_cm.max()],
                   origin='lower', aspect='equal', cmap='viridis')
    _format_axes_cm(ax)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative PTE proxy")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_rotation(angles_deg: np.ndarray, y_single: np.ndarray, y_steered: np.ndarray, title: str, outfile: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(angles_deg, y_single, label="single-coil")
    ax.plot(angles_deg, y_steered, label="steered")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Relative PTE proxy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_motion_trace(t: np.ndarray, pte_min: np.ndarray, pte_med: np.ndarray, outfile: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t, pte_min, label="min PTE")
    ax.plot(t, pte_med, label="median PTE")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative PTE proxy")
    ax.set_title("PTE over motion")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)


