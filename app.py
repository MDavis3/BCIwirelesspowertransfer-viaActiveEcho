import argparse
import json
import numpy as np

from core import dipole_field_at_points, pte_proxy, unit_axis_from_angle, make_grid_xy
from control import coupling_for_implant, steering_weights_from_c, steer_weights_fair
from scene import Scene, scene_arrays
from viz import plot_heatmap, plot_rotation, plot_motion_trace


def single_vs_steered_maps(scene: Scene, depth_m: float, grid_n: int, budget: float, fair: bool):
    coil_pos, coil_mom, imp_pos, imp_axis = scene_arrays(scene)
    # grid
    pts, X, Y = make_grid_xy(grid_n, span_cm=12.0, depth_m=depth_m)
    # choose axis for title from first implant
    u = np.array(scene.implants[0].axis, dtype=float)

    # single-coil baseline: coil 0 only
    B0 = dipole_field_at_points(coil_pos[0], coil_mom[0], pts)
    heat_single = pte_proxy(B0, u).reshape(grid_n, grid_n)
    heat_single /= (heat_single.max() + 1e-12)

    # steered: compute weights for first implant (or fairness)
    if fair and len(scene.implants) > 1:
        c_rows = []
        for i in range(len(scene.implants)):
            c = coupling_for_implant(coil_pos, coil_mom, imp_pos[i], imp_axis[i], dipole_field_at_points)
            c_rows.append(c)
        cmat = np.stack(c_rows, axis=0)
        w = steer_weights_fair(cmat, budget=budget)
    else:
        c = coupling_for_implant(coil_pos, coil_mom, imp_pos[0], imp_axis[0], dipole_field_at_points)
        w = steering_weights_from_c(c, budget=budget)

    # field superposition
    B = np.zeros((pts.shape[0], 3))
    for k in range(coil_pos.shape[0]):
        if np.abs(w[k]) < 1e-15:
            continue
        B += w[k] * dipole_field_at_points(coil_pos[k], coil_mom[k], pts)
    heat_steer = pte_proxy(B, u).reshape(grid_n, grid_n)
    heat_steer /= (heat_steer.max() + 1e-12)

    return X, Y, heat_single, heat_steer, w


def rotation_sweep(scene: Scene, depth_m: float, budget: float, fair: bool):
    coil_pos, coil_mom, imp_pos, imp_axis = scene_arrays(scene)
    origin = np.array([0.0, 0.0, depth_m])
    angles = np.linspace(0, 180, 37)
    y_single, y_steer = [], []
    for a in angles:
        u = unit_axis_from_angle(a)
        # single
        B0 = dipole_field_at_points(coil_pos[0], coil_mom[0], origin.reshape(1, 3))[0]
        y_single.append(np.abs(np.dot(B0, u)))
        # steered
        if fair and len(scene.implants) > 1:
            c_rows = []
            for i in range(len(scene.implants)):
                c_rows.append(coupling_for_implant(coil_pos, coil_mom, imp_pos[i], imp_axis[i], dipole_field_at_points))
            w = steer_weights_fair(np.stack(c_rows, axis=0), budget=budget)
        else:
            c = coupling_for_implant(coil_pos, coil_mom, origin, u, dipole_field_at_points)
            w = steering_weights_from_c(c, budget=budget)
        B = np.zeros(3)
        for k in range(coil_pos.shape[0]):
            B += w[k] * dipole_field_at_points(coil_pos[k], coil_mom[k], origin.reshape(1, 3))[0]
        y_steer.append(np.abs(np.dot(B, u)))
    y_single = np.array(y_single)
    y_steer = np.array(y_steer)
    # normalize each to own max for visualization
    y_single = y_single / (y_single.max() + 1e-12)
    y_steer = y_steer / (y_steer.max() + 1e-12)
    return angles, y_single, y_steer


def run_motion(scene: Scene, duration_s: float, rate_hz: float, budget: float, fair: bool):
    coil_pos, coil_mom, imp_pos, imp_axis = scene_arrays(scene)
    t = np.arange(0.0, duration_s + 1e-9, 1.0 / rate_hz)
    pte_min = []
    pte_med = []
    budget_hits = 0
    recovery_steps = None
    prev_pte = None

    # motion: small drift + step at mid time
    yaw = np.zeros_like(t)
    yaw += 5.0 * np.sin(2 * np.pi * 0.1 * t)
    step_idx = int(len(t) / 2)
    yaw[step_idx:] += 20.0

    for idx, ti in enumerate(t):
        # update axes by yaw around y-axis (x–z plane rotation)
        rot = np.deg2rad(yaw[idx])
        R = np.array([[ np.cos(rot), 0, np.sin(rot)],
                      [ 0,           1, 0          ],
                      [-np.sin(rot), 0, np.cos(rot)]])
        axes_now = (R @ imp_axis.T).T

        # couplings for each implant
        c_rows = []
        for i in range(len(scene.implants)):
            c_rows.append(coupling_for_implant(coil_pos, coil_mom, imp_pos[i], axes_now[i], dipole_field_at_points))
        cmat = np.stack(c_rows, axis=0)

        # steering
        if fair and len(scene.implants) > 1:
            w = steer_weights_fair(cmat, budget=budget)
        else:
            w = steering_weights_from_c(cmat[0], budget=budget)

        if np.sum(np.abs(w)) >= budget - 1e-6:
            budget_hits += 1

        # evaluate PTE at each implant
        gains = cmat @ w  # signed projection
        ptes = np.abs(gains)

        pte_min.append(np.min(ptes))
        pte_med.append(np.median(ptes))

        # recovery metric after step
        if idx == step_idx - 1:
            prev_pte = np.min(ptes)
            recovery_steps = None
        if idx >= step_idx and recovery_steps is None and prev_pte is not None:
            if np.min(ptes) >= 0.8 * prev_pte:
                recovery_steps = idx - step_idx

    pte_min = np.array(pte_min)
    pte_med = np.array(pte_med)
    return t, pte_min / (pte_min.max() + 1e-12), pte_med / (pte_med.max() + 1e-12), budget_hits, (recovery_steps if recovery_steps is not None else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--axis", type=float, default=60.0)
    ap.add_argument("--depth", type=float, default=0.03)
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--fair", action="store_true")
    ap.add_argument("--motion", action="store_true")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--recover", type=float, default=0.8)
    args = ap.parse_args()

    scene = Scene.from_json(args.scene)

    # set axis of first implant for static maps
    if scene.implants:
        scene.implants[0].axis = list(unit_axis_from_angle(args.axis))

    X, Y, Hs, Hst, w = single_vs_steered_maps(scene, args.depth, args.grid, args.budget, args.fair)
    plot_heatmap(X, Y, Hs, f"Single-coil, axis={args.axis}°", "heatmap_single.png")
    plot_heatmap(X, Y, Hst, f"Steered, axis={args.axis}°", "heatmap_steered.png")

    ang, ys, yt = rotation_sweep(scene, args.depth, args.budget, args.fair)
    plot_rotation(ang, ys, yt, f"Rotation sweep (axis in x–z)", "rotation.png")

    metrics = {}
    metrics["weights_L1"] = float(np.sum(np.abs(w)))
    metrics["weights_budget"] = float(args.budget)
    metrics["gain_vs_baseline_60deg"] = float((yt[np.argmin(np.abs(ang-60))] + 1e-12) / (ys[np.argmin(np.abs(ang-60))] + 1e-12))
    metrics["gain_vs_baseline_90deg"] = float((yt[np.argmin(np.abs(ang-90))] + 1e-12) / (ys[np.argmin(np.abs(ang-90))] + 1e-12))

    # implant-wise PTE at their given positions with current weights
    coil_pos, coil_mom, imp_pos, imp_axis = scene_arrays(scene)
    c_rows = []
    for i in range(len(scene.implants)):
        c_rows.append(coupling_for_implant(coil_pos, coil_mom, imp_pos[i], imp_axis[i], dipole_field_at_points))
    if len(c_rows) > 0:
        cmat_now = np.stack(c_rows, axis=0)
        gains_now = cmat_now @ w
        ptes_now = np.abs(gains_now)
        # normalize to max for relativity
        norm = ptes_now.max() + 1e-12
        ptes_rel = ptes_now / norm
        metrics["min_pte_across_implants"] = float(ptes_rel.min())
        metrics["median_pte_across_implants"] = float(np.median(ptes_rel))

    # coverage above threshold in steered heatmap
    cover = float(np.mean(Hst >= args.thr))
    metrics["coverage_steered"] = cover
    metrics["coverage_threshold"] = float(args.thr)

    if args.motion:
        t, pmin, pmed, budget_hits, rec_steps = run_motion(scene, args.duration, args.rate, args.budget, args.fair)
        plot_motion_trace(t, pmin, pmed, "motion_trace.png")
        metrics["motion_budget_hits"] = int(budget_hits)
        metrics["recovery_steps"] = int(rec_steps)

    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("w L1=", metrics["weights_L1"], "/ budget=", metrics["weights_budget"])
    print("gain@60deg=", metrics["gain_vs_baseline_60deg"], "gain@90deg=", metrics["gain_vs_baseline_90deg"])
    if args.motion:
        print("motion budget hits=", metrics["motion_budget_hits"], "recovery steps=", metrics["recovery_steps"])


if __name__ == "__main__":
    main()


