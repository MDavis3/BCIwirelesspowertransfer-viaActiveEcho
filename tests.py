import numpy as np

from core import dipole_field_at_points, unit_axis_from_angle, pte_proxy, make_grid_xy
from control import steering_weights_from_c, steer_weights_fair, coupling_for_implant
from scene import Scene
from viz import plot_heatmap, plot_rotation
from app import single_vs_steered_maps, rotation_sweep
import os


def test_inverse_cube():
    m = np.array([0, 0, 1.0])
    p = np.array([0.0, 0.0, 0.0])
    B1 = dipole_field_at_points(p, m, np.array([[0, 0, 0.1]]))[0]
    B2 = dipole_field_at_points(p, m, np.array([[0, 0, 0.2]]))[0]
    r = np.linalg.norm(B1) / (np.linalg.norm(B2) + 1e-12)
    assert 7.9 < r < 8.1  # (0.2/0.1)^3


def test_weights_budget():
    c = np.array([1.0, -2.0, 0.5])
    w = steering_weights_from_c(c, budget=1.0)
    assert abs(np.sum(np.abs(w)) - 1.0) < 1e-8


def test_fairness_shapes():
    cmat = np.array([[1.0, -0.5, 0.2], [0.1, 0.3, -0.4]])
    w = steer_weights_fair(cmat, budget=0.5, iters=10)
    assert w.shape == (3,)
    assert np.sum(np.abs(w)) <= 0.5 + 1e-8


def test_axis_helper():
    u0 = unit_axis_from_angle(0)
    u90 = unit_axis_from_angle(90)
    assert np.allclose(u0, np.array([0.0, 0.0, 1.0]), atol=1e-9)
    assert np.allclose(u90, np.array([1.0, 0.0, 0.0]), atol=1e-9)


def test_pte_proxy_alignment():
    B = np.array([[0, 0, 2.0], [1.0, 0, 0]])
    u = np.array([0.0, 0.0, 1.0])
    p = pte_proxy(B, u)
    assert p[0] > p[1]


def test_grid_shape_and_units():
    pts, X, Y = make_grid_xy(11, span_cm=12.0, depth_m=0.03)
    assert pts.shape == (121, 3)
    assert X.min() == -0.06 and X.max() == 0.06
    assert Y.min() == -0.06 and Y.max() == 0.06


def test_coupling_sign_consistency():
    coil_pos = np.array([[0.0, 0.0, 0.0]])
    coil_mom = np.array([[0.0, 0.0, 1.0]])
    ip = np.array([0.0, 0.0, 0.05])
    ia = np.array([0.0, 0.0, 1.0])
    c = coupling_for_implant(coil_pos, coil_mom, ip, ia, dipole_field_at_points)
    assert c[0] > 0.0


def test_steering_gain_formula():
    # Check gain equals budget * sum|c|^2 / sum|c|
    rng = np.random.default_rng(0)
    c = rng.normal(size=5)
    w = steering_weights_from_c(c, budget=1.0)
    gain = float(np.dot(w, c))
    expected = (np.sum(np.abs(c)**2) / (np.sum(np.abs(c)) + 1e-12))
    assert abs(gain - expected) < 1e-9


def test_fairness_improves_min_gain():
    # Two implants need different coils
    cmat = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_fair = steer_weights_fair(cmat, budget=1.0, iters=50)
    gains_fair = np.abs(cmat @ w_fair)
    w_single = steering_weights_from_c(cmat[0], budget=1.0)
    gains_single = np.abs(cmat @ w_single)
    assert gains_fair.min() > gains_single.min()


def test_scene_roundtrip():
    scene = Scene.from_json("scenes/two_coils.json")
    path = "scenes/tmp_roundtrip.json"
    scene.to_json(path)
    scene2 = Scene.from_json(path)
    os.remove(path)
    assert scene.name == scene2.name
    assert len(scene.coils) == len(scene2.coils)
    assert len(scene.implants) == len(scene2.implants)


def test_viz_smoke():
    X = np.linspace(-0.01, 0.01, 5)
    Y = np.linspace(-0.01, 0.01, 5)
    Xg, Yg = np.meshgrid(X, Y, indexing='xy')
    Z = np.hypot(Xg, Yg)
    plot_heatmap(Xg, Yg, Z, "smoke", "_smoke_heat.png")
    assert os.path.exists("_smoke_heat.png")
    os.remove("_smoke_heat.png")
    plot_rotation(np.linspace(0, 180, 5), np.linspace(0, 1, 5), np.linspace(1, 0, 5), "smoke", "_smoke_rot.png")
    assert os.path.exists("_smoke_rot.png")
    os.remove("_smoke_rot.png")


def test_app_helpers_smoke():
    from scene import Scene
    scene = Scene.from_json("scenes/three_coils.json")
    X, Y, Hs, Hst, w = single_vs_steered_maps(scene, depth_m=0.03, grid_n=21, budget=1.0, fair=False)
    assert Hs.shape == Hst.shape == (21, 21)
    angles, ys, yt = rotation_sweep(scene, depth_m=0.03, budget=1.0, fair=False)
    assert len(angles) == len(ys) == len(yt)


def run_all():
    test_inverse_cube()
    test_weights_budget()
    test_fairness_shapes()
    test_axis_helper()
    test_pte_proxy_alignment()
    test_grid_shape_and_units()
    test_coupling_sign_consistency()
    test_steering_gain_formula()
    test_fairness_improves_min_gain()
    test_scene_roundtrip()
    test_viz_smoke()
    test_app_helpers_smoke()
    print("All tests passed.")


if __name__ == "__main__":
    run_all()


