import numpy as np
import pytest

from kite_data import config
from kite_data.geometry import AxisStretch, BoxShift, Compose, link_deformation, sample_geometry
from kite_data.leap import load_base_model
from kite_data.topology import Topology


def _random_unit(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.parametrize("sign,delta", [(+1, 0.009), (+1, -0.008), (-1, 0.0124), (-1, -0.0099)])
def test_axis_stretch(sign, delta):
    L = 0.04
    D = AxisStretch(sign, L, delta)
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.02, 0.07, size=(2000, 3)) * np.array([1, sign, 1])
    Q = D.map_points(P)
    u, u2 = sign * P[:, 1], sign * Q[:, 1]
    assert np.allclose(Q[:, [0, 2]], P[:, [0, 2]])
    assert np.allclose(u2[u < D.u0], u[u < D.u0])
    assert np.allclose(u2[u > D.u1], u[u > D.u1] + delta)
    assert np.all(np.diff(u2[np.argsort(u)]) >= -1e-15)  # monotone: never folds
    N = _random_unit(len(P), rng)
    assert np.allclose(np.linalg.norm(D.map_normals(P, N), axis=1), 1)
    assert np.all(D.area_factor(P, N) > 0)


def test_box_shift_never_folds():
    D = BoxShift([0, 0, 0], [0.011, 0.015, 0.019], config.MOUNT_AXIS, config.MOUNT_SHIFT_MAX, config.MOUNT_FALLOFF)
    rng = np.random.default_rng(1)
    P = rng.uniform(-0.04, 0.04, size=(5000, 3))
    assert np.all(np.linalg.det(D.jacobian(P)) >= 1 - config.MOUNT_SHIFT_MAX / config.MOUNT_FALLOFF - 1e-12)
    inside = np.all(np.abs(P) <= [0.011, 0.015, 0.019], axis=1)
    assert np.allclose(D.map_points(P[inside]) - P[inside], np.array(config.MOUNT_AXIS) * config.MOUNT_SHIFT_MAX)
    # Jacobian matches finite differences
    eps = 1e-7
    p = np.array([[0.016, 0.019, -0.021]])
    J = D.jacobian(p)[0]
    Jfd = np.stack([(D.map_points(p + eps * e) - D.map_points(p - eps * e))[0] / (2 * eps) for e in np.eye(3)], axis=1)
    assert np.allclose(J, Jfd, atol=1e-6)


def test_mount_regions_disjoint():
    base = load_base_model()
    boxes = {g.name: g for g in base.collision[config.PALM]}
    centers = [boxes[config.MOUNT_BOX[f]].pos for f in config.MOUNT_FINGERS]
    reach = boxes[config.MOUNT_BOX["index"]].half_size[1] + config.MOUNT_CORE_GROW + config.MOUNT_FALLOFF
    assert min(abs(a[1] - b[1]) for a, b in zip(centers, centers[1:])) > 2 * reach


def test_sampling_is_deterministic_and_in_range():
    t = Topology.from_id("I4-M3-R0-T2")
    assert sample_geometry(t, 0).scales == {} and sample_geometry(t, 0).shifts == {}
    g1, g2 = sample_geometry(t, 3, seed=5), sample_geometry(t, 3, seed=5)
    assert g1 == g2 and g1 != sample_geometry(t, 4, seed=5) and g1 != sample_geometry(t, 3, seed=5, attempt=1)
    lo, hi = config.LENGTH_SCALE_RANGE
    assert all(lo <= s <= hi for s in g1.scales.values())
    assert set(g1.scales) == {"mcp_joint", "dip", "fingertip", "mcp_joint_2", "dip_2", "fingertip_2", "thumb_dip", "thumb_fingertip"}
    assert set(g1.shifts) == {"index", "middle"}  # ring absent, thumb mount fixed
    assert all(abs(s) <= config.MOUNT_SHIFT_MAX for s in g1.shifts.values())


def test_brackets_never_deform():
    base = load_base_model()
    t = Topology.from_id("I4-M4-R4-T4")
    g = sample_geometry(t, 1)
    for link in ("pip", "pip_2", "pip_3", "pip_4", "thumb_pip"):
        assert type(link_deformation(base, link, g)).__name__ == "Identity"
    assert isinstance(link_deformation(base, config.PALM, g), Compose)
