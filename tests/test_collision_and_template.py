import numpy as np

from kite_data import config
from kite_data.collision import Collider, exact_colliding_pairs, validity_gate
from kite_data.geometry import Geometry, long_links_of, sample_geometry
from kite_data.hand import Hand
from kite_data.template import build_template
from kite_data.topology import Topology

FULL = Topology.from_id("I4-M4-R4-T4")


def test_nominal_hand_passes_gate():
    h = Hand(FULL, Geometry())
    gate = validity_gate(h, n_samples=400)
    assert gate["ok"], gate
    assert 0.10 < gate["free_rate"] < 0.35  # ~19% measured for the full hand


def test_extreme_geometry_is_collision_free_at_zero():
    for s, shifts in [(1.2, {"index": -0.004, "middle": 0.004, "ring": -0.004}),
                      (0.8, {"index": 0.004, "middle": -0.004, "ring": 0.004})]:
        h = Hand(FULL, Geometry(scales={l: s for l in long_links_of(FULL)}, shifts=shifts, draw=99))
        assert not Collider(h).colliding_pairs(np.zeros(16))
        assert not exact_colliding_pairs(h, np.zeros(16), faces=config.GATE_MESH_FACES)


def test_box_model_agrees_with_meshes():
    """Menagerie's boxes vs. meshes on random configurations of a deformed hand (~95% at full resolution)."""
    h = Hand(FULL, sample_geometry(FULL, 3))
    col = Collider(h)
    Q = np.random.default_rng(2).uniform(h.lower, h.upper, size=(30, h.n_dof))
    box = np.array([col.collides(q) for q in Q])
    mesh = np.array([bool(exact_colliding_pairs(h, q, faces=config.GATE_MESH_FACES)) for q in Q])
    assert box.mean() > 0.5  # most random configurations collide
    assert (box == mesh).mean() >= 0.8


def test_template():
    t = Topology.from_id("I3-M0-R2-T4")
    h = Hand(t, sample_geometry(t, 2))
    tpl = build_template(h)
    n = config.TEMPLATE_SIZE
    assert tpl["pos"].shape == (n, 3) and tpl["normal"].shape == (n, 3) and tpl["link"].shape == (n,)
    assert np.allclose(np.linalg.norm(tpl["normal"], axis=1), 1, atol=1e-5)
    assert set(np.unique(tpl["link"])) <= set(range(len(h.links)))
    regions = {config.REGIONS[r] for r in np.unique(tpl["region"])}
    assert regions == {"palm", "index", "ring", "thumb"}  # middle finger absent
    assert len(np.unique(tpl["fps"])) == config.FPS_SIZE
    again = build_template(h)
    assert all(np.array_equal(tpl[k], again[k]) for k in tpl)
