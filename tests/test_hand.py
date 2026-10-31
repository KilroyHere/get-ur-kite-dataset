import numpy as np
import pytest

from kite_data import config
from kite_data.collision import Collider, to_mjcf
from kite_data.geometry import Geometry, sample_geometry
from kite_data.hand import Hand
from kite_data.leap import load_base_model
from kite_data.topology import Topology, enumerate_topologies

FULL = Topology.from_id("I4-M4-R4-T4")


def _cases():
    rng = np.random.default_rng(0)
    ts = enumerate_topologies()
    picks = [FULL] + [ts[i] for i in rng.choice(len(ts), 6, replace=False)]
    return [f"{t.id}_g{d}" for t in picks for d in (0, 2)]


@pytest.mark.parametrize("case", _cases())
def test_fk_matches_mujoco(case):
    import mujoco

    tid, draw = case.split("_g")
    t = Topology.from_id(tid)
    hand = Hand(t, sample_geometry(t, int(draw)))
    col = Collider(hand)
    m, d = col.model, col.data
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(hand.lower, hand.upper)
        d.qpos[:] = 0
        d.qpos[col.qadr] = q
        mujoco.mj_kinematics(m, d)
        T = hand.fk(q)
        for link in hand.links:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
            assert np.allclose(T[link][:3, 3], d.xpos[b], atol=1e-8)
            assert np.allclose(T[link][:3, :3], d.xmat[b].reshape(3, 3), atol=1e-8)


def test_nominal_full_hand_matches_base_urdf():
    base = load_base_model()
    h = Hand(FULL, Geometry())
    for j in h.joints:
        assert np.allclose(j.origin, base.joints[j.name].origin)
    assert [j.name for j in h.dof_joints] == ["1", "0", "2", "3", "5", "4", "6", "7", "9", "8", "10", "11", "12", "13", "14", "15"]


def test_locking_keeps_finger_geometry():
    """A locked finger has the same links and, at the zero pose, the same shape as the full finger."""
    g = sample_geometry(FULL, 4)
    full = Hand(FULL, g)
    t = Topology.from_id("I2-M3-R4-T2")
    locked = Hand(t, Geometry(scales=g.scales, shifts=g.shifts, draw=4))
    Tf, Tl = full.fk(np.zeros(full.n_dof)), locked.fk(np.zeros(locked.n_dof))
    for link in locked.links:
        assert np.allclose(Tf[link], Tl[link])
    assert locked.n_dof == 11 and len(locked.links) == len(full.links)
    # tip of the 2-joint index finger only moves with its two proximal joints
    assert locked.link_dof_ancestors("fingertip") == [0, 1]


def test_length_and_mount_changes():
    base = load_base_model()
    g = Geometry(scales={"dip": 1.2, "thumb_dip": 0.8}, shifts={"middle": 0.003}, draw=9)
    h = Hand(FULL, g)
    by_name = {j.name: j for j in h.joints}
    L = base.long_link_lengths
    # joint 3 (DIP of index) hangs off the dip link: moves by +0.2*L along the link's distal axis (-y)
    assert np.isclose(by_name["3"].origin[1, 3] - base.joints["3"].origin[1, 3], -0.2 * L["dip"])
    assert np.isclose(by_name["15"].origin[1, 3] - base.joints["15"].origin[1, 3], -0.2 * L["thumb_dip"])
    # middle-finger root moves along the palm's spacing axis; others don't
    assert np.allclose(by_name["5"].origin[:3, 3] - base.joints["5"].origin[:3, 3], [0, 0.003, 0])
    assert np.allclose(by_name["1"].origin, base.joints["1"].origin)


def test_graph_matrices():
    h = Hand(Topology.from_id("I3-M0-R2-T4"), Geometry())
    g = h.graph()
    n = h.n_dof + 1
    assert g["spd"].shape == (n, n) and np.array_equal(g["spd"], g["spd"].T)
    assert np.array_equal(g["desc"], g["anc"].T)
    assert (g["spd"] >= 0).all() and (np.diag(g["spd"]) == 0).all()
    # index root (node 1) to ring root (node 4) goes through the wrist
    assert g["spd"][1, 4] == 2 and g["anc"][3, 0] == 3
    gz = h.graph(include_wrist=False, sibling_edges=True)  # GET-Zero convention
    assert gz["spd"][0, 3] == 1


def test_descriptor_and_mjcf_for_every_topology():
    import mujoco

    for t in enumerate_topologies():
        h = Hand(t, sample_geometry(t, 1))
        d = h.descriptor()
        assert d["n_dof"] == t.n_dof == sum(not j["locked"] for j in d["joints"])
        assert sum(j["leaf"] for j in d["joints"]) == len(t.fingers)
        m = mujoco.MjModel.from_xml_string(to_mjcf(h))
        assert m.njnt == t.n_dof and m.nbody == len(h.links) + 1
