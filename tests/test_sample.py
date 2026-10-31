import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kite_data import config
from kite_data.build_hands import make_hand
from kite_data.sample import REGIMES, HandSampler, _matrix, verify
from kite_data.template import build_template


@pytest.fixture(scope="module", params=["I4-M4-R4-T4_g0", "I3-M0-R2-T2_g1", "I0-M3-R3-T0_g2"])
def sampler(request):
    tid, draw = request.param.rsplit("_g", 1)
    hand = make_hand(tid, int(draw), 0, 0)
    return HandSampler(hand, build_template(hand))


def test_invariants(sampler):
    ex = sampler.sample(400, 0)
    v = verify(sampler, ex, check_collisions=400)
    assert v["ok"], v
    assert np.all(np.abs(ex["wrist_pos_contact"]) <= 0.4 + 1e-6)
    assert np.allclose(np.linalg.norm(ex["wrist_quat_contact"], axis=1), 1, atol=1e-6)
    assert np.all(ex["wrist_quat_contact"][:, 0] >= 0)


def test_deterministic(sampler):
    a, b = sampler.sample(50, 7), sampler.sample(50, 7)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    c = sampler.sample(50, 8)
    assert not np.array_equal(a["joints_contact"], c["joints_contact"])


def test_contacts_are_region_balanced(sampler):
    ex = sampler.sample(1500, 1)
    tidx = ex["template_idx"].astype(int)
    picked = sampler.tpl_region[tidx[tidx >= 0]]
    freq = np.array([np.mean(picked == r) for r in sampler.regions])
    assert np.allclose(freq, 1 / len(sampler.regions), atol=0.03)
    counts = np.bincount(ex["k"], minlength=7)[2:]
    assert np.allclose(counts / counts.sum(), 0.2, atol=0.05)


def test_noise_magnitudes():
    hand = make_hand("I4-M4-R4-T4", 0, 0, 0)
    s = HandSampler(hand, build_template(hand))
    ex = s.sample(3000, 2)
    rng_ = s.upper - s.lower
    d = (ex["joints_init"] - ex["joints_contact"]) / rng_
    inside = (ex["joints_init"] > s.lower + 1e-5) & (ex["joints_init"] < s.upper - 1e-5)
    assert abs(d[inside].std() - 0.05) < 0.006  # clamping trims the tails slightly
    assert abs((ex["wrist_pos_init"] - ex["wrist_pos_contact"]).std() - 0.005) < 0.0003
    rel = Rotation.from_matrix(np.swapaxes(_matrix(ex["wrist_quat_contact"]), 1, 2) @ _matrix(ex["wrist_quat_init"]))
    mean_angle = np.degrees(rel.magnitude()).mean()
    assert abs(mean_angle - 5.0 * np.sqrt(8 / np.pi)) < 0.4  # chi(3) mean of a 5 deg per-axis rotation vector
    assert not ex["wide"].any()
    wide = s.sample(2000, 3, REGIMES["wide"])
    assert abs(wide["wide"].mean() - 0.2) < 0.03


def test_mask_means_what_it_says():
    """Moving an unmasked joint never moves a selected contact; moving a masked one usually does."""
    hand = make_hand("I4-M4-R4-T4", 2, 0, 0)
    s = HandSampler(hand, build_template(hand))
    ex = s.sample(100, 4)
    tidx = ex["template_idx"].astype(int)
    pos, R = ex["wrist_pos_goal"].astype(float), _matrix(ex["wrist_quat_goal"])
    base, _ = s.contact_points(ex["joints_goal"].astype(float), pos, R, tidx)
    moved_masked, total_masked = 0, 0
    for j in range(hand.n_dof):
        q = ex["joints_goal"].astype(float).copy()
        q[:, j] += 0.2
        x, _ = s.contact_points(q, pos, R, tidx)
        delta = np.abs(x - base).max(axis=(1, 2))
        m = ex["mask"][:, j]
        assert np.all(delta[~m] < 1e-9)
        moved_masked += int(np.sum(delta[m] > 1e-4))
        total_masked += int(m.sum())
    assert moved_masked / total_masked > 0.95
