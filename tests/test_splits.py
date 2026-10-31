from kite_data.splits import SUBSET_HOLDOUT, make_splits
from kite_data.topology import Topology


def test_counts_and_determinism():
    s = make_splits()
    assert s == make_splits()
    t = s["topologies"]
    assert (len(t["train"]), len(t["val"]), len(t["test_finger_subset"]), len(t["test_joint_count"]), len(t["test_geometry"])) == (145, 8, 36, 54, 10)
    h = s["hands"]
    assert (len(h["train"]), len(h["val"]), len(h["test_finger_subset"]), len(h["test_combined"]), len(h["test_joint_count"]), len(h["test_geometry"])) == (1015, 56, 36, 216, 378, 40)


def test_no_leakage():
    s = make_splits()
    t, h = s["topologies"], s["hands"]
    train, val = set(t["train"]), set(t["val"])
    tests = set(t["test_finger_subset"]) | set(t["test_joint_count"])
    assert not (train & val) and not (train & tests) and not (val & tests)
    assert len(train | val | tests) == 243
    assert set(t["test_geometry"]) <= train  # same layouts, unseen geometry
    all_hands = [x for v in h.values() for x in v]
    assert len(all_hands) == len(set(all_hands))  # every hand instance in exactly one split
    train_draws = {int(x.rsplit("_g", 1)[1]) for x in h["train"]}
    assert all(int(x.rsplit("_g", 1)[1]) not in train_draws for x in h["test_geometry"])


def test_bucket_rules():
    s = make_splits()
    for tid in s["topologies"]["test_finger_subset"]:
        assert frozenset(Topology.from_id(tid).fingers) in SUBSET_HOLDOUT
    for tid in s["topologies"]["train"] + s["topologies"]["val"]:
        t = Topology.from_id(tid)
        assert frozenset(t.fingers) not in SUBSET_HOLDOUT
        assert t.count("index") != 3
    assert all(Topology.from_id(x).count("index") == 3 for x in s["topologies"]["test_joint_count"])
    # thumb and ring still appear together in training, inside larger hands
    assert any({"thumb", "ring"} <= set(Topology.from_id(x).fingers) for x in s["topologies"]["train"])
