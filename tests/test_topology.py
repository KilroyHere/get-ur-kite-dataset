import pytest

from kite_data.topology import Topology, enumerate_topologies


def test_family_counts():
    all_ = enumerate_topologies()
    assert len(all_) == 243
    assert sum(t.has_thumb for t in all_) == 189
    assert len(enumerate_topologies(include_thumbless=False)) == 189
    assert len({t.id for t in all_}) == 243
    assert all(len(t.fingers) >= 2 and all(t.count(f) in (2, 3, 4) for f in t.fingers) for t in all_)


def test_id_round_trip():
    for t in enumerate_topologies():
        assert Topology.from_id(t.id) == t


def test_invalid():
    with pytest.raises(ValueError):
        Topology((1, 4, 4, 4))  # 1-joint fingers are not in the family
    with pytest.raises(ValueError):
        Topology((4, 0, 0, 0))  # fewer than two fingers
    with pytest.raises(ValueError):
        Topology.from_id("T4-I4-M4-R4")


def test_locking_is_tip_first():
    t = Topology.from_id("I2-M3-R4-T2")
    assert [t.is_locked("index", k) for k in range(4)] == [False, False, True, True]
    assert [t.is_locked("middle", k) for k in range(4)] == [False, False, False, True]
    assert not any(t.is_locked("ring", k) for k in range(4))
    assert t.n_dof == 11
