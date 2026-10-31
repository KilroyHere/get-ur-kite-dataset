"""Hand-level splits (DATA_GENERATION_PLAN.md, Stage B), fixed before any example is sampled.

    python -m kite_data.splits        # writes data/splits.json

Each topology belongs to exactly one of train / val / the three topology-level tests. The geometry
test reuses training topologies with geometry draws that training never sees (draws 7-10).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import config
from .topology import Topology, enumerate_topologies

# Finger-subset test: hands whose fingers are exactly one of these sets (agreed reading of the
# overview's "fingers never seen together"; thumb and ring still co-occur in larger training hands).
SUBSET_HOLDOUT = (frozenset({"thumb", "ring"}), frozenset({"thumb", "index", "ring"}))
# Joint-count test: every remaining hand with a 3-joint index finger.
JOINT_COUNT_HOLDOUT = ("index", 3)
N_VALIDATION = 8
N_GEOMETRY_TOPOLOGIES = 10
TRAIN_DRAWS = tuple(range(0, 7))  # nominal + 6 random geometries
GEOMETRY_TEST_DRAWS = tuple(range(7, 11))  # geometries never used in training
SPLIT_SEED = 0
SPLITS_PATH = config.DATA_DIR / "splits.json"


def _hands(topologies, draws):
    return [f"{t.id}_g{d}" for t in topologies for d in draws]


def make_splits(seed: int = SPLIT_SEED) -> dict:
    ts = enumerate_topologies()
    subset = [t for t in ts if frozenset(t.fingers) in SUBSET_HOLDOUT]
    rest = [t for t in ts if t not in subset]
    finger, count = JOINT_COUNT_HOLDOUT
    joint = [t for t in rest if t.count(finger) == count]
    rest = [t for t in rest if t not in joint]
    rng = np.random.default_rng(seed)
    val_idx = set(rng.choice(len(rest), N_VALIDATION, replace=False).tolist())
    val = [t for i, t in enumerate(rest) if i in val_idx]
    train = [t for i, t in enumerate(rest) if i not in val_idx]
    geo_idx = set(rng.choice(len(train), N_GEOMETRY_TOPOLOGIES, replace=False).tolist())
    geo = [t for i, t in enumerate(train) if i in geo_idx]
    topologies = {"train": train, "val": val, "test_finger_subset": subset, "test_joint_count": joint, "test_geometry": geo}
    hands = {
        "train": _hands(train, TRAIN_DRAWS),
        "val": _hands(val, TRAIN_DRAWS),
        "test_finger_subset": _hands(subset, (0,)),
        "test_combined": _hands(subset, TRAIN_DRAWS[1:]),
        "test_joint_count": _hands(joint, TRAIN_DRAWS),
        "test_geometry": _hands(geo, GEOMETRY_TEST_DRAWS),
    }
    return {
        "version": 1,
        "seed": seed,
        "rules": {
            "test_finger_subset": "fingers exactly {thumb, ring} or {thumb, index, ring}; nominal geometry",
            "test_combined": "the finger-subset topologies with random geometries (draws 1-6)",
            "test_joint_count": "every other hand with a 3-joint index finger; draws 0-6",
            "val": f"{N_VALIDATION} random remaining topologies; draws 0-6; model selection only",
            "train": "all other topologies; draws 0-6",
            "test_geometry": f"{N_GEOMETRY_TOPOLOGIES} training topologies with unseen geometries (draws 7-10)",
        },
        "topologies": {k: [t.id for t in v] for k, v in topologies.items()},
        "hands": hands,
    }


def load_splits(path=SPLITS_PATH) -> dict:
    if not path.exists():
        write_splits(path)
    return json.loads(path.read_text())


def write_splits(path=SPLITS_PATH, seed: int = SPLIT_SEED) -> dict:
    s = make_splits(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s, indent=1) + "\n")
    return s


def split_kind(split: str) -> str:
    """'train', 'val' or 'test' (selects the default examples-per-hand)."""
    return "test" if split.startswith("test") else split


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--out", default=str(SPLITS_PATH))
    args = ap.parse_args(argv)
    from pathlib import Path

    s = write_splits(Path(args.out), args.seed)
    for k, v in s["hands"].items():
        topo = s["topologies"].get(k, s["topologies"]["test_finger_subset"] if k == "test_combined" else [])
        print(f"{k:20s} {len(topo):4d} topologies  {len(v):5d} hands")
    print(args.out)


if __name__ == "__main__":
    main()
