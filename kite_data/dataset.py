"""Load generated examples for training.

    from kite_data.dataset import load_split
    data = load_split("train")            # dict of arrays over all hands in the split
    data["hand"][i]                       # index into data["hand_ids"]
    data["joints_goal"][i, :data["n_dof"][i]]

Joint arrays are padded to MAX_DOF (16) with zeros; `dof_valid` marks the real entries. Hand
descriptors and templates stay in data/hands/<hand_id>/ (load them once per hand).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

MAX_DOF = 16
JOINT_KEYS = ("joints_contact", "joints_init", "joints_goal", "mask")


def load_split(split: str, regime: str = "kite", root: Path | None = None) -> dict[str, np.ndarray]:
    root = Path(root or config.DATA_DIR / "examples" / regime) / split
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no examples in {root}; run python -m kite_data.generate --splits {split}")
    parts: dict[str, list] = {}
    hand_ids = []
    for hi, f in enumerate(files):
        z = np.load(f)
        n, nd = z["joints_goal"].shape
        hand_ids.append(str(z["hand_id"]))
        for key in z.files:
            if key in ("hand_id", "split", "regime", "seed", "kite_data"):
                continue
            a = z[key]
            if key in JOINT_KEYS:
                padded = np.zeros((n, MAX_DOF), dtype=a.dtype)
                padded[:, :nd] = a
                a = padded
            parts.setdefault(key, []).append(a)
        parts.setdefault("hand", []).append(np.full(n, hi, dtype=np.int32))
        parts.setdefault("n_dof", []).append(np.full(n, nd, dtype=np.int8))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["dof_valid"] = np.arange(MAX_DOF)[None, :] < data["n_dof"][:, None]
    data["hand_ids"] = np.array(hand_ids)
    return data
