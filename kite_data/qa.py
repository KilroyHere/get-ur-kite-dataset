"""Visual QA: contact sheets of hands drawn from their surface templates, coloured by region.

    python -m kite_data.qa I4-M4-R4-T4_g0 I3-M0-R2-T2_g1 --out data/qa/sheet.png [--pose random]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import config
from .build_hands import make_hand
from .collision import Collider

COLORS = {"palm": "#8a8a8a", "index": "#2a6fdb", "middle": "#2fa84f", "ring": "#e08a1e", "thumb": "#c8324b"}


def load_hand(hand_dir: Path):
    desc = json.loads((hand_dir / "hand.json").read_text())
    tid, draw = desc["id"].rsplit("_g", 1)
    hand = make_hand(tid, int(draw), desc["build"]["seed"], desc["build"]["geometry_attempt"])
    return hand, dict(np.load(hand_dir / "template.npz"))


def world_points(hand, tpl, q):
    T = hand.fk(q)
    out = np.empty_like(tpl["pos"], dtype=float)
    for li, link in enumerate(hand.links):
        m = tpl["link"] == li
        out[m] = tpl["pos"][m] @ T[link][:3, :3].T + T[link][:3, 3]
    return out


def contact_sheet(hand_dirs, out_png, pose="zero", seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(hand_dirs)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(4.2 * cols, 4.2 * rows))
    rng = np.random.default_rng(seed)
    for k, d in enumerate(hand_dirs):
        hand, tpl = load_hand(Path(d))
        q = np.zeros(hand.n_dof)
        if pose == "random":
            col = Collider(hand)
            while True:
                q = rng.uniform(hand.lower, hand.upper)
                if not col.collides(q):
                    break
        P = world_points(hand, tpl, q) * 1000
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        regions = np.array(config.REGIONS)[tpl["region"]]
        for r, c in COLORS.items():
            m = regions == r
            ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=0.6, c=c, label=r, depthshade=False)
        fps = P[tpl["fps"]]
        ax.scatter(fps[:, 0], fps[:, 1], fps[:, 2], s=12, c="k", marker="x")
        span = (P.max(0) - P.min(0)).max() / 2
        mid = (P.max(0) + P.min(0)) / 2
        for setlim, m_ in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
            setlim(m_ - span, m_ + span)
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"{hand.id}  ({hand.n_dof} DoF)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--hands", default=str(config.DATA_DIR / "hands"))
    ap.add_argument("--out", default=str(config.DATA_DIR / "qa" / "sheet.png"))
    ap.add_argument("--pose", choices=["zero", "random"], default="zero")
    args = ap.parse_args(argv)
    contact_sheet([Path(args.hands) / i for i in args.ids], args.out, args.pose)
    print(args.out)


if __name__ == "__main__":
    main()
