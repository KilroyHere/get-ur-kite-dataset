"""Visual QA, drawn from the hands' surface templates (coloured by region).

    python -m kite_data.qa hands I4-M4-R4-T4_g0 I3-M0-R2-T2_g1 [--pose random]   # data/qa/hands.png
    python -m kite_data.qa examples data/examples/kite/val/I0-M2-R0-T4_g0.npz     # data/qa/examples.png

The examples view shows the hand at q_init (grey) and at the label q_goal (coloured), with the
contact targets and their outward normals in black: what the decoder is asked to do.
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


def world_points(hand, tpl, q, wrist=None):
    T = hand.fk(q, wrist)
    out = np.empty_like(tpl["pos"], dtype=float)
    for li, link in enumerate(hand.links):
        m = tpl["link"] == li
        out[m] = tpl["pos"][m] @ T[link][:3, :3].T + T[link][:3, 3]
    return out


def _square_axes(ax, P):
    span = (P.max(0) - P.min(0)).max() / 2
    mid = (P.max(0) + P.min(0)) / 2
    for setlim, m_ in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setlim(m_ - span, m_ + span)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def example_sheet(npz_path, out_png, n=4, hands_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .sample import _matrix

    z = np.load(npz_path)
    hand, tpl = load_hand(Path(hands_dir or config.DATA_DIR / "hands") / str(z["hand_id"]))
    regions = np.array(config.REGIONS)[tpl["region"]]
    fig = plt.figure(figsize=(4.2 * n, 4.4))
    for i in range(n):
        W = {}
        for tag in ("init", "goal"):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _matrix(z[f"wrist_quat_{tag}"][i : i + 1])[0], z[f"wrist_pos_{tag}"][i]
            W[tag] = world_points(hand, tpl, z[f"joints_{tag}"][i].astype(float), T) * 1000
        k = int(z["k"][i])
        x, nrm = z["contact_pos"][i, :k] * 1000, z["contact_normal"][i, :k]
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.scatter(*W["init"].T, s=0.3, c="#c9c9c9", depthshade=False)
        for r, c in COLORS.items():
            m = regions == r
            ax.scatter(*W["goal"][m].T, s=0.5, c=c, depthshade=False)
        ax.scatter(*x.T, s=40, c="k", depthshade=False)
        for p, v in zip(x, nrm):
            ax.plot(*np.stack([p, p + 15 * v]).T, c="k", lw=1.5)
        _square_axes(ax, np.concatenate([W["init"], W["goal"]]))
        moving = int(z["mask"][i].sum())
        ax.set_title(f"example {i}: K={k}, {moving}/{hand.n_dof} joints move", fontsize=9)
    fig.suptitle(f"{hand.id}: start pose (grey) -> label (coloured); contacts + normals (black)", fontsize=10)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


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
        _square_axes(ax, P)
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"{hand.id}  ({hand.n_dof} DoF)", fontsize=9)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hands", help="contact sheet of hands")
    h.add_argument("ids", nargs="+")
    h.add_argument("--pose", choices=["zero", "random"], default="zero")
    h.add_argument("--out", default=str(config.DATA_DIR / "qa" / "hands.png"))
    e = sub.add_parser("examples", help="the first n examples of one generated file")
    e.add_argument("file")
    e.add_argument("--n", type=int, default=4)
    e.add_argument("--out", default=str(config.DATA_DIR / "qa" / "examples.png"))
    for p in (h, e):
        p.add_argument("--hands", default=str(config.DATA_DIR / "hands"))
    args = ap.parse_args(argv)
    if args.cmd == "hands":
        contact_sheet([Path(args.hands) / i for i in args.ids], args.out, args.pose)
    else:
        example_sheet(args.file, args.out, args.n, args.hands)
    print(args.out)


if __name__ == "__main__":
    main()
