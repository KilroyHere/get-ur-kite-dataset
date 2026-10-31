"""Generate training examples for the hands in data/splits.json (KITE's recipe; see sample.py).

    python -m kite_data.generate                                  # every split, default counts
    python -m kite_data.generate --splits val --per-hand val=200   # a quick sample

Writes data/examples/<regime>/<split>/<hand_id>.npz, one file per hand, plus summary.json and
index.csv per regime. Hands that are not built yet are built first. Every file is checked with
sample.verify() before it is written; seeds differ per (split, hand), so no two files share draws.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path

import numpy as np

from . import __version__, config
from .build_hands import build_one, make_hand
from .sample import REGIMES, HandSampler, verify
from .splits import load_splits, split_kind

DEFAULT_PER_HAND = {"train": 1000, "val": 1000, "test": 2000}  # ~1M training examples, as in the overview
EXAMPLES_DIR = config.DATA_DIR / "examples"


def load_hand(hand_id: str, hands_dir: Path, build_seed: int = 0):
    d = hands_dir / hand_id
    if not (d / "hand.json").exists():
        tid, draw = hand_id.rsplit("_g", 1)
        build_one((tid, int(draw), build_seed, str(hands_dir), config.GATE_SAMPLES))
    desc = json.loads((d / "hand.json").read_text())
    tid, draw = hand_id.rsplit("_g", 1)
    hand = make_hand(tid, int(draw), desc["build"]["seed"], desc["build"]["geometry_attempt"])
    return hand, dict(np.load(d / "template.npz"))


def generate_one(job) -> dict:
    hand_id, split, n, seed, regime, hands_dir, out_dir = job
    t0 = time.time()
    hand, tpl = load_hand(hand_id, Path(hands_dir))
    sampler = HandSampler(hand, tpl)
    rng = np.random.default_rng([seed, zlib.crc32(split.encode()), zlib.crc32(hand_id.encode())])
    ex = sampler.sample(n, rng, REGIMES[regime])
    check = verify(sampler, ex, check_collisions=min(n, 50))
    if not check["ok"]:
        raise RuntimeError(f"{hand_id}: invariant check failed: {check}")
    path = Path(out_dir) / split / f"{hand_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ex, hand_id=hand_id, split=split, regime=regime, seed=seed, kite_data=__version__)
    return {
        "hand_id": hand_id, "split": split, "n": n, "n_dof": hand.n_dof,
        "contact_accept_rate": round(sampler.accepted / sampler.tried, 4),
        "mean_k": round(float(ex["k"].mean()), 3), "mask_density": round(float(ex["mask"].mean()), 4),
        "max_contact_err_m": check["contact_pos_err"], "seconds": round(time.time() - t0, 2),
    }


def _parse_counts(items) -> dict:
    counts = dict(DEFAULT_PER_HAND)
    for item in items or []:
        k, v = item.split("=")
        counts[k] = int(v)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="*", help="split names from data/splits.json (default: all)")
    ap.add_argument("--per-hand", nargs="*", metavar="KIND=N",
                    help="examples per hand by split kind: train, val, test (default: train=1000 val=1000 test=2000)")
    ap.add_argument("--regime", choices=sorted(REGIMES), default="kite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hands", default=str(config.DATA_DIR / "hands"))
    ap.add_argument("--out", default=None, help="default: data/examples/<regime>")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)

    splits = load_splits()
    names = args.splits or list(splits["hands"])
    counts = _parse_counts(args.per_hand)
    out = Path(args.out or EXAMPLES_DIR / args.regime)
    jobs = [(h, s, counts[split_kind(s)], args.seed, args.regime, args.hands, str(out)) for s in names for h in splits["hands"][s]]
    total = sum(j[2] for j in jobs)
    print(f"generating {total:,} examples for {len(jobs)} hands ({', '.join(names)}) with {args.workers} workers")
    t0, rows = time.time(), []
    with mp.Pool(args.workers) as pool:
        for row in pool.imap_unordered(generate_one, jobs, chunksize=1):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(jobs):
                print(f"  {len(rows)}/{len(jobs)} hands  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["split"], r["hand_id"]))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_split = {}
    for s in names:
        r = [x for x in rows if x["split"] == s]
        by_split[s] = {"hands": len(r), "examples": sum(x["n"] for x in r),
                       "mask_density": round(float(np.mean([x["mask_density"] for x in r])), 4),
                       "max_contact_err_m": max(x["max_contact_err_m"] for x in r)}
    summary = {"kite_data": __version__, "regime": args.regime, "sampler": REGIMES[args.regime].to_dict(),
               "seed": args.seed, "per_hand": counts, "elapsed_s": round(time.time() - t0, 1), "splits": by_split}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["splits"], indent=1))


if __name__ == "__main__":
    main()
