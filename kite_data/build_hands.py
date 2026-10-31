"""Build the LEAP hand family.

    python -m kite_data.build_hands                      # all 243 topologies x (nominal + 6 draws)
    python -m kite_data.build_hands --topologies I4-M4-R4-T4 I3-M0-R2-T2 --draws 2

Writes data/hands/<hand_id>/{hand.json, hand.xml, hand.urdf, template.npz}, data/hands/index.csv and
data/hands/summary.json. Hand ids are <topology>_g<draw>; draw 0 is the nominal LEAP geometry.
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
from .collision import Collider, to_mjcf, validity_gate
from .geometry import sample_geometry
from .hand import Hand
from .leap import link_type, load_base_model
from .template import build_template, candidate_pool
from .topology import Topology, enumerate_topologies
from .urdf import to_urdf

MAX_ATTEMPTS = 20


def make_hand(topology_id: str, draw: int, seed: int, attempt: int) -> Hand:
    t = Topology.from_id(topology_id)
    return Hand(t, sample_geometry(t, draw, seed, attempt))


def build_one(job) -> dict:
    topology_id, draw, seed, out_root, gate_samples = job
    for attempt in range(MAX_ATTEMPTS):
        hand = make_hand(topology_id, draw, seed, attempt)
        xml = to_mjcf(hand)
        gate = validity_gate(hand, gate_samples, seed=zlib.crc32(hand.id.encode()), collider=Collider(hand, xml))
        if gate["ok"] or draw == 0:
            break
    out = Path(out_root) / hand.id
    out.mkdir(parents=True, exist_ok=True)
    desc = hand.descriptor()
    desc["build"] = {"seed": seed, "geometry_attempt": attempt, "gate": gate}
    (out / "hand.json").write_text(json.dumps(desc, indent=1) + "\n")
    (out / "hand.xml").write_text(xml)
    (out / "hand.urdf").write_text(to_urdf(hand, out))
    tpl = build_template(hand, seed)
    np.savez_compressed(out / "template.npz", **tpl)
    counts = np.bincount(tpl["region"], minlength=len(config.REGIONS))
    return {
        "id": hand.id, "topology": topology_id, "draw": draw, "n_dof": hand.n_dof,
        "fingers": "+".join(hand.topology.fingers), "has_thumb": hand.topology.has_thumb,
        "gate_ok": gate["ok"], "attempts": attempt + 1, "free_rate": gate["free_rate"],
        **{f"template_{r}": int(c) for r, c in zip(config.REGIONS, counts)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.DATA_DIR / "hands"))
    ap.add_argument("--draws", type=int, default=6, help="random geometry draws per topology, besides the nominal one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--topologies", nargs="*", help="topology ids such as I4-M3-R0-T4 (default: all)")
    ap.add_argument("--no-thumbless", action="store_true", help="skip hands without a thumb")
    ap.add_argument("--gate-samples", type=int, default=config.GATE_SAMPLES)
    args = ap.parse_args(argv)

    topologies = ([Topology.from_id(s) for s in args.topologies] if args.topologies
                  else enumerate_topologies(include_thumbless=not args.no_thumbless))
    base = load_base_model()
    for lt in sorted({link_type(l) for l in base.links}):  # fill the visibility cache before forking
        candidate_pool(lt)
    jobs = [(t.id, d, args.seed, args.out, args.gate_samples) for t in topologies for d in range(args.draws + 1)]
    print(f"building {len(jobs)} hands ({len(topologies)} topologies x {args.draws + 1} geometries) with {args.workers} workers")
    t0, rows = time.time(), []
    with mp.Pool(args.workers) as pool:
        for row in pool.imap_unordered(build_one, jobs, chunksize=2):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(jobs):
                print(f"  {len(rows)}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["topology"], r["draw"]))

    out = Path(args.out)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    free = np.array([r["free_rate"] for r in rows])
    summary = {
        "kite_data": __version__, "args": vars(args), "elapsed_s": round(time.time() - t0, 1),
        "n_hands": len(rows), "n_topologies": len(topologies),
        "gate_failures": [r["id"] for r in rows if not r["gate_ok"]],
        "redrawn": sum(r["attempts"] > 1 for r in rows),
        "free_rate": {"min": float(free.min()), "median": float(np.median(free)), "max": float(free.max())},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
