"""Reproduce the Round 4 live campaign's selections offline.

The agents' confirmation rerun could not start (platform session limit), so
this script replays the same selection procedure against the stored campaign:
same pre-filter (OptB88vdW gap 0.5-1.5 eV), same strategy (warm-started
surrogate, explore 0.5, leakage guard on), batches of 10.

    .venv/bin/python scripts/reproduce_round4.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labmcp import engine  # noqa: E402


def main() -> None:
    live = json.loads((ROOT / "runs" / "campaigns" / "round4_gap05_15.json").read_text())
    live_batches = [b["jids"] for b in live["batches"]]
    cand = engine.candidates({"gap_min": 0.5, "gap_max": 1.5})
    print(f"pool size after gap filter: {len(cand)}")

    measured: list = []
    for i, live_jids in enumerate(live_batches, 1):
        rng = np.random.default_rng(len(measured))  # same seeding as select_candidates
        picks = engine.pick_surrogate(cand, measured, len(live_jids), rng, explore=0.5, warm_start=True, leakage_guard=True)
        jids = list(cand.loc[picks, "jid"])
        print(f"batch {i}: identical order {jids == live_jids}, overlap {len(set(jids) & set(live_jids))}/{len(live_jids)}")
        measured += picks

    hits = engine.is_hit(cand.loc[measured], live["min_slme"])
    live_hits = engine.is_hit(engine.load_pool().set_index("jid").loc[live["measured"]], live["min_slme"])
    print(f"hits: reproduced {int(hits.sum())}/{len(measured)}, live {int(live_hits.sum())}/{len(live['measured'])}")


if __name__ == "__main__":
    main()
