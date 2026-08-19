#!/usr/bin/env python3
"""Merge sharded run_mmr_quant.py outputs. Same meters as merge_mmr.py, plus
exact per-subset cIoU (the rows carry their own inter/union counts)."""
import glob
import json
import sys

import numpy as np


def um(rs):
    if not rs:
        return {"n": 0, "gIoU": 0.0, "cIoU": 0.0, "iou50": 0.0}
    iou = np.array([r["union_iou"] for r in rs])
    i = sum(r.get("inter", 0) for r in rs)
    u = sum(r.get("union", 0) for r in rs)
    return {"n": len(rs), "gIoU": round(100 * float(iou.mean()), 2),
            "cIoU": round(100 * i / u, 2) if u else 0.0,
            "iou50": round(100 * float((iou >= 0.5).mean()), 2)}


def merge(files, label):
    rows, info, secs = [], None, 0.0
    for f in files:
        d = json.load(open(f))
        rows += d["rows"]
        info = info or d["summary"].get("build_info")
        secs += d["summary"].get("seconds", 0.0)
    return {"label": label, "build_info": info, "shard_seconds_sum": round(secs, 1),
            "n_shards": len(files), "overall": um(rows),
            "by_gran": {g: um([r for r in rows if r["gran"] == g])
                        for g in ("obj", "part", "obj&part")},
            "single_target": um([r for r in rows if r["T"] == 1]),
            "multi_target": um([r for r in rows if r["T"] > 1])}


if __name__ == "__main__":
    label = sys.argv[1]
    files = sorted(f for pat in sys.argv[2:] for f in glob.glob(pat))
    print(f"[{label}] merging {len(files)} shards: {[f.split('/')[-1] for f in files]}")
    res = merge(files, label)
    print(json.dumps({k: v for k, v in res.items() if k != "build_info"}, indent=1))
    outp = f"/workspace/reasonseg/quant/eval/{label}.json"
    json.dump(res, open(outp, "w"), indent=1)
    print("wrote", outp)
