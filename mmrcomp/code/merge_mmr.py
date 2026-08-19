#!/usr/bin/env python3
"""Merge sharded MMR-eval outputs into final gIoU/cIoU (overall, by granularity,
single vs multi-target, and M2SA native per-target if present)."""
import glob, json, sys
import numpy as np


def merge(files, label):
    rows = []
    inter_sum = union_sum = 0
    n_inter = n_union = n_acc = n_tgt = 0.0
    have_native = False
    for f in files:
        d = json.load(open(f))
        rows += d["rows"]
        s = d["summary"]
        inter_sum += s.get("inter_sum", 0)
        union_sum += s.get("union_sum", 0)
        if "native_targets" in s and s["native_targets"]:
            have_native = True
            n_inter += s["native_inter"]; n_union += s["native_union"]
            n_acc += s["native_acc"]; n_tgt += s["native_targets"]

    def um(rs):
        if not rs:
            return {"n": 0, "gIoU": 0.0, "cIoU": 0.0, "iou50": 0.0}
        iou = np.array([r["union_iou"] for r in rs])
        # cIoU needs cumulative inter/union; only have it globally -> recompute per-subset unavailable.
        return {"n": len(rs), "gIoU": round(100 * iou.mean(), 2),
                "iou50": round(100 * (iou >= 0.5).mean(), 2)}

    overall = um(rows)
    overall["cIoU"] = round(100 * inter_sum / union_sum, 2) if union_sum else 0.0
    out = {"label": label, "overall": overall,
           "by_gran": {g: um([r for r in rows if r["gran"] == g]) for g in ("obj", "part", "obj&part")},
           "single_target": um([r for r in rows if r["T"] == 1]),
           "multi_target": um([r for r in rows if r["T"] > 1])}
    if have_native:
        out["native_per_target"] = {"targets": int(n_tgt),
                                    "gIoU": round(100 * n_acc / n_tgt, 2),
                                    "cIoU": round(100 * n_inter / n_union, 2)}
    return out


if __name__ == "__main__":
    label = sys.argv[1]
    files = []
    for pat in sys.argv[2:]:
        files += glob.glob(pat)
    files = sorted(files)
    print(f"[{label}] merging {len(files)} shards: {[f.split('/')[-1] for f in files]}")
    res = merge(files, label)
    print(json.dumps(res, indent=1))
    outp = f"/workspace/reasonseg/mmr_eval/{label}.json"
    json.dump(res, open(outp, "w"), indent=1)
    print("wrote", outp)
