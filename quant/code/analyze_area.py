#!/usr/bin/env python3
"""How does quantization actually fail -- by under-firing or by over-firing?

The eval rows carry per-question `inter` and `union` counts, and GT area is a
property of the benchmark (identical across every config), so

    pred_area = union + inter - gt_area

recovers the predicted mask area for every config already evaluated, with no
extra inference. That separates the two failure modes the README cares about:
losing masks (recall) vs asserting mask everywhere (precision).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/reasonseg/mmrcomp/code")
import mmr_common as C  # noqa: E402

EVAL = "/workspace/reasonseg/quant/eval"
SHARDS = "/mnt/data0/ameen/quant_out"      # per-shard files, the ones with rows
CACHE = "/mnt/data0/ameen/quant_out/gt_areas.json"
ORDER = ["fp32", "bf16", "int8_g128", "int6_g128", "int5_g128", "int4_g128",
         "int3_g128", "int2_g128",
         "int3_vision_only", "int3_language_only", "int3_heads_only",
         "int3_g128_head8", "int4_g128_head8",
         "int8_pc", "int5_pc", "int4_pc", "int4_g64", "int4_g32",
         "int3_g64", "int3_g32", "int4_g128_noemb",
         "int4_g128_bf16rest", "int5_g128_bf16rest", "int4_from_ckpt", "int3_g64_head8"]
LABEL = {"fp32": "fp32 (baseline)", "bf16": "bf16 cast", "int8_g128": "int8 g128",
         "int6_g128": "int6 g128", "int5_g128": "int5 g128", "int4_g128": "int4 g128",
         "int3_g128": "int3 g128", "int2_g128": "int2 g128",
         "int3_vision_only": "int3, vision backbone only",
         "int3_language_only": "int3, language backbone only",
         "int3_heads_only": "int3, detection heads only",
         "int3_g128_head8": "int3 g128 + int8 heads",
         "int4_g128_head8": "int4 g128 + int8 heads",
         "int3_g64_head8": "int3 g64 + int8 heads",
         "int8_pc": "int8 per-channel", "int5_pc": "int5 per-channel",
         "int4_pc": "int4 per-channel", "int4_g64": "int4 g64", "int4_g32": "int4 g32",
         "int3_g64": "int3 g64", "int3_g32": "int3 g32",
         "int4_g128_noemb": "int4 g128, no emb",
         "int4_g128_bf16rest": "int4 g128, bf16 remainder",
         "int5_g128_bf16rest": "int5 g128, bf16 remainder",
         "int4_from_ckpt": "int4 g128, from the artifact"}


def gt_areas():
    if os.path.exists(CACHE):
        return {k: v for k, v in json.load(open(CACHE)).items()}
    out = {}
    for rec in C.load_val():
        for qi in range(len(rec["questions"])):
            try:
                _, gt, _ = C.decode_gt(rec, qi)
            except Exception:
                continue
            out[f"{rec['image_id']}:{qi}"] = int(gt.sum())
    json.dump(out, open(CACHE, "w"))
    return out


def main():
    ga = gt_areas()
    print(f"GT areas for {len(ga)} (image, question) pairs", flush=True)
    # the merged summaries carry no rows; re-read the shards that produced them
    rows = {}
    for f in sorted(glob.glob(os.path.join(EVAL, "mmr_*.json"))):
        tag = os.path.basename(f)[len("mmr_"):-len(".json")]
        rs = []
        for sf in sorted(glob.glob(os.path.join(SHARDS, f"mmr_{tag}_s*.json"))):
            rs += json.load(open(sf))["rows"]
        if rs:
            rows[tag] = rs
    tags = [t for t in ORDER if t in rows] + sorted(t for t in rows if t not in ORDER)

    print("\n| weights | mean pred area / GT area | empty preds | pred>2x GT | mean IoU |")
    print("|---|--:|--:|--:|--:|")
    for t in tags:
        pa, gt = [], []
        for r in rows[t]:
            g = ga.get(f"{r['image_id']}:{r['qi']}")
            if g is None:
                continue
            pa.append(r["union"] + r["inter"] - g)
            gt.append(g)
        pa, gt = np.array(pa, float), np.array(gt, float)
        ratio = pa.sum() / gt.sum()
        empty = float((pa <= 0).mean())
        over = float((pa > 2 * gt).mean())
        iou = float(np.mean([r["union_iou"] for r in rows[t]]))
        print(f"| {LABEL.get(t, t)} | {ratio:.3f} | {100*empty:.1f} % | "
              f"{100*over:.1f} % | {100*iou:.2f} |")


if __name__ == "__main__":
    main()
