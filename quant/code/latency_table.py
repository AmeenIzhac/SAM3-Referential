#!/usr/bin/env python3
"""Latency + VRAM table from quant/eval/lat_*.json."""
import glob
import json
import os

EVAL = "/workspace/reasonseg/quant/eval"
ORDER = ["fp32", "int8_dense", "int6_dense", "int5_dense", "int4_dense",
         "int4_g32_dense", "int3_dense", "int2_dense", "int4_from_ckpt_dense",
         "int8_packed", "int6_packed", "int5_packed", "int4_packed", "int3_packed",
         "int2_packed", "int4_from_ckpt_packed"]
LABEL = {"fp32": "fp32 (baseline)",
         "int8_dense": "int8 g128, dense", "int6_dense": "int6 g128, dense",
         "int5_dense": "int5 g128, dense", "int4_dense": "int4 g128, dense",
         "int4_g32_dense": "int4 g32, dense", "int3_dense": "int3 g128, dense",
         "int2_dense": "int2 g128, dense",
         "int4_from_ckpt_dense": "int4 artifact, dense",
         "int8_packed": "int8 g128, **packed**", "int6_packed": "int6 g128, **packed**",
         "int5_packed": "int5 g128, **packed**", "int4_packed": "int4 g128, **packed**",
         "int3_packed": "int3 g128, **packed**", "int2_packed": "int2 g128, **packed**",
         "int4_from_ckpt_packed": "int4 artifact, **packed**"}

res = {}
for f in glob.glob(os.path.join(EVAL, "lat_*.json")):
    res[os.path.basename(f)[4:-5]] = json.load(open(f))
if not res:
    raise SystemExit(0)
tags = [t for t in ORDER if t in res] + sorted(t for t in res if t not in ORDER)
print("| runtime | image encode (mean) | per-prompt fwd (mean) | full fwd, 1 img + 1 prompt | "
      "amortized / prediction | weights VRAM | peak VRAM |")
print("|---|--:|--:|--:|--:|--:|--:|")
Q_PER_IMG = 8194 / 2404
for t in tags:
    r = res[t]
    v = r["vram"]
    ie, tp = r["image_encode"]["mean_ms"], r["text_prompt_fwd"]["mean_ms"]
    print(f"| {LABEL.get(t, t)} | {ie:.1f} ms | {tp:.1f} ms | {ie + tp:.1f} ms | "
          f"**{ie / Q_PER_IMG + tp:.1f} ms** | {v['weights_MB']:.0f} MB | {v['peak_MB']:.0f} MB |")
