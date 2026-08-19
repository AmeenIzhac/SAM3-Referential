#!/usr/bin/env python3
"""On-disk size of every artifact, against the real fp32 checkpoint file."""
import glob
import json
import os

import torch

CK = "/mnt/data0/ameen/quant_ckpts"
FP32 = "/mnt/data0/ameen/mmr_runs/mmr_scale/checkpoints/checkpoint.pt"
EVAL = "/workspace/reasonseg/quant/eval"

base = os.path.getsize(FP32)
print(f"fp32 trainer checkpoint: {base/1e6:.1f} MB\n")
print("| artifact | bits | group | codes | scales+zeros | fp32 remainder | file | vs fp32 | gIoU |")
print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
rows = []
for f in sorted(glob.glob(os.path.join(CK, "mmr_sam3_*.pt"))):
    tag = os.path.basename(f)[len("mmr_sam3_"):-len(".pt")]
    cfg = torch.load(f, map_location="cpu", weights_only=False)["config"]
    n_groups = sum(info["n_codes"] // (cfg["group_size"] if cfg["group_size"] > 0
                                       else info["shape"][1])
                   for info in cfg["quantized"].values())
    meta = n_groups * 3                      # bf16 scale (2B) + uint8 zero (1B)
    codes = cfg["bytes_quantized"] - meta
    size = os.path.getsize(f)
    ev = os.path.join(EVAL, f"mmr_{tag}.json")
    giou = json.load(open(ev))["overall"]["gIoU"] if os.path.exists(ev) else None
    rows.append((cfg["bits"], cfg["group_size"], tag, codes, meta,
                 cfg["bytes_dense"], size, base / size, giou))
for bits, g, tag, codes, meta, dense, size, x, giou in sorted(rows, key=lambda r: -r[0]):
    print(f"| `mmr_sam3_{tag}.pt` | {bits} | {g} | {codes/1e6:.0f} MB | {meta/1e6:.0f} MB | "
          f"{dense/1e6:.0f} MB | **{size/1e6:.0f} MB** | **{x:.2f}×** | "
          f"{f'{giou:.2f}' if giou else '—'} |")
