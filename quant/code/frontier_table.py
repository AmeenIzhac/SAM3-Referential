#!/usr/bin/env python3
"""Accuracy-per-stored-bit frontier over every uniform/mixed weight config.

The point of ranking by bits/weight rather than by nominal bit width: the levers
(more code bits / finer scale groups / higher-precision heads) all cost storage,
and they are only comparable once that cost is on the same axis.
"""
import glob
import json
import os

EVAL = "/workspace/reasonseg/quant/eval"
SKIP = ("fp32", "bf16", "_vision_only", "_language_only", "_heads_only",
        "_from_ckpt", "_bf16rest", "_noemb")
NAME = {"int8_pc": "int8 per-channel", "int5_pc": "int5 per-channel",
        "int4_pc": "int4 per-channel"}


def label(tag):
    if tag in NAME:
        return NAME[tag]
    p = tag.split("_")
    out = p[0] + (" " + p[1] if len(p) > 1 else "")
    return out + (" + int8 heads" if "head8" in tag else "")


rows = []
for f in glob.glob(os.path.join(EVAL, "mmr_*.json")):
    tag = os.path.basename(f)[len("mmr_"):-len(".json")]
    if any(s in tag for s in SKIP):
        continue
    d = json.load(open(f))
    i = d.get("build_info") or {}
    if not i.get("bits"):
        continue
    g = i.get("group_size", 128)
    gg = g if g > 0 else 1500                      # mean in_features, per-channel
    body = i["bits"] + 24.0 / gg
    nh, nt = i.get("head_quantized_params", 0), i.get("quantized_params", 1)
    bpw = body if not nh else ((nt - nh) * body + nh * (i["head_bits"] + 24.0 / gg)) / nt
    rows.append((bpw, label(tag), d["overall"]["gIoU"]))

rows.sort()
best = -1.0
print("| bits/weight | config | gIoU | on frontier? |")
print("|--:|---|--:|---|")
for bpw, lab, g in rows:
    on = g > best
    if on:
        best = g
    print(f"| {bpw:.2f} | {lab} | {g:.2f} | {'**yes**' if on else 'dominated' } |")
