#!/usr/bin/env python3
"""Stitch the per-run dw-out panels into one side-by-side sheet per image."""
import os

import cv2
import numpy as np

OUT = "/workspace/reasonseg/dw-out"
COLS = ["base_simple", "base_complex", "mmr_complex", "reasonseg_complex"]
IMGS = ["r1", "r2", "r3"]


def load(name, col):
    p = os.path.join(OUT, f"{name}_{col}.jpg")
    return cv2.imread(p) if os.path.exists(p) else None


for name in IMGS:
    panels = [p for p in (load(name, c) for c in COLS) if p is not None]
    if not panels:
        continue
    h = max(p.shape[0] for p in panels)
    # pad to equal height, then hstack with a thin separator
    fixed = []
    for p in panels:
        if p.shape[0] < h:
            p = np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)])
        fixed.append(p)
        fixed.append(np.full((h, 6, 3), 210, np.uint8))
    sheet = np.hstack(fixed[:-1])
    # keep the sheet a sane width for viewing
    if sheet.shape[1] > 3600:
        s = 3600 / sheet.shape[1]
        sheet = cv2.resize(sheet, (3600, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    path = os.path.join(OUT, f"{name}_comparison.jpg")
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {path}  {sheet.shape[1]}x{sheet.shape[0]}")
