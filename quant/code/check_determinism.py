#!/usr/bin/env python3
"""Is a repeated eval of the same config bit-identical?

Everything in the path is deterministic in principle -- RTN has no randomness,
the val order is fixed, no sampling anywhere -- so any gIoU difference between
two configs should be a real difference and not run noise. That is worth
checking rather than assuming, because it decides whether a 0.05 gIoU gap in
the results table means anything at all.
"""
import json
import sys

a, b = sys.argv[1], sys.argv[2]
da, db = json.load(open(a)), json.load(open(b))
ra = {(r["image_id"], r["qi"]): r for r in da["rows"]}
rb = {(r["image_id"], r["qi"]): r for r in db["rows"]}
print(f"rows: {len(ra)} vs {len(rb)}, same keys: {set(ra) == set(rb)}")
diff = [k for k in ra if ra[k]["inter"] != rb[k]["inter"] or ra[k]["union"] != rb[k]["union"]]
sa, sb = da["summary"]["overall"], db["summary"]["overall"]
print(f"summary A: {sa}")
print(f"summary B: {sb}")
print(f"per-question mask differences: {len(diff)} / {len(ra)}")
print("VERDICT:", "bit-identical" if not diff and sa == sb else f"differs on {len(diff)} questions")
