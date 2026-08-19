#!/usr/bin/env python3
"""Inject the generated tables into quant.md's <!--RESULTS_*--> placeholders.

Keeps the prose in quant.md and the numbers in quant/eval, so a re-run
refreshes the document without hand-transcribing anything.
"""
import io
import os
import re
import subprocess
import sys

CODE = "/workspace/reasonseg/quant/code"
DOC = "/workspace/reasonseg/quant.md"
PY = sys.executable


def run(script, *args):
    out = subprocess.run([PY, os.path.join(CODE, script), *args],
                         capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"{script} failed:\n{out.stderr[-2000:]}")
    return out.stdout


def md_tables(text):
    """Split a stdout blob into its markdown tables, in order."""
    tables, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            tables.append("\n".join(cur)); cur = []
    if cur:
        tables.append("\n".join(cur))
    return tables


def rows_matching(table, tags):
    """Header rows plus the rows whose label is in `tags`, **in tags order**.

    Ordering by the caller's list rather than the source table's matters: the
    tables in the writeup are meant to be read down a single axis (ascending
    bits/weight within a bit width), which is not the order quant_table.py emits.
    """
    lines = table.splitlines()
    by_label = {}
    for ln in lines[2:]:
        by_label[ln.split("|")[1].strip().replace("**", "")] = ln
    missing = [t for t in tags if t not in by_label]
    if missing:
        print(f"  warn: no row for {missing}")
    return "\n".join(lines[:2] + [by_label[t] for t in tags if t in by_label])


MAIN_TAGS = ["fp32 (baseline)", "bf16 cast", "int8 g128", "int6 g128", "int5 g128",
             "int4 g128", "int3 g128", "int2 g128"]
ABL_TAGS = ["int8 per-channel", "int5 per-channel", "int5 g128", "int4 per-channel",
            "int4 g128", "int4 g64", "int4 g32", "int4 g128, no emb",
            "int3 g128", "int3 g64", "int3 g32",
            "int3, vision backbone only", "int3, language backbone only",
            "int3, detection heads only",
            "int4 g128, bf16 remainder", "int5 g128, bf16 remainder",
            "int4 g128, from the artifact"]
# ascending bits/weight within each bit width, so the two levers interleave
MIXED_TAGS = ["int3 g128", "int3 g128 + int8 heads", "int3 g64",
              "int3 g64 + int8 heads", "int3 g32",
              "int4 g128", "int4 g128 + int8 heads", "int4 g64", "int4 g32"]

main_t, detail_t = md_tables(run("quant_table.py"))[:2]
area_t = md_tables(run("analyze_area.py"))[0]
size_t = md_tables(run("size_table.py"))[0]
front_t = md_tables(run("frontier_table.py"))[0]
lat = run("latency_table.py") if os.path.exists(os.path.join(CODE, "latency_table.py")) else ""
lat_t = md_tables(lat)[0] if lat.strip() else "_(not measured)_"

doc = open(DOC).read()
repl = {
    "RESULTS_MAIN": rows_matching(main_t, MAIN_TAGS),
    "RESULTS_ABLATION": rows_matching(main_t, ABL_TAGS) + "\n\nPer-tensor fit quality and coverage for every row above:\n\n" + detail_t,
    "RESULTS_MIXED": rows_matching(main_t, MIXED_TAGS),
    "RESULTS_FRONTIER": front_t,
    "RESULTS_AREA": area_t,
    "RESULTS_SIZE": size_t,
    "RESULTS_LATENCY": lat_t,
}
for k, v in repl.items():
    block = f"<!--BEGIN:{k}-->\n{v}\n<!--END:{k}-->"
    if f"<!--BEGIN:{k}-->" in doc:                       # re-run: replace in place
        pre = doc[: doc.index(f"<!--BEGIN:{k}-->")]
        post = doc[doc.index(f"<!--END:{k}-->") + len(f"<!--END:{k}-->"):]
        doc = pre + block + post
    elif f"<!--{k}-->" in doc:                           # first run: fill the slot
        doc = doc.replace(f"<!--{k}-->", block)
    else:
        print(f"  warn: no slot for {k}")
open(DOC, "w").write(doc)
print(f"wrote {DOC} ({len(doc.splitlines())} lines)")
