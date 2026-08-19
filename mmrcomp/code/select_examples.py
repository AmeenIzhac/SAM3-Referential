#!/usr/bin/env python3
"""Pick instructive MMR-val examples by joining the per-question IoU rows of all
runs, so the rendered panels tell the story (reasoning gap, fine-tune effect, etc.)."""
import glob, json, collections
import sys
sys.path.insert(0, "/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad")
import mmr_common as C

OUT = "/mnt/data0/ameen/mmr_out"
RUNS = ["m2sa_teacher", "sam3_base_question", "sam3_trained_question",
        "sam3_base_concept", "sam3_trained_concept"]

def load_rows(run):
    d = {}
    for f in glob.glob(f"{OUT}/{run}_s*.json"):
        for r in json.load(open(f))["rows"]:
            d[(r["image_id"], r["qi"])] = r
    return d

data = {run: load_rows(run) for run in RUNS}
records = {r["image_id"]: r for r in C.load_val()}

keys = set(data["m2sa_teacher"])
for run in RUNS:
    keys &= set(data[run])
print("joined keys:", len(keys))

def iou(run, k):
    return data[run][k]["union_iou"]

rows = []
for k in keys:
    base = data["m2sa_teacher"][k]
    rows.append({
        "key": k, "T": base["T"], "gran": base["gran"],
        "m2sa": iou("m2sa_teacher", k),
        "sam3_base_q": iou("sam3_base_question", k),
        "sam3_tr_q": iou("sam3_trained_question", k),
        "sam3_base_c": iou("sam3_base_concept", k),
        "sam3_tr_c": iou("sam3_trained_concept", k),
    })

picked = []
used_imgs = set()
def take(cands, note, n=1):
    for r in cands:
        iid = r["key"][0]
        if iid in used_imgs:
            continue
        r2 = dict(r); r2["note"] = note
        picked.append(r2); used_imgs.add(iid)
        if sum(1 for p in picked if p["note"] == note) >= n:
            break

# 1) reasoning gap on OBJECTS: M2SA good, SAM3-trained-Q fails, SAM3 oracle nails it
take(sorted([r for r in rows if r["gran"]=="obj" and r["m2sa"]>.6 and r["sam3_tr_q"]<.15 and r["sam3_base_c"]>.7],
            key=lambda r:-(r["sam3_base_c"]+r["m2sa"])), "reasoning_gap_obj", 2)
# 2) fine-tune HELPS the question: trained-Q >> base-Q
take(sorted([r for r in rows if r["sam3_tr_q"]>.45 and r["sam3_tr_q"]-r["sam3_base_q"]>.4],
            key=lambda r:-(r["sam3_tr_q"]-r["sam3_base_q"])), "finetune_helps_q", 1)
# 3) SAM3 oracle BEATS M2SA (mask quality > reasoning model)
take(sorted([r for r in rows if r["gran"]=="obj" and r["sam3_base_c"]-r["m2sa"]>.3 and r["sam3_base_c"]>.7],
            key=lambda r:-(r["sam3_base_c"]-r["m2sa"])), "sam3_oracle_beats_m2sa", 1)
# 4) multi-target union
take(sorted([r for r in rows if r["T"]>=3 and r["m2sa"]>.4 and r["sam3_base_c"]>.4],
            key=lambda r:-r["T"]), "multi_target", 1)
# 5) part-level: everyone struggles more (honest)
take(sorted([r for r in rows if r["gran"]=="part" and r["m2sa"]>.4],
            key=lambda r:-r["sam3_base_c"]), "part_level", 1)
# 6) hard for all
take(sorted([r for r in rows if r["m2sa"]<.3 and r["sam3_tr_q"]<.1 and r["sam3_base_c"]<.4],
            key=lambda r:r["m2sa"]+r["sam3_base_c"]), "hard_for_all", 1)

# attach question + cat names, save
exs = []
for p in picked:
    iid, qi = p["key"]
    rec = records[iid]
    _, _, names = C.decode_gt(rec, qi)
    exs.append({"image_id": iid, "file_name": rec["file_name"], "qi": qi,
                "question": rec["questions"][qi], "cat_names": names,
                "T": p["T"], "gran": p["gran"], "note": p["note"],
                "ious": {k: round(p[k],3) for k in ("m2sa","sam3_base_q","sam3_tr_q","sam3_base_c","sam3_tr_c")}})
json.dump(exs, open("/workspace/reasonseg/mmrcomp/examples.json","w"), indent=1)
print(f"picked {len(exs)} examples:")
for e in exs:
    print(f"  [{e['note']:24s}] img {e['image_id']} q{e['qi']} T={e['T']} {e['gran']:8s} "
          f"m2sa={e['ious']['m2sa']} tr_q={e['ious']['sam3_tr_q']} base_c={e['ious']['sam3_base_c']} "
          f":: {e['question'][:60]}")
