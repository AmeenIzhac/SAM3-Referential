#!/usr/bin/env python3
"""Convert MMR train (image,question) samples into SAM3-COCO chunk files for
incremental training. Each sample -> one COCO category (name=question) with one
annotation per target mask (multi-instance). 4 incremental chunks: 1000/1000/2000/4000
(cumulative unique 1000/2000/4000/8000)."""
import json, os, random
import numpy as np
from pycocotools import mask as mask_util

TRAIN = "/mnt/data0/ameen/mmr_data/MMR_train.json"
OUTDIR = "/mnt/data0/ameen/mmr_data/coco"
os.makedirs(OUTDIR, exist_ok=True)
BOUNDS = [0, 1000, 2000, 4000, 8000]          # chunk edges
N = BOUNDS[-1]

d = json.load(open(TRAIN))
print("train records:", len(d))

# all (record_idx, qi) question pairs, deterministic shuffle
pairs = [(ri, qi) for ri, r in enumerate(d) for qi in range(len(r["questions"]))]
random.seed(0)
random.shuffle(pairs)
pairs = pairs[:N]
print("selected samples:", len(pairs))


def rle_bytes(seg):
    r = {"size": seg["size"], "counts": seg["counts"]}
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode()
    return r


needed_images = {}   # image_id -> (file_name, coco_url)
cat_id = 0
ann_id = 0
for c, (lo, hi) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:]), 1):
    images, categories, annotations = {}, [], []
    for (ri, qi) in pairs[lo:hi]:
        r = d[ri]
        iid = r["image_id"]
        needed_images[iid] = (r["file_name"], r.get("coco_url"))
        if iid not in images:
            images[iid] = {"id": iid, "file_name": r["file_name"],
                           "width": r["width"], "height": r["height"]}
        cat_id += 1
        names = [a.get("category_name", "?") for a in r["answers"][qi]]
        categories.append({"id": cat_id, "name": r["questions"][qi].strip(),
                           "supercategory": names[0].split(":")[0] if names else "object",
                           "reasoning_type": "mmr", "kind": "multi"})
        for a in r["answers"][qi]:
            rle = rle_bytes(a["segmentation"])
            bbox = [float(x) for x in mask_util.toBbox(rle)]
            area = float(mask_util.area(rle))
            seg = {"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"]}
            if isinstance(seg["counts"], bytes):
                seg["counts"] = seg["counts"].decode("ascii")
            ann_id += 1
            annotations.append({"id": ann_id, "image_id": iid, "category_id": cat_id,
                                "bbox": bbox, "area": area, "segmentation": seg, "iscrowd": 0})
    coco = {"images": list(images.values()), "categories": categories, "annotations": annotations}
    out = f"{OUTDIR}/mmr_chunk{c}_{lo}-{hi}.json"
    json.dump(coco, open(out, "w"))
    print(f"chunk{c} [{lo}:{hi}]: {len(images)} imgs, {len(categories)} samples, {len(annotations)} masks -> {out}")

# empty negatives sidecar (include_negatives=false, but path is referenced)
json.dump({}, open(f"{OUTDIR}/image_negatives.json", "w"))
# needed images list
json.dump(needed_images, open(f"{OUTDIR}/needed_images.json", "w"))
print(f"unique images needed: {len(needed_images)}")
