#!/usr/bin/env python3
"""Convert the ENTIRE MMR train split into one SAM3-COCO json for the
data-scaling ladder (1/2/4/8/16/32/64/100 % of the data, one continuous run).

Each (image, question) sample -> one COCO category (name = the question) with one
annotation per target mask (multi-instance, object AND part granularity).

The sample ORDER is a deterministic shuffle with seed 0 -- identical to the older
4-chunk converter (convert_mmr_to_coco.py), so the first 8,000 samples here are
exactly the 8,000 used by the previous 1k/2k/4k/8k staged curve. That makes the
new ladder directly comparable to `mmrcomp/scaling_curve.md`.
"""
import json, os, random, sys, time

from pycocotools import mask as mask_util

TRAIN = "/mnt/data0/ameen/mmr_data/MMR_train.json"
OUTDIR = "/mnt/data0/ameen/mmr_data/coco"
OUT = f"{OUTDIR}/mmr_full.json"
os.makedirs(OUTDIR, exist_ok=True)

t0 = time.time()
d = json.load(open(TRAIN))
print(f"train records: {len(d)}  ({time.time()-t0:.0f}s)", flush=True)

pairs = [(ri, qi) for ri, r in enumerate(d) for qi in range(len(r["questions"]))]
random.seed(0)
random.shuffle(pairs)
print("total (image,question) samples:", len(pairs), flush=True)

images, categories, annotations = {}, [], []
cat_id = 0
ann_id = 0
missing_img = 0
IMG_ROOT = "/mnt/data0/ameen/mmr_data/images"

for n, (ri, qi) in enumerate(pairs, 1):
    r = d[ri]
    iid = r["image_id"]
    if iid not in images:
        if not os.path.exists(os.path.join(IMG_ROOT, r["file_name"])):
            missing_img += 1
            continue
        images[iid] = {"id": iid, "file_name": r["file_name"],
                       "width": r["width"], "height": r["height"]}
    cat_id += 1
    names = [a.get("category_name", "?") for a in r["answers"][qi]]
    categories.append({"id": cat_id, "name": r["questions"][qi].strip(),
                       "supercategory": names[0].split(":")[0] if names else "object",
                       "reasoning_type": "mmr", "kind": "multi"})
    for a in r["answers"][qi]:
        rle = {"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"]}
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode()
        bbox = [float(x) for x in mask_util.toBbox(rle)]
        area = float(mask_util.area(rle))
        seg = {"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"]}
        if isinstance(seg["counts"], bytes):
            seg["counts"] = seg["counts"].decode("ascii")
        ann_id += 1
        annotations.append({"id": ann_id, "image_id": iid, "category_id": cat_id,
                            "bbox": bbox, "area": area, "segmentation": seg, "iscrowd": 0})
    if n % 20000 == 0:
        print(f"  {n}/{len(pairs)}  cats={len(categories)} anns={len(annotations)} "
              f"imgs={len(images)}  ({time.time()-t0:.0f}s)", flush=True)

coco = {"images": list(images.values()), "categories": categories, "annotations": annotations}
json.dump(coco, open(OUT, "w"))
if not os.path.exists(f"{OUTDIR}/image_negatives.json"):
    json.dump({}, open(f"{OUTDIR}/image_negatives.json", "w"))

meta = {"n_samples": len(categories), "n_images": len(images), "n_masks": len(annotations),
        "skipped_missing_image": missing_img, "shuffle_seed": 0, "source": TRAIN}
json.dump(meta, open(f"{OUTDIR}/mmr_full_meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1), flush=True)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB, {time.time()-t0:.0f}s)", flush=True)
