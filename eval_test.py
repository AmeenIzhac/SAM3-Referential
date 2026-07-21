#!/usr/bin/env python3
"""Evaluate a SAM3 checkpoint on the held-out dataset_mixed test split.

Per sample (= one COCO category = one prompt on one image): text-prompt the model,
union the predicted masks (score >= threshold, best-mask fallback), and IoU against the
union of GT instance masks. Reports overall / per-kind / per-reasoning-category means.
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

sys.path.insert(0, "/workspace/newsam/sam3")          # model code the ckpts train with
sys.path.insert(0, "/workspace/newsam")               # benchmark_refexp.build_model
from benchmark_refexp import build_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_DIR = Path("/mnt/data0/ameen/reasonseg_data/images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ann-file", default="/workspace/reasonseg/out/dataset_mixed/test/_annotations.coco.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    d = json.load(open(args.ann_file))
    imgs = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c for c in d["categories"]}
    by_cat = collections.defaultdict(list)
    for a in d["annotations"]:
        by_cat[a["category_id"]].append(a)

    model = build_model(args.checkpoint, args.device)
    model = model.to(args.device)   # builder only moves on exact device=="cuda"
    proc = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)

    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, (cid, anns) in enumerate(sorted(by_cat.items()), 1):
            cat = cats[cid]
            im = imgs[anns[0]["image_id"]]
            gt = None
            for a in anns:
                r = dict(a["segmentation"]); r["counts"] = r["counts"].encode() \
                    if isinstance(r["counts"], str) else r["counts"]
                m = mask_util.decode(r).astype(bool)
                gt = m if gt is None else (gt | m)
            try:
                img = Image.open(IMAGE_DIR / im["file_name"]).convert("RGB")
            except Exception:
                continue
            state = proc.set_image(img)
            out = proc.set_text_prompt(state=state, prompt=cat["name"])
            if out["masks"].shape[0] == 0:
                pred = np.zeros_like(gt)
            else:
                masks = out["masks"].squeeze(1).detach().cpu().numpy().astype(bool)
                scores = out["scores"].detach().float().cpu().numpy()
                keep = scores >= args.threshold
                pred = masks[keep].any(axis=0) if keep.any() else masks[int(scores.argmax())]
            inter = (pred & gt).sum(); union = (pred | gt).sum()
            iou = float(inter / union) if union else 1.0
            rows.append({"prompt": cat["name"], "kind": cat.get("kind", "complex"),
                         "reasoning_type": cat.get("reasoning_type", "?"), "iou": iou})
            if i % 200 == 0:
                print(f"  {i}/{len(by_cat)}", flush=True)

    def summ(rs):
        v = [r["iou"] for r in rs]
        return {"n": len(v), "mean_iou": round(float(np.mean(v)), 4) if v else 0,
                "iou50": round(float(np.mean([x >= .5 for x in v])), 4) if v else 0}
    result = {"overall": summ(rows),
              "by_kind": {k: summ([r for r in rows if r["kind"] == k])
                          for k in sorted({r["kind"] for r in rows})},
              "by_category": {k: summ([r for r in rows if r["reasoning_type"] == k])
                              for k in sorted({r["reasoning_type"] for r in rows})}}
    json.dump({"summary": result, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
