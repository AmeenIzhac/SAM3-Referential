#!/usr/bin/env python3
"""SA-Co/Gold benchmark eval (paper protocol): run a checkpoint over every (image, phrase)
pair of a gold subset, write COCO predictions, score with the repo's CGF1Evaluator against
the a/b/c triple annotations (oracle selection, instance-exhaustive queries only)."""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

sys.path.insert(0, "/workspace/newsam/sam3")
sys.path.insert(0, "/workspace/newsam")
from benchmark_refexp import build_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.eval.cgf1_eval import CGF1Evaluator

GOLD = Path("/workspace/newsam/SA-Co-Gold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subset", default="sa1b")
    ap.add_argument("--img-sub", default=None, help="images/<sub> dir (default = subset)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="predictions json path")
    ap.add_argument("--limit-images", type=int, default=None)
    args = ap.parse_args()
    img_sub = args.img_sub or args.subset

    ann_a = GOLD / "annotations" / f"gold_{args.subset}_merged_a_release_test.json"
    d = json.load(open(ann_a))
    # group (image,phrase) pair rows by underlying picture so we encode each picture once
    by_file = collections.defaultdict(list)
    for im in d["images"]:
        by_file[im["file_name"]].append(im)
    files = sorted(by_file)
    if args.limit_images:
        files = files[: args.limit_images]
    print(f"[gold:{args.subset}] {len(files)} pictures / "
          f"{sum(len(by_file[f]) for f in files)} pairs", flush=True)

    model = build_model(args.checkpoint, args.device)
    model = model.to(args.device)   # builder only moves on exact device=="cuda"
    proc = Sam3Processor(model, device=args.device, confidence_threshold=0.05)

    preds = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, fn in enumerate(files, 1):
            path = GOLD / "images" / img_sub / fn
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            state = proc.set_image(img)
            for pair in by_file[fn]:
                proc.reset_all_prompts(state)
                out = proc.set_text_prompt(state=state, prompt=pair["text_input"])
                masks_t, scores_t = out.get("masks"), out.get("scores")
                if masks_t is None or len(scores_t) == 0:
                    continue
                masks = masks_t.squeeze(1).detach().cpu().numpy() > 0.5
                scores = scores_t.detach().float().cpu().numpy()
                for m, s in zip(masks, scores):
                    rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
                    bb = [float(x) for x in mask_util.toBbox(rle)]
                    rle["counts"] = rle["counts"].decode("ascii")
                    # plain-list bbox up front: the repo's loadRes mutates shared preds with
                    # numpy bboxes on the 1st annotator, crashing the 2nd's `bbox == []` check
                    preds.append({"image_id": pair["id"], "category_id": 1,
                                  "segmentation": rle, "bbox": bb, "score": float(s)})
            if i % 100 == 0:
                print(f"  {i}/{len(files)} pictures, {len(preds)} dets", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(args.out, "w"))
    print(f"wrote {len(preds)} predictions -> {args.out}", flush=True)

    gts = [str(GOLD / "annotations" / f"gold_{args.subset}_merged_{x}_release_test.json")
           for x in ("a", "b", "c")]
    gts = [g for g in gts if Path(g).exists()]
    print(f"scoring against {len(gts)} annotator gts (oracle)...", flush=True)
    ev = CGF1Evaluator(gt_path=gts, iou_type="segm", verbose=False)
    if args.limit_images:   # smoke runs: only score pairs we actually predicted on
        pred_ids = {p["image_id"] for p in preds}
        ev.eval_img_ids = [i for i in ev.eval_img_ids if i in pred_ids]
        print(f"[smoke] scoring restricted to {len(ev.eval_img_ids)} predicted pairs", flush=True)
    out = ev.evaluate(args.out)
    json.dump(out, open(args.out.replace(".json", "_metrics.json"), "w"), indent=1)
    key = {k: round(v, 4) for k, v in out.items()
           if any(t in k for t in ("_cgF1", "IL_MCC", "IL_F1")) and "@" not in k}
    print("KEY METRICS:", json.dumps(key, indent=1), flush=True)


if __name__ == "__main__":
    main()
