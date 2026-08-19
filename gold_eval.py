#!/usr/bin/env python3
"""SA-Co/Gold benchmark eval (paper protocol): run a checkpoint over every (image, phrase)
pair of a gold subset, write COCO predictions, score with the repo's CGF1Evaluator against
the a/b/c triple annotations (oracle selection, instance-exhaustive queries only)."""
import argparse, collections, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

sys.path.insert(0, "/workspace/sam3")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.eval.cgf1_eval import CGF1Evaluator

GOLD = Path("/mnt/data0/ameen/saco_gold")
ANN = GOLD / "gts"
# 6 of the 7 subsets draw on MetaCLIP images; only the sa1b captioner subset uses SA-1B
IMG_ROOT = {"sa1b": GOLD / "images/all/sa1b-images"}
DEFAULT_IMG_ROOT = GOLD / "images/all/metaclip-images"
SUBSETS = ("sa1b", "metaclip", "attributes", "crowded", "fg_food",
           "fg_sports_equipment", "wiki_common")


def build_model(checkpoint, device):
    """Load a trainer checkpoint (plain model state dict under 'model', no 'detector.'
    prefix — so model_builder's own _load_checkpoint would silently match nothing)."""
    model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False,
                                   device="cpu", enable_inst_interactivity=False)
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        sd = {k.replace("detector.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"state dict mismatch: {len(missing)=} {len(unexpected)=}\n"
                             f"{missing[:5]=}\n{unexpected[:5]=}")
        print(f"loaded {len(sd)} tensors from {checkpoint}", flush=True)
    else:
        print("no --checkpoint: using stock facebook/sam3 release weights", flush=True)
        from sam3.model_builder import download_ckpt_from_hf, _load_checkpoint
        _load_checkpoint(model, download_ckpt_from_hf(version="sam3"))
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="omit for stock facebook/sam3")
    ap.add_argument("--subset", default="sa1b", choices=SUBSETS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="predictions json path")
    ap.add_argument("--limit-images", type=int, default=None)
    # CGF1Evaluator builds CGF1Eval without passing `threshold`, so it always scores at that
    # class's default of 0.5 — anything below is discarded before the metric sees it. Predicting
    # at 0.45 is therefore provably identical to the 0.05 the SAM3 configs use, and for the mmr
    # checkpoint (90 masks/pair at 0.05, 3.8% of which survive 0.45) it is ~25x less encoding.
    # Keep 0.05 if you want to be able to re-score BELOW 0.5 later (see gold_threshold_sweep.py).
    ap.add_argument("--conf", type=float, default=0.05)
    # shard a single subset across GPUs: each shard predicts on its own slice of pictures and
    # skips scoring; --merge-shards then concatenates the slices and scores the whole subset
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-shards", type=int, default=0, metavar="N",
                    help="skip inference: concatenate <out>.shard{0..N-1} and score")
    args = ap.parse_args()

    if args.merge_shards:
        preds = []
        for i in range(args.merge_shards):
            part = f"{args.out}.shard{i}"
            preds.extend(json.load(open(part)))
            print(f"  merged {part}", flush=True)
        json.dump(preds, open(args.out, "w"))
        print(f"merged {len(preds)} predictions -> {args.out}", flush=True)
        score(args, preds)
        return

    img_root = IMG_ROOT.get(args.subset, DEFAULT_IMG_ROOT)
    d = json.load(open(ANN / f"gold_{args.subset}_merged_a_release_test.json"))
    # group (image,phrase) pair rows by underlying picture so we encode each picture once
    by_file = collections.defaultdict(list)
    for im in d["images"]:
        by_file[im["file_name"]].append(im)
    files = sorted(by_file)
    if args.limit_images:
        files = files[: args.limit_images]
    if args.num_shards > 1:
        # stride, not contiguous blocks: pictures are sorted by name and cost varies with
        # pairs-per-picture, so striding keeps the shards balanced
        files = files[args.shard::args.num_shards]
    n_pairs = sum(len(by_file[f]) for f in files)
    print(f"[gold:{args.subset}] shard {args.shard}/{args.num_shards}: "
          f"{len(files)} pictures / {n_pairs} pairs", flush=True)

    model = build_model(args.checkpoint, args.device)
    proc = Sam3Processor(model, device=args.device, confidence_threshold=args.conf)

    preds, t0, done = [], time.time(), 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, fn in enumerate(files, 1):
            path = img_root / fn
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            state = proc.set_image(img)
            for pair in by_file[fn]:
                proc.reset_all_prompts(state)
                out = proc.set_text_prompt(state=state, prompt=pair["text_input"])
                masks_t, scores_t = out.get("masks"), out.get("scores")
                done += 1
                if masks_t is None or len(scores_t) == 0:
                    continue
                # threshold on-device and encode the whole stack in one call: both are
                # bit-identical to the per-mask CPU path but move 4x less over PCIe, which
                # matters for checkpoints that emit ~100 masks/pair (see mmr in the writeup)
                masks = (masks_t > 0.5).squeeze(1).detach().cpu().numpy()
                scores = scores_t.detach().float().cpu().numpy()
                stack = np.asfortranarray(
                    np.ascontiguousarray(masks.transpose(1, 2, 0)).astype(np.uint8))
                for rle, s in zip(mask_util.encode(stack), scores):
                    bb = [float(x) for x in mask_util.toBbox(rle)]
                    rle["counts"] = rle["counts"].decode("ascii")
                    # plain-list bbox up front: the repo's loadRes mutates shared preds with
                    # numpy bboxes on the 1st annotator, crashing the 2nd's `bbox == []` check
                    preds.append({"image_id": pair["id"], "category_id": 1,
                                  "segmentation": rle, "bbox": bb, "score": float(s)})
            if i % 100 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(files)} pictures, {done}/{n_pairs} pairs, "
                      f"{len(preds)} dets, {el/60:.1f} min elapsed, "
                      f"eta {(el/done*(n_pairs-done))/60:.1f} min", flush=True)

    out_path = args.out if args.num_shards == 1 else f"{args.out}.shard{args.shard}"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(out_path, "w"))
    print(f"wrote {len(preds)} predictions -> {out_path}", flush=True)

    if args.num_shards > 1:
        print("shard complete; run with --merge-shards to score", flush=True)
        return
    score(args, preds)


def score(args, preds):
    gts = [str(ANN / f"gold_{args.subset}_merged_{x}_release_test.json") for x in "abc"]
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
