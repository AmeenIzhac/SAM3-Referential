#!/usr/bin/env python3
"""Run a SAM3 checkpoint over MMR val and score union gIoU/cIoU.

 prompt-mode=question : feed the raw MMR reasoning question as the text prompt
                        (the honest head-to-head vs M2SA-gen; SAM3 is a concept
                        segmenter so this is out-of-distribution for it).
 prompt-mode=concept  : ORACLE diagnostic -- feed the GT target category name(s)
                        (part 'a:b' -> 'a b', '_' -> ' '), union the results.
                        Removes the language-reasoning step; measures mask ceiling.

Run with the sam3 env python.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/reasonseg/mmrcomp/code")
sys.path.insert(0, "/workspace/newsam/sam3")
sys.path.insert(0, "/workspace/newsam")
import mmr_common as C
from benchmark_refexp import build_model
from sam3.model.sam3_image_processor import Sam3Processor

BASE = "/workspace/.cache/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
TRAINED = "/mnt/data0/ameen/reasonseg_runs/sam3-reasonseg-mixed/checkpoints/checkpoint.pt"


def concept_phrase(name):
    return name.replace(":", " ").replace("_", " ").strip()


def predict_union(proc, state, prompt, ori, thr):
    proc.reset_all_prompts(state)
    out = proc.set_text_prompt(state=state, prompt=prompt)
    m = out.get("masks")
    if m is None or m.shape[0] == 0:
        return np.zeros(ori, bool)
    masks = m.squeeze(1).detach().cpu().numpy().astype(bool)
    scores = out["scores"].detach().float().cpu().numpy()
    keep = scores >= thr
    if keep.any():
        return masks[keep].any(axis=0)
    return masks[int(scores.argmax())]  # best-mask fallback (matches eval_test.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="custom", help="base|trained|custom (with --ckpt-path)")
    ap.add_argument("--ckpt-path", default=None, help="explicit checkpoint path (overrides --ckpt)")
    ap.add_argument("--prompt-mode", choices=["question", "concept"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = args.ckpt_path or (BASE if args.ckpt == "base" else TRAINED)
    model = build_model(ckpt, args.device).to(args.device)
    proc = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)

    records = C.load_val()
    sharded = C.shard_records(records, args.shard, args.nshards)
    if args.limit:
        sharded = sharded[: args.limit]

    union = C.UnionMeter()
    by_gran = {g: C.UnionMeter() for g in ("obj", "part", "obj&part")}
    rows = []
    t0 = time.time(); nq = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ri, (rec_idx, rec) in enumerate(sharded, 1):
            path = os.path.join(C.IMAGE_ROOT, rec["file_name"])
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            ori = (rec["height"], rec["width"])
            state = proc.set_image(img)
            for qi, question in enumerate(rec["questions"]):
                try:
                    per_target, gt_union, names = C.decode_gt(rec, qi)
                except Exception:
                    continue
                if args.prompt_mode == "question":
                    pred_union = predict_union(proc, state, question.strip(), ori, args.threshold)
                else:
                    pred_union = np.zeros(ori, bool)
                    for cn in sorted(set(names)):
                        pred_union |= predict_union(proc, state, concept_phrase(cn), ori, args.threshold)
                gran = C.granularity(names)
                union.add(pred_union, gt_union)
                by_gran[gran].add(pred_union, gt_union)
                rows.append({"image_id": rec["image_id"], "qi": qi, "T": len(names),
                             "gran": gran, "union_iou": union.ious[-1]})
                nq += 1
            if ri % 100 == 0:
                el = time.time() - t0
                print(f"[sam3-{args.ckpt}-{args.prompt_mode} s{args.shard}] {ri}/{len(sharded)} imgs, "
                      f"{nq} q, {el:.0f}s ({nq/el:.2f} q/s) | {union.result()}", flush=True)

    res = {"ckpt": args.ckpt, "prompt_mode": args.prompt_mode, "threshold": args.threshold,
           "overall": union.result(),
           "by_gran": {g: m.result() for g, m in by_gran.items()},
           "inter_sum": union.inter_sum, "union_sum": union.union_sum}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": res, "rows": rows}, open(args.out, "w"), indent=1)
    print("SUMMARY", json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
