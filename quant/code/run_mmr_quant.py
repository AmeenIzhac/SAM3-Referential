#!/usr/bin/env python3
"""Run a (optionally weight-quantized) SAM3 checkpoint over MMR val, union gIoU/cIoU.

Protocol is byte-for-byte the one mmrcomp/code/run_sam3_mmr.py used for the
numbers in mmr.md: raw MMR question as the text prompt, threshold 0.5, union of
masks above threshold with best-mask fallback, scored by mmr_common's UnionMeter
against the same GT decode. Only the model construction differs.

  --bits k            quantize on the fly from the fp32 checkpoint (RTN, group-wise)
  --quant-ckpt f.pt   load a packed artifact written by quantize_ckpt.py
  --weight-dtype      plain cast baseline (bf16 / fp16), no integer quantization
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/reasonseg/mmrcomp/code")
sys.path.insert(0, "/workspace/reasonseg/quant/code")
sys.path.insert(0, "/workspace/sam3")

import mmr_common as C                              # noqa: E402
import sam3_quant_model as Q                        # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402


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
    return masks[int(scores.argmax())]              # best-mask fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=Q.MMR_CKPT)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--include-embeddings", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=[],
                    help="regexes; quantize ONLY matching params (subsystem probe)")
    ap.add_argument("--weight-dtype", default="fp32", choices=list(Q.DTYPES))
    ap.add_argument("--quant-ckpt", default=None)
    ap.add_argument("--quant-mode", default="dense", choices=["dense", "packed"])
    ap.add_argument("--head-bits", type=int, default=None,
                    help="hold the detection heads at this width (mixed precision)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, info = Q.build(args.checkpoint, args.device, args.bits, args.group_size,
                          args.include_embeddings, tuple(args.skip), args.weight_dtype,
                          args.quant_ckpt, args.quant_mode, tuple(args.only),
                          args.head_bits)
    proc = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)

    records = C.load_val()
    sharded = C.shard_records(records, args.shard, args.nshards)
    if args.limit:
        sharded = sharded[: args.limit]

    union = C.UnionMeter()
    by_gran = {g: C.UnionMeter() for g in ("obj", "part", "obj&part")}
    by_T = {"single": C.UnionMeter(), "multi": C.UnionMeter()}
    rows = []
    t0 = time.time()
    nq = 0
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
                pred_union = predict_union(proc, state, question.strip(), ori, args.threshold)
                gran = C.granularity(names)
                union.add(pred_union, gt_union)
                by_gran[gran].add(pred_union, gt_union)
                by_T["single" if len(names) == 1 else "multi"].add(pred_union, gt_union)
                inter, uni = C.fg_inter_union(pred_union, gt_union)
                rows.append({"image_id": rec["image_id"], "qi": qi, "T": len(names),
                             "gran": gran, "union_iou": union.ious[-1],
                             "inter": inter, "union": uni})
                nq += 1
            if ri % 100 == 0:
                el = time.time() - t0
                print(f"[{args.tag} s{args.shard}] {ri}/{len(sharded)} imgs, {nq} q, "
                      f"{el:.0f}s ({nq/el:.2f} q/s) | {union.result()}", flush=True)

    res = {"tag": args.tag, "build_info": info, "threshold": args.threshold,
           "overall": union.result(),
           "by_gran": {g: m.result() for g, m in by_gran.items()},
           "by_targets": {k: m.result() for k, m in by_T.items()},
           "inter_sum": union.inter_sum, "union_sum": union.union_sum,
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": res, "rows": rows}, open(args.out, "w"), indent=1)
    print("SUMMARY " + json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
