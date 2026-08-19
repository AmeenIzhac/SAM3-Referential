#!/usr/bin/env python3
"""Per-prediction latency and resident VRAM for a quantized SAM3 checkpoint.

Same fairness rules as mmrcomp/code/bench_sam3_latency.py -- model load and
autotune excluded via warmup, torch.cuda.synchronize() around every timed
region, images decoded into RAM up front -- so the numbers are directly
comparable to the SAM3 rows in mmr.md §9.

VRAM is reported two ways:
  weights_MB : torch.cuda.memory_allocated() right after the model lands on the
               GPU and before any activation exists -- the number quantization
               is supposed to move
  peak_MB    : torch.cuda.max_memory_allocated() over the timed region, i.e.
               weights + activations at 1008x1008
"""
import argparse
import json
import os
import statistics as st
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


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", default=Q.MMR_CKPT)
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--include-embeddings", action="store_true")
    ap.add_argument("--weight-dtype", default="fp32", choices=list(Q.DTYPES))
    ap.add_argument("--quant-ckpt", default=None)
    ap.add_argument("--quant-mode", default="dense", choices=["dense", "packed"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--n-prompts", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    model, info = Q.build(args.checkpoint, args.device, args.bits, args.group_size,
                          args.include_embeddings, (), args.weight_dtype,
                          args.quant_ckpt, args.quant_mode)
    sync()
    weights_mb = torch.cuda.memory_allocated() / 1e6
    proc = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)

    records = C.load_val()
    items = []
    for rec in records:
        if len(items) >= args.n_images:
            break
        try:
            img = Image.open(os.path.join(C.IMAGE_ROOT, rec["file_name"])).convert("RGB")
            img.load()
        except Exception:
            continue
        qs = [q.strip() for q in rec["questions"]]
        if qs:
            items.append((img, qs))
    assert items
    print(f"preloaded {len(items)} images / {sum(len(q) for _, q in items)} questions", flush=True)

    t_img, t_text = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(args.warmup):
            img, qs = items[i % len(items)]
            state = proc.set_image(img)
            proc.reset_all_prompts(state)
            proc.set_text_prompt(state=state, prompt=qs[0])
        sync()
        torch.cuda.reset_peak_memory_stats()

        done, img_i = 0, 0
        while done < args.n_prompts:
            img, qs = items[img_i % len(items)]
            img_i += 1
            sync(); t0 = time.perf_counter()
            state = proc.set_image(img)
            sync(); t_img.append(time.perf_counter() - t0)
            for q in qs:
                if done >= args.n_prompts:
                    break
                sync(); t0 = time.perf_counter()
                proc.reset_all_prompts(state)
                proc.set_text_prompt(state=state, prompt=q)
                sync(); t_text.append(time.perf_counter() - t0)
                done += 1
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    def s(v):
        return {"n": len(v), "mean_ms": round(1000 * st.mean(v), 2),
                "median_ms": round(1000 * st.median(v), 2),
                "p90_ms": round(1000 * float(np.percentile(v, 90)), 2),
                "min_ms": round(1000 * min(v), 2)}

    Q_PER_IMG = 8194 / 2404
    im, tm = st.median(t_img), st.median(t_text)
    res = {"tag": args.tag, "build_info": info,
           "device": torch.cuda.get_device_name(args.device),
           "dtype": "bf16 autocast", "resolution": proc.resolution,
           "vram": {"weights_MB": round(weights_mb, 1), "peak_MB": round(peak_mb, 1)},
           "image_encode": s(t_img), "text_prompt_fwd": s(t_text),
           "per_prediction_ms": {"cold_1img_1q": round(1000 * (im + tm), 2),
                                 "amortized": round(1000 * (im / Q_PER_IMG + tm), 2),
                                 "q_per_img_used": round(Q_PER_IMG, 3)}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "build_info"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
