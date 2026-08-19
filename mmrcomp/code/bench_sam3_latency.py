#!/usr/bin/env python3
"""Steady-state per-prediction latency for a SAM3 checkpoint on MMR val.

Fairness rules (all shared with bench_m2sa_latency.py):
  * model load / CUDA context / autotune are excluded via WARMUP iterations
  * every timed region is bracketed by torch.cuda.synchronize()
  * images are decoded from disk ONCE, up front, and held in RAM -> no disk I/O
    inside the timed region
  * we report median over many real MMR val (image, question) pairs, so the
    number is what you'd actually see amortized over a long run

SAM3 splits work into two stages that cost differently:
  set_image        -> vision backbone, once per IMAGE (reusable across prompts)
  set_text_prompt  -> text encoder + grounding decoder, once per PROMPT
so we time them separately and report both the cold single-prediction cost
(1 image + 1 prompt) and the amortized cost at MMR's real 3.41 questions/image.
"""
import argparse, json, os, statistics as st, sys, time

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


def sync():
    torch.cuda.synchronize()


def concept_phrase(name):
    return name.replace(":", " ").replace("_", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", choices=["base", "trained"], required=True)
    ap.add_argument("--prompt-mode", choices=["question", "concept"], default="question")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-images", type=int, default=40, help="images to preload/time")
    ap.add_argument("--n-prompts", type=int, default=200, help="timed text-prompt calls")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = BASE if args.ckpt == "base" else TRAINED
    model = build_model(ckpt, args.device).to(args.device)
    proc = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)

    # ---- preload real MMR val work items into RAM (no disk I/O while timing) ----
    records = C.load_val()
    items = []          # (PIL image, [prompt, ...])
    for rec in records:
        if len(items) >= args.n_images:
            break
        path = os.path.join(C.IMAGE_ROOT, rec["file_name"])
        try:
            img = Image.open(path).convert("RGB")
            img.load()
        except Exception:
            continue
        prompts = []
        for qi, question in enumerate(rec["questions"]):
            if args.prompt_mode == "question":
                prompts.append([question.strip()])
            else:
                try:
                    _, _, names = C.decode_gt(rec, qi)
                except Exception:
                    continue
                prompts.append([concept_phrase(n) for n in sorted(set(names))])
        if prompts:
            items.append((img, prompts))
    assert items, "no MMR val images loaded"
    n_q = sum(len(p) for _, p in items)
    print(f"preloaded {len(items)} images / {n_q} questions", flush=True)

    t_img, t_text, t_post, prompts_per_q = [], [], [], []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # ---------------- warmup (excluded) ----------------
        for i in range(args.warmup):
            img, prompts = items[i % len(items)]
            state = proc.set_image(img)
            proc.reset_all_prompts(state)
            out = proc.set_text_prompt(state=state, prompt=prompts[0][0])
            if out.get("masks") is not None and out["masks"].shape[0]:
                out["masks"].squeeze(1).detach().cpu().numpy()
        sync()

        # ---------------- timed ----------------
        done = 0
        img_i = 0
        while done < args.n_prompts:
            img, prompts = items[img_i % len(items)]
            img_i += 1

            sync(); t0 = time.perf_counter()
            state = proc.set_image(img)
            sync(); t_img.append(time.perf_counter() - t0)

            for plist in prompts:
                if done >= args.n_prompts:
                    break
                # one "prediction" = all text prompts this question needs
                sync(); t0 = time.perf_counter()
                masks_all = []
                for p in plist:
                    proc.reset_all_prompts(state)
                    out = proc.set_text_prompt(state=state, prompt=p)
                    masks_all.append(out)
                sync(); t_text.append(time.perf_counter() - t0)
                prompts_per_q.append(len(plist))

                # mask -> CPU/numpy union, reported separately
                sync(); t0 = time.perf_counter()
                for out in masks_all:
                    m = out.get("masks")
                    if m is not None and m.shape[0]:
                        mm = m.squeeze(1).detach().cpu().numpy().astype(bool)
                        sc = out["scores"].detach().float().cpu().numpy()
                        keep = sc >= args.threshold
                        _ = mm[keep].any(axis=0) if keep.any() else mm[int(sc.argmax())]
                sync(); t_post.append(time.perf_counter() - t0)
                done += 1

    def s(v):
        return {"n": len(v), "mean_ms": round(1000 * st.mean(v), 2),
                "median_ms": round(1000 * st.median(v), 2),
                "p90_ms": round(1000 * np.percentile(v, 90), 2),
                "min_ms": round(1000 * min(v), 2)}

    Q_PER_IMG = 8194 / 2404  # MMR val: real questions-per-image ratio
    img_med, text_med, post_med = st.median(t_img), st.median(t_text), st.median(t_post)
    res = {
        "model": f"sam3-{args.ckpt}",
        "prompt_mode": args.prompt_mode,
        "device": torch.cuda.get_device_name(args.device),
        "dtype": "bf16 autocast",
        "resolution": proc.resolution,
        "image_encode": s(t_img),
        "text_prompt_fwd": s(t_text),
        "mask_to_cpu": s(t_post),
        "avg_text_prompts_per_question": round(float(np.mean(prompts_per_q)), 3),
        "per_prediction_ms": {
            "cold_1img_1q_gpu": round(1000 * (img_med + text_med), 2),
            "cold_1img_1q_incl_post": round(1000 * (img_med + text_med + post_med), 2),
            "amortized_gpu": round(1000 * (img_med / Q_PER_IMG + text_med), 2),
            "amortized_incl_post": round(1000 * (img_med / Q_PER_IMG + text_med + post_med), 2),
            "q_per_img_used": round(Q_PER_IMG, 3),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
