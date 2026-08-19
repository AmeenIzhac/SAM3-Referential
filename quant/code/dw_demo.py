#!/usr/bin/env python3
"""Segment the dw/ images with base SAM3 vs the fine-tuned checkpoints, and
draw the predicted masks as red outlines.

For each image we run three things, because two of them are only interpretable
next to the third:
  base   + simple prompt   ("green box")                    <- what stock SAM3 does
  base   + complex prompt  ("green box in the left container") <- the control: can
                            stock SAM3 already resolve the referring expression?
  tuned  + complex prompt                                    <- the fine-tune
Without the control you cannot tell whether a change came from the model or
from the prompt.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/reasonseg/quant/code")
sys.path.insert(0, "/workspace/sam3")
import sam3_quant_model as Q                                   # noqa: E402
from sam3.model_builder import build_sam3_image_model          # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor      # noqa: E402

DW = "/workspace/reasonseg/dw"
OUT = "/workspace/reasonseg/dw-out"
MMR = Q.MMR_CKPT
REASONSEG = "/mnt/data0/ameen/reasonseg_runs/sam3-reasonseg-mixed/checkpoints/checkpoint.pt"
BASE = Q.BASE_CKPT

# (image, simple prompt, complex prompt) -- assigned by image content, see README note
JOBS = [
    ("r1", "green box", "green box in the left container"),
    ("r2", "vase", "vase in front of the radio"),
    ("r3", "plate", "the small clean plate with nothing on it"),
]


def build(kind, device="cuda:0"):
    """base = stock release checkpoint (has a 'detector.' prefix and extra video
    tensors); the fine-tunes are trainer checkpoints and load through the same
    path the evals use."""
    if kind == "base":
        m = build_sam3_image_model(checkpoint_path=None, load_from_HF=False,
                                   device="cpu", enable_inst_interactivity=False)
        ck = torch.load(BASE, map_location="cpu", mmap=True, weights_only=True)
        sd = {k[len("detector."):]: v for k, v in ck.items() if k.startswith("detector.")}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing, f"missing {len(missing)}: {missing[:5]}"
        print(f"[base] loaded {len(sd)} tensors ({len(unexpected)} video/tracker tensors ignored)",
              flush=True)
        return m.to(device).eval()
    path = MMR if kind == "mmr" else REASONSEG
    m, _ = Q.build(checkpoint=path, device=device, verbose=False)
    print(f"[{kind}] loaded {path}", flush=True)
    return m


def predict(proc, state, prompt, thr):
    proc.reset_all_prompts(state)
    out = proc.set_text_prompt(state=state, prompt=prompt)
    m = out.get("masks")
    if m is None or m.shape[0] == 0:
        return np.zeros((0, 0), bool), np.array([]), 0
    masks = (m > 0.5).squeeze(1).detach().cpu().numpy().astype(bool)
    scores = out["scores"].detach().float().cpu().numpy()
    keep = scores >= thr
    return masks, scores, int(keep.sum())


def draw(img_bgr, masks, scores, thr, thickness=3):
    """Red outline per kept mask. Falls back to nothing if none pass."""
    vis = img_bgr.copy()
    keep = np.where(scores >= thr)[0]
    for i in keep:
        m = masks[i].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), thickness, lineType=cv2.LINE_AA)
    return vis, len(keep)


def banner(img_bgr, lines, pad=10):
    """White caption strip under the image, black text -- same idea as the
    preview captions in generate_dataset.py."""
    h, w = img_bgr.shape[:2]
    fs = max(0.5, w / 1400)
    th = max(1, int(round(fs * 2)))
    hs = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[1] + 12 for t in lines]
    strip = np.full((sum(hs) + 2 * pad, w, 3), 255, np.uint8)
    y = pad
    for t, hh in zip(lines, hs):
        y += hh
        cv2.putText(strip, t, (pad, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    th, cv2.LINE_AA)
    return np.vstack([img_bgr, strip])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", default="mmr", choices=["mmr", "reasonseg"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # (tag, model kind, which prompt)
    RUNS = [("base_simple", "base", "simple"),
            ("base_complex", "base", "complex"),
            (f"{args.tuned}_complex", args.tuned, "complex")]

    imgs = {t: Image.open(os.path.join(DW, f"{t}.jpeg")).convert("RGB") for t, _, _ in JOBS}
    log = []
    for tag, kind, which in RUNS:
        model = build(kind, args.device)
        proc = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, simple, complex_ in JOBS:
                prompt = simple if which == "simple" else complex_
                img = imgs[name]
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                state = proc.set_image(img)
                masks, scores, n_keep = predict(proc, state, prompt, args.threshold)
                vis, n = (bgr.copy(), 0) if masks.size == 0 else draw(bgr, masks, scores, args.threshold)
                model_label = {"base": "SAM3 (stock, not fine-tuned)",
                               "mmr": "SAM3 fine-tuned on MMR",
                               "reasonseg": "SAM3 fine-tuned on SA-1B reasonseg"}[kind]
                top = f'{n} mask(s) @ score>={args.threshold}'
                if len(scores):
                    top += f'   top scores: {", ".join(f"{s:.2f}" for s in sorted(scores)[::-1][:4])}'
                vis = banner(vis, [f'{model_label}', f'prompt: "{prompt}"', top])
                path = os.path.join(OUT, f"{name}_{tag}.jpg")
                cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
                log.append({"image": name, "run": tag, "model": kind, "prompt": prompt,
                            "n_masks_total": int(len(scores)), "n_kept": n,
                            "scores": [round(float(s), 4) for s in scores[:8]],
                            "out": path})
                print(f"  {name} [{tag}] '{prompt}' -> {n} kept of {len(scores)}", flush=True)
        del model, proc
        torch.cuda.empty_cache()

    json.dump(log, open(os.path.join(OUT, f"results_{args.tuned}.json"), "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
