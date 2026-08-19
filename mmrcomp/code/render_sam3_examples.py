#!/usr/bin/env python3
"""Compute SAM3 (base & fine-tuned) masks for the selected examples; save to npz.
Run with the sam3 env python."""
import json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad")
sys.path.insert(0, "/workspace/newsam/sam3"); sys.path.insert(0, "/workspace/newsam")
import mmr_common as C
from benchmark_refexp import build_model
from sam3.model.sam3_image_processor import Sam3Processor

BASE = "/workspace/.cache/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
TRAINED = "/mnt/data0/ameen/reasonseg_runs/sam3-reasonseg-mixed/checkpoints/checkpoint.pt"
THR = 0.5
exs = json.load(open("/workspace/reasonseg/mmrcomp/examples.json"))
records = {r["image_id"]: r for r in C.load_val()}


def cphrase(n): return n.replace(":", " ").replace("_", " ").strip()

def predict_union(proc, state, prompt, ori):
    proc.reset_all_prompts(state)
    out = proc.set_text_prompt(state=state, prompt=prompt)
    m = out.get("masks")
    if m is None or m.shape[0] == 0:
        return np.zeros(ori, bool)
    masks = m.squeeze(1).detach().cpu().numpy().astype(bool)
    sc = out["scores"].detach().float().cpu().numpy()
    keep = sc >= THR
    return masks[keep].any(0) if keep.any() else masks[int(sc.argmax())]

store = {}
for tag, ckpt in [("base", BASE), ("trained", TRAINED)]:
    model = build_model(ckpt, "cuda:0").to("cuda:0")
    proc = Sam3Processor(model, device="cuda:0", confidence_threshold=THR)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, e in enumerate(exs):
            rec = records[e["image_id"]]
            ori = (rec["height"], rec["width"])
            img = Image.open(os.path.join(C.IMAGE_ROOT, rec["file_name"])).convert("RGB")
            state = proc.set_image(img)
            store[f"{i}_sam3_{tag}_q"] = predict_union(proc, state, e["question"].strip(), ori).astype(np.uint8)
            u = np.zeros(ori, bool)
            for cn in sorted(set(e["cat_names"])):
                u |= predict_union(proc, state, cphrase(cn), ori)
            store[f"{i}_sam3_{tag}_c"] = u.astype(np.uint8)
    del model, proc; torch.cuda.empty_cache()
    print(f"done {tag}", flush=True)

# GT unions too
for i, e in enumerate(exs):
    _, gt, _ = C.decode_gt(records[e["image_id"]], e["qi"])
    store[f"{i}_gt"] = gt.astype(np.uint8)
np.savez_compressed("/workspace/reasonseg/mmrcomp/_masks_sam3.npz", **store)
print("saved", len(store), "masks -> _masks_sam3.npz")
