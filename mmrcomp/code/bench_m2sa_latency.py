#!/usr/bin/env python3
"""Steady-state per-prediction latency for M2SA-7B on MMR val.

Same fairness rules as bench_sam3_latency.py:
  * model load / CUDA context / autotune excluded via WARMUP iterations
  * every timed region bracketed by torch.cuda.synchronize()
  * images decoded from disk ONCE up front, held in RAM
  * median over many real MMR val (image, question) pairs

Unlike SAM3, M2SA recomputes the vision tower inside every forward, so there is
no per-image encoding to amortize: one (image, question) pair = one full forward.
  mode=teacher : single forward, GT [SEG] count teacher-forced (MMR official)
  mode=gen     : question-only autoregressive generation (the honest setting)
CPU-side preprocessing (CLIP preprocess + SAM resize/normalize) is timed
separately so it can be included or excluded from the comparison.

Run with the m2sa conda env python, cwd = the MMR repo.
"""
import argparse, json, os, statistics as st, sys, time

import numpy as np
import torch

sys.path.insert(0, "/workspace/reasonseg/mmrcomp/code")
sys.path.insert(0, "/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad/MMR")
import mmr_common as C

from transformers import AutoTokenizer, CLIPImageProcessor
from model.M2SA import M2SAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

MODEL = "/mnt/data0/ameen/models/M2SA-7B"
IMAGE_TOKEN = "<image>"
IM_START, IM_END = "<im_start>", "<im_end>"
PIX_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
PIX_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
IMG_SIZE = 1024


def sync():
    torch.cuda.synchronize()


def sam_preprocess(x):
    x = (x - PIX_MEAN) / PIX_STD
    h, w = x.shape[-2:]
    return torch.nn.functional.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))


def build(device, dtype):
    tok = AutoTokenizer.from_pretrained(MODEL, model_max_length=2048,
                                        padding_side="right", use_fast=False)
    tok.pad_token = tok.unk_token
    tok.add_tokens("[SEG]")
    seg_idx = tok("[SEG]", add_special_tokens=False).input_ids[0]
    tok.add_tokens([IM_START, IM_END], special_tokens=True)
    model = M2SAForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, low_cpu_mem_usage=True,
        seg_token_idx=seg_idx, vision_pretrained=None, out_dim=256,
        train_mask_decoder=True, vision_tower="openai/clip-vit-large-patch14",
        use_mm_start_end=True,
    )
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=dtype, device=device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model = model.to(device).eval()
    assert seg_idx == 32003, seg_idx
    return tok, model


def with_imtoken(prompt):
    return prompt.replace(IMAGE_TOKEN, IM_START + IMAGE_TOKEN + IM_END)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "gen"], default="teacher")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--n-preds", type=int, default=100, help="timed forward passes")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    transform = ResizeLongestSide(IMG_SIZE)
    tok, model = build(args.device, dtype)
    import cv2

    # ---- preload real MMR val work items (no disk I/O while timing) ----
    records = C.load_val()
    items = []   # dict(img_sam, image_clip, ori, resize, questions=[(q, per_target, text_answer)])
    t_prep = []
    for rec in records:
        if len(items) >= args.n_images:
            break
        path = os.path.join(C.IMAGE_ROOT, rec["file_name"])
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ori = image.shape[:2]

        t0 = time.perf_counter()
        image_clip = clip_proc.preprocess(image, return_tensors="pt")["pixel_values"][0]
        img_sam = transform.apply_image(image)
        resize = img_sam.shape[:2]
        img_sam = sam_preprocess(torch.from_numpy(img_sam).permute(2, 0, 1).contiguous())
        t_prep.append(time.perf_counter() - t0)   # CPU preprocessing, per image

        img_sam = img_sam.to(dtype).to(args.device).unsqueeze(0)
        image_clip = image_clip.to(dtype).to(args.device).unsqueeze(0)

        qs = []
        for qi, question in enumerate(rec["questions"]):
            try:
                per_target, gt_union, names = C.decode_gt(rec, qi)
            except Exception:
                continue
            ta = rec["text_answers"][qi]
            if args.mode == "teacher" and ta.count("{seg}") != per_target.shape[0]:
                continue
            qs.append((question.strip(), per_target, ta))
        if qs:
            items.append({"img_sam": img_sam, "image_clip": image_clip, "ori": ori,
                          "resize": resize, "qs": qs})
    assert items, "no MMR val images loaded"
    print(f"preloaded {len(items)} images / {sum(len(i['qs']) for i in items)} questions", flush=True)

    def build_inputs(it, q, per_target, text_answer):
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0],
                            IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(q))
        if args.mode == "teacher":
            conv.append_message(conv.roles[1], "{}.".format(text_answer.format(seg="[SEG]")))
        else:
            conv.append_message(conv.roles[1], None)
        prompt = with_imtoken(conv.get_prompt())
        return tokenizer_image_token(prompt, tok, return_tensors="pt").unsqueeze(0).to(args.device)

    def run_one(it, q, per_target, text_answer, input_ids):
        if args.mode == "teacher":
            attn = input_ids.ne(tok.pad_token_id)
            with torch.no_grad():
                out = model(images=it["img_sam"], images_clip=it["image_clip"],
                            input_ids=input_ids, labels=None, attention_masks=attn,
                            offset=torch.LongTensor([0, 1]).to(args.device),
                            masks_list=[torch.from_numpy(per_target.astype(np.float32)).to(args.device)],
                            label_list=[torch.zeros(it["ori"], device=args.device)],
                            resize_list=[it["resize"]], inference=True)
            return out["pred_masks"][0], input_ids.shape[1]
        with torch.no_grad():
            out_ids, pred_masks = model.evaluate(
                it["image_clip"], it["img_sam"], input_ids,
                resize_list=[it["resize"]], original_size_list=[it["ori"]],
                max_new_tokens=256, tokenizer=tok)
        return pred_masks[0], int(out_ids.shape[1])

    # ---------------- warmup (excluded) ----------------
    for i in range(args.warmup):
        it = items[i % len(items)]
        q, pt, ta = it["qs"][0]
        pm, _ = run_one(it, q, pt, ta, build_inputs(it, q, pt, ta))
        if pm is not None and pm.shape[0]:
            (pm[0] > 0).detach().cpu().numpy()
    sync()

    # ---------------- timed ----------------
    t_fwd, t_post, ntok = [], [], []
    done = 0
    ii = 0
    while done < args.n_preds:
        it = items[ii % len(items)]
        ii += 1
        for (q, pt, ta) in it["qs"]:
            if done >= args.n_preds:
                break
            input_ids = build_inputs(it, q, pt, ta)   # tokenization: not timed (trivial CPU)
            sync(); t0 = time.perf_counter()
            pm, ntk = run_one(it, q, pt, ta, input_ids)
            sync(); t_fwd.append(time.perf_counter() - t0)
            ntok.append(ntk)

            sync(); t0 = time.perf_counter()
            union = np.zeros(it["ori"], bool)
            for k in range(pm.shape[0]):
                union |= (pm[k] > 0).detach().cpu().numpy()
            sync(); t_post.append(time.perf_counter() - t0)
            done += 1

    def s(v):
        return {"n": len(v), "mean_ms": round(1000 * st.mean(v), 2),
                "median_ms": round(1000 * st.median(v), 2),
                "p90_ms": round(1000 * np.percentile(v, 90), 2),
                "min_ms": round(1000 * min(v), 2)}

    fwd_med, post_med = st.median(t_fwd), st.median(t_post)
    res = {
        "model": "m2sa-7b",
        "mode": args.mode,
        "device": torch.cuda.get_device_name(args.device),
        "dtype": args.dtype,
        "forward": s(t_fwd),
        "mask_to_cpu": s(t_post),
        "cpu_preprocess_per_image": s(t_prep),
        "tokens": {"mean": round(float(np.mean(ntok)), 1), "median": float(np.median(ntok))},
        "per_prediction_ms": {
            "gpu": round(1000 * fwd_med, 2),
            "incl_post": round(1000 * (fwd_med + post_med), 2),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
