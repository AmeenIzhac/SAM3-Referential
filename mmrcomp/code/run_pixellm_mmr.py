#!/usr/bin/env python3
"""Run PixelLM-7B over MMR val and score union gIoU/cIoU.

FREE GENERATION ONLY, by design. PixelLM does not use a single `[SEG]` token: it
emits a codebook of `seg_token_num(3) x image_feature_scale_num(2) = 6` tokens
`[SEG0]..[SEG5]` per target. MMR's `text_answers` carry exactly one `{seg}`
placeholder per target, so the teacher-forced protocol used for LISA/M2SA cannot
be mapped onto PixelLM without inventing a 1->6 expansion. Free generation needs
no such choice -- the model writes its own answer and emits its own masks -- and
it is the honest, no-answer-leak setting that is the primary data point for every
other model in this comparison.

The prompt is byte-identical to the one given to LISA and M2SA:
    "{question} Please output segmentation mask."
which is also PixelLM's own LONG_QUESTION_LIST template, so no model is being
prompted out of its native convention.

The image pipeline mirrors PixelLM's chat.py exactly (it differs from LISA/M2SA):
a 448 CLIP branch with pad_train_clip_images + resize_vision_tower, and the usual
1024 SAM branch.

Must run with the m2sa conda env python (same py3.8/torch2.1/tf4.31 stack),
cwd = the PixelLM repo.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/tmp/claude-1009/-workspace-reasonseg/d6cd4b24-35e9-4fa3-8cad-354194dce19a/scratchpad/PixelLM"
sys.path.insert(0, "/workspace/reasonseg/mmrcomp/code")
sys.path.insert(0, REPO)
import mmr_common as C

from transformers import AutoTokenizer, CLIPImageProcessor
from model.PixelLM import PixelLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

MODEL = "/mnt/data0/ameen/models/PixelLM-7B/hf_model"
PREPROCESSOR = f"{REPO}/configs/preprocessor_448.json"
# Settings for the released checkpoint, taken verbatim from the repo's README
# inference command (they are NOT the argparse defaults).
SEG_TOKEN_NUM = 3
SCALE_NUM = 2
VISION_TOWER = "openai/clip-vit-large-patch14-336"
RESIZE_VT_SIZE = 448
IMAGE_TOKEN = "<image>"
IM_START, IM_END = "<im_start>", "<im_end>"
IMG_SIZE = 1024
PIX_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
PIX_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


def preprocess(x, img_size=IMG_SIZE):
    """Normalize and pad to square -- chat.py's helper."""
    x = (x - PIX_MEAN) / PIX_STD
    h, w = x.shape[-2:]
    return F.pad(x, (0, img_size - w, 0, img_size - h))


def build(device, dtype):
    tok = AutoTokenizer.from_pretrained(
        MODEL, model_max_length=512, padding_side="right", use_fast=False
    )
    tok.pad_token = tok.unk_token
    new_tokens = [f"[SEG{i}]" for i in range(SEG_TOKEN_NUM * SCALE_NUM)]
    tok.add_tokens(new_tokens)
    seg_idx = [tok(t, add_special_tokens=False).input_ids[0] for t in new_tokens]

    model = PixelLMForCausalLM.from_pretrained(
        MODEL, low_cpu_mem_usage=True, vision_tower=VISION_TOWER,
        seg_token_idx=seg_idx, torch_dtype=dtype,
        seg_token_num=SEG_TOKEN_NUM, image_feature_scale_num=SCALE_NUM,
        pad_train_clip_images=True, resize_vision_tower=True,
        resize_vision_tower_size=RESIZE_VT_SIZE, vision_tower_for_mask=True,
        separate_mm_projector=True,
    )
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=dtype)
    model = model.bfloat16().cuda() if dtype == torch.bfloat16 else model.cuda()
    model.get_model().get_vision_tower().to(device=device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model.eval()
    print(f"[pixellm] vocab={len(tok)} seg_token_idx={seg_idx}", flush=True)
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue from a partial --out file instead of restarting")
    ap.add_argument("--save-every", type=int, default=100,
                    help="flush partial results every N questions")
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.half, "fp32": torch.float32}[args.dtype]

    clip_proc = CLIPImageProcessor.from_pretrained(PREPROCESSOR)
    transform = ResizeLongestSide(IMG_SIZE)
    transform_clip = ResizeLongestSide(clip_proc.size["shortest_edge"])
    tok, model = build(args.device, dtype)
    import cv2

    records = C.load_val()
    sharded = C.shard_records(records, args.shard, args.nshards)
    if args.limit:
        sharded = sharded[: args.limit]

    union = C.UnionMeter()
    rows = []
    done_keys = set()
    # This box has other jobs on it; a 5 h run that dies to someone else's memory
    # spike should not start from zero. Partial results are flushed periodically
    # and --resume skips whatever is already scored.
    if args.resume and os.path.exists(args.out):
        prev = json.load(open(args.out))
        rows = prev.get("rows", [])
        union.ious = [r["union_iou"] for r in rows]
        union.inter_sum = prev["summary"].get("inter_sum", 0)
        union.union_sum = prev["summary"].get("union_sum", 0)
        done_keys = {(r["image_id"], r["qi"]) for r in rows}
        print(f"[pixellm s{args.shard}] resuming with {len(rows)} questions already scored",
              flush=True)

    def flush():
        res = {"model": "pixellm-7b", "mode": "gen", "shard": args.shard,
               "nshards": args.nshards, "union": union.result(),
               "native": {"targets": 0, "gIoU": 0.0, "cIoU": 0.0},
               "inter_sum": union.inter_sum, "union_sum": union.union_sum,
               "complete": False}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        tmp = args.out + ".tmp"
        json.dump({"summary": res, "rows": rows}, open(tmp, "w"), indent=1)
        os.replace(tmp, args.out)

    t0 = time.time()
    nq = 0
    for ri, (rec_idx, rec) in enumerate(sharded, 1):
        path = os.path.join(C.IMAGE_ROOT, rec["file_name"])
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        image_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ori = image_np.shape[:2]

        # CLIP branch: 448, pad-to-square (pad_train_clip_images)
        clip_np = transform_clip.apply_image(image_np)
        clip_resize = clip_np.shape[:2]
        image_clip = preprocess(
            torch.from_numpy(clip_np).permute(2, 0, 1).contiguous(),
            img_size=clip_proc.size["shortest_edge"],
        ).unsqueeze(0).to(dtype).to(args.device)

        # mask branch: 1024
        img_np = transform.apply_image(image_np)
        resize = img_np.shape[:2]
        image = preprocess(
            torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        ).unsqueeze(0).to(dtype).to(args.device)

        for qi, question in enumerate(rec["questions"]):
            if (rec["image_id"], qi) in done_keys:
                continue
            try:
                per_target, gt_union, names = C.decode_gt(rec, qi)
            except Exception:
                continue

            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            prompt = IMAGE_TOKEN + "\n" + "{} Please output segmentation mask.".format(
                question.strip()
            )
            prompt = prompt.replace(IMAGE_TOKEN, IM_START + IMAGE_TOKEN + IM_END)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            input_ids = tokenizer_image_token(
                conv.get_prompt(), tok, return_tensors="pt"
            ).unsqueeze(0).to(args.device)

            pred_masks = None
            for attempt in (0, 1):
                try:
                    with torch.no_grad():
                        _, pred_masks, _, _ = model.evaluate(
                            image_clip, image, input_ids,
                            resize_list=[resize], clip_resize_list=[clip_resize],
                            original_size_list=[ori],
                            max_new_tokens=args.max_new_tokens, tokenizer=tok,
                        )
                    break
                except torch.cuda.OutOfMemoryError:
                    # Transient contention from another job on this GPU. Drop our
                    # cached blocks and try once more; if it still fails, save what
                    # we have so --resume can pick up rather than losing hours.
                    torch.cuda.empty_cache()
                    if attempt == 1:
                        print(f"[pixellm s{args.shard}] OOM at {rec['image_id']}/{qi}; "
                              f"flushing {len(rows)} results and exiting", flush=True)
                        flush()
                        raise

            pred_union = np.zeros(ori, bool)
            for _pm in pred_masks:
                if _pm is None or _pm.shape[0] == 0:
                    continue
                for k in range(_pm.shape[0]):
                    pred_union |= (_pm[k] > 0).detach().cpu().numpy()

            union.add(pred_union, gt_union)
            rows.append({"image_id": rec["image_id"], "qi": qi, "T": len(names),
                         "gran": C.granularity(names), "union_iou": union.ious[-1]})
            nq += 1
            if nq % args.save_every == 0:
                flush()
        if ri % 25 == 0:
            el = time.time() - t0
            print(f"[pixellm s{args.shard}] {ri}/{len(sharded)} imgs, {nq} q, "
                  f"{el:.0f}s ({nq/el:.2f} q/s) | union {union.result()}", flush=True)

    res = {"model": "pixellm-7b", "mode": "gen", "shard": args.shard,
           "nshards": args.nshards, "union": union.result(),
           "native": {"targets": 0, "gIoU": 0.0, "cIoU": 0.0},
           "inter_sum": union.inter_sum, "union_sum": union.union_sum,
           "complete": True}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": res, "rows": rows}, open(args.out, "w"), indent=1)
    print("SUMMARY", json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
