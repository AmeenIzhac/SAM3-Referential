#!/usr/bin/env python3
"""Run M2SA-7B over MMR val and score gIoU/cIoU.

 mode=teacher : MMR's OFFICIAL protocol -- teacher-force the GT answer text
                (which names the targets) so the model decodes exactly the GT
                number of [SEG] masks. Reproduces the paper's per-target gIoU/cIoU
                AND yields a union mask per question.
 mode=gen     : question-only free generation via model.evaluate() -- the honest
                setting comparable to how SAM3 is prompted. Union metric only.

Must run with the m2sa conda env python, cwd = the MMR repo.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad")
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


def sam_preprocess(x):
    x = (x - PIX_MEAN) / PIX_STD
    h, w = x.shape[-2:]
    x = torch.nn.functional.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x


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
    ap.add_argument("--limit", type=int, default=None, help="limit #images (records)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    transform = ResizeLongestSide(IMG_SIZE)
    tok, model = build(args.device, dtype)
    import cv2

    records = C.load_val()
    sharded = C.shard_records(records, args.shard, args.nshards)
    if args.limit:
        sharded = sharded[: args.limit]

    native = C.NativeMeter()
    union = C.UnionMeter()
    rows = []
    t0 = time.time()
    nq = 0
    for ri, (rec_idx, rec) in enumerate(sharded, 1):
        path = os.path.join(C.IMAGE_ROOT, rec["file_name"])
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ori = image.shape[:2]
        image_clip = clip_proc.preprocess(image, return_tensors="pt")["pixel_values"][0]
        img_sam = transform.apply_image(image)
        resize = img_sam.shape[:2]
        img_sam = sam_preprocess(torch.from_numpy(img_sam).permute(2, 0, 1).contiguous())
        img_sam = img_sam.to(dtype).to(args.device).unsqueeze(0)
        image_clip = image_clip.to(dtype).to(args.device).unsqueeze(0)

        for qi, question in enumerate(rec["questions"]):
            try:
                per_target, gt_union, names = C.decode_gt(rec, qi)
            except Exception as e:
                continue
            gt_bins = [per_target[k] for k in range(per_target.shape[0])]
            text_answer = rec["text_answers"][qi]

            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            if args.mode == "teacher":
                if text_answer.count("{seg}") != len(gt_bins):
                    continue  # malformed; MMR loader skips these too
                seg_ans = text_answer.format(seg="[SEG]")
                conv.append_message(conv.roles[0],
                    IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(question.strip()))
                conv.append_message(conv.roles[1], "{}.".format(seg_ans))
                prompt = with_imtoken(conv.get_prompt())
                input_ids = tokenizer_image_token(prompt, tok, return_tensors="pt").unsqueeze(0).to(args.device)
                attn = input_ids.ne(tok.pad_token_id)
                with torch.no_grad():
                    out = model(
                        images=img_sam, images_clip=image_clip, input_ids=input_ids,
                        labels=None, attention_masks=attn, offset=torch.LongTensor([0, 1]).to(args.device),
                        masks_list=[torch.from_numpy(per_target.astype(np.float32)).to(args.device)],
                        label_list=[torch.zeros(ori, device=args.device)],
                        resize_list=[resize], inference=True,
                    )
                pm = out["pred_masks"][0]  # [T,H,W] logits
                pred_bins = [(pm[k] > 0).detach().cpu().numpy() for k in range(pm.shape[0])]
                if len(pred_bins) == len(gt_bins):
                    native.add_question(pred_bins, gt_bins)
                pred_union = np.zeros(ori, bool)
                for pb in pred_bins:
                    pred_union |= pb
            else:  # gen
                conv.append_message(conv.roles[0],
                    IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(question.strip()))
                conv.append_message(conv.roles[1], None)
                prompt = with_imtoken(conv.get_prompt())
                input_ids = tokenizer_image_token(prompt, tok, return_tensors="pt").unsqueeze(0).to(args.device)
                with torch.no_grad():
                    _, pred_masks = model.evaluate(
                        image_clip, img_sam, input_ids,
                        resize_list=[resize], original_size_list=[ori],
                        max_new_tokens=256, tokenizer=tok,
                    )
                pm = pred_masks[0]
                pred_union = np.zeros(ori, bool)
                for k in range(pm.shape[0]):
                    pred_union |= (pm[k] > 0).detach().cpu().numpy()

            union.add(pred_union, gt_union)
            rows.append({"image_id": rec["image_id"], "qi": qi, "T": len(gt_bins),
                         "gran": C.granularity(names),
                         "union_iou": union.ious[-1]})
            nq += 1
        if ri % 25 == 0:
            el = time.time() - t0
            print(f"[{args.mode} s{args.shard}] {ri}/{len(sharded)} imgs, {nq} q, "
                  f"{el:.0f}s ({nq/el:.2f} q/s) | union {union.result()} | native {native.result()}",
                  flush=True)

    res = {"mode": args.mode, "shard": args.shard, "nshards": args.nshards,
           "union": union.result(), "native": native.result(),
           "inter_sum": union.inter_sum, "union_sum": union.union_sum,
           "native_inter": native.inter_sum, "native_union": native.union_sum,
           "native_acc": native.acc_iou_sum, "native_targets": native.target_count}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": res, "rows": rows}, open(args.out, "w"), indent=1)
    print("SUMMARY", json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
