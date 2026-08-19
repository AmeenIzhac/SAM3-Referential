#!/usr/bin/env python3
"""Compute M2SA-7B (teacher-forced, official protocol) masks for the selected
examples; save to npz. Run with the m2sa env python, cwd = MMR repo."""
import json, os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad")
sys.path.insert(0, "/tmp/claude-1009/-workspace-reasonseg/97d462f0-0abb-44be-b9f3-2c40c69ab6d3/scratchpad/MMR")
import mmr_common as C
from run_m2sa_mmr import build, sam_preprocess, with_imtoken, IMAGE_TOKEN
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

exs = json.load(open("/workspace/reasonseg/mmrcomp/examples.json"))
records = {r["image_id"]: r for r in C.load_val()}
dtype = torch.bfloat16
clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
transform = ResizeLongestSide(1024)
tok, model = build("cuda:0", dtype)

store = {}
for i, e in enumerate(exs):
    rec = records[e["image_id"]]
    per_target, gt_union, _ = C.decode_gt(rec, e["qi"])
    image = cv2.cvtColor(cv2.imread(os.path.join(C.IMAGE_ROOT, rec["file_name"])), cv2.COLOR_BGR2RGB)
    ori = image.shape[:2]
    image_clip = clip_proc.preprocess(image, return_tensors="pt")["pixel_values"][0].to(dtype).to("cuda:0").unsqueeze(0)
    img_sam = transform.apply_image(image); resize = img_sam.shape[:2]
    img_sam = sam_preprocess(torch.from_numpy(img_sam).permute(2,0,1).contiguous()).to(dtype).to("cuda:0").unsqueeze(0)
    ta = rec["text_answers"][e["qi"]]
    conv = conversation_lib.default_conversation.copy(); conv.messages = []
    conv.append_message(conv.roles[0], IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(e["question"].strip()))
    conv.append_message(conv.roles[1], "{}.".format(ta.format(seg="[SEG]")))
    prompt = with_imtoken(conv.get_prompt())
    input_ids = tokenizer_image_token(prompt, tok, return_tensors="pt").unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        out = model(images=img_sam, images_clip=image_clip, input_ids=input_ids, labels=None,
                    attention_masks=input_ids.ne(tok.pad_token_id),
                    offset=torch.LongTensor([0,1]).to("cuda:0"),
                    masks_list=[torch.from_numpy(per_target.astype(np.float32)).to("cuda:0")],
                    label_list=[torch.zeros(ori, device="cuda:0")], resize_list=[resize], inference=True)
    pm = out["pred_masks"][0]
    u = np.zeros(ori, bool)
    for k in range(pm.shape[0]):
        u |= (pm[k] > 0).detach().cpu().numpy()
    store[f"{i}_m2sa"] = u.astype(np.uint8)
    print(f"  ex{i} m2sa done", flush=True)
np.savez_compressed("/workspace/reasonseg/mmrcomp/_masks_m2sa.npz", **store)
print("saved -> _masks_m2sa.npz")
