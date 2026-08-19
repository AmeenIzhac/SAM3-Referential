#!/usr/bin/env python3
"""Compose per-example comparison panels for mmrcomp.
Row = PROMPT banner + [Image | GT | M2SA | SAM3 base-Q | SAM3 ft-Q | SAM3 base-oracle | SAM3 ft-oracle].
Overlays: yellow = correct (pred∩GT), red = false positive, green = missed (GT only)."""
import json, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = "/workspace/reasonseg/mmrcomp"
IMG_ROOT = "/mnt/data0/ameen/mmr_data/images"
PW, PH = 360, 300          # panel image area
LBL = 40                   # per-panel label strip height
PAD = 6
exs = json.load(open(f"{ROOT}/examples.json"))
sam3 = np.load(f"{ROOT}/_masks_sam3.npz")
m2sa = np.load(f"{ROOT}/_masks_m2sa.npz")

def font(sz, bold=False):
    for p in [f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

F_BAN, F_BANs, F_LBL = font(20, True), font(15), font(15, True)

# columns: (key, title). key None -> raw image; 'gt' -> GT only
COLS = [
    (None, "Image + PROMPT"),
    ("gt", "Ground truth"),
    ("m2sa", "M2SA-7B (official)"),
    ("sam3_base_q", "SAM3 base  ·  question"),
    ("sam3_trained_q", "SAM3 fine-tuned  ·  question"),
    ("sam3_base_c", "SAM3 base  ·  oracle concept"),
    ("sam3_trained_c", "SAM3 fine-tuned  ·  oracle"),
]
IOU_KEY = {"m2sa": "m2sa", "sam3_base_q": "sam3_base_q", "sam3_trained_q": "sam3_tr_q",
           "sam3_base_c": "sam3_base_c", "sam3_trained_c": "sam3_tr_c"}

def letterbox(arr, is_mask=False):
    h, w = arr.shape[:2]
    s = min(PW / w, PH / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    im = Image.fromarray(arr)
    im = im.resize((nw, nh), Image.NEAREST if is_mask else Image.BILINEAR)
    canvas = Image.new("RGB" if not is_mask else "L", (PW, PH), (30, 30, 30) if not is_mask else 0)
    canvas.paste(im, ((PW - nw) // 2, (PH - nh) // 2))
    return np.array(canvas)

def overlay(img, pred, gt, is_gt=False):
    """GT panel: green fill. Pred panel: red fill + bright-green GT outline."""
    out = img.astype(np.float32).copy()
    if is_gt:
        m = gt.astype(bool)
        out[m] = .55 * out[m] + .45 * np.array((0, 200, 0), np.float32)
        return out.clip(0, 255).astype(np.uint8)
    p = pred.astype(bool)
    out[p] = .5 * out[p] + .5 * np.array((235, 40, 40), np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 235, 0), 2)
    return out

def wrap(draw, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines

rows_imgs = []
for i, e in enumerate(exs):
    img = np.array(Image.open(os.path.join(IMG_ROOT, e["file_name"])).convert("RGB"))
    base = letterbox(img)
    gt = letterbox(sam3[f"{i}_gt"], True) > 0
    # banner
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    qlines = wrap(tmp, "PROMPT:  " + e["question"], F_BAN, len(COLS) * (PW + PAD) - 20)
    sub = f"target(s): {', '.join(sorted(set(e['cat_names'])))}   |   granularity: {e['gran']}   |   #targets: {e['T']}   |   [{e['note']}]"
    BANH = 12 + len(qlines) * 24 + 22
    rowW = len(COLS) * PW + (len(COLS) + 1) * PAD
    rowH = BANH + PH + LBL + 2 * PAD
    row = Image.new("RGB", (rowW, rowH), (245, 245, 245))
    d = ImageDraw.Draw(row)
    y = 8
    for ln in qlines:
        d.text((12, y), ln, fill=(10, 10, 10), font=F_BAN); y += 24
    d.text((12, y), sub, fill=(90, 90, 90), font=F_BANs)
    # panels
    for c, (key, title) in enumerate(COLS):
        x = PAD + c * (PW + PAD)
        yimg = BANH + PAD
        if key is None:
            panel = base.copy()
        elif key == "gt":
            panel = overlay(base, gt, gt, is_gt=True)
        else:
            arr = m2sa[f"{i}_m2sa"] if key == "m2sa" else sam3[f"{i}_{key}"]
            pred = letterbox(arr, True) > 0
            panel = overlay(base, pred, gt)
        row.paste(Image.fromarray(panel), (x, yimg))
        pd = ImageDraw.Draw(row)
        # label strip
        ly = yimg + PH
        pd.rectangle([x, ly, x + PW, ly + LBL], fill=(255, 255, 255))
        pd.text((x + 4, ly + 3), title, fill=(0, 0, 0), font=F_LBL)
        if key in IOU_KEY:
            iou = e["ious"][IOU_KEY[key]]
            col = (0, 140, 0) if iou >= .5 else (200, 60, 0)
            pd.text((x + 4, ly + 21), f"IoU = {iou:.3f}", fill=col, font=F_LBL)
        elif key is None:
            # overlay prompt directly on the image panel too
            ov = Image.new("RGBA", (PW, 46), (0, 0, 0, 150))
            od = ImageDraw.Draw(ov)
            for j, ln in enumerate(wrap(od, e["question"], F_BANs, PW - 10)[:2]):
                od.text((5, 3 + j * 20), ln, fill=(255, 255, 255, 255), font=F_BANs)
            row.paste(Image.alpha_composite(Image.fromarray(base[:46]).convert("RGBA"), ov).convert("RGB"), (x, yimg))
        elif key == "gt":
            pd.text((x + 4, ly + 21), "green = target", fill=(0, 140, 0), font=F_LBL)
    row.save(f"{ROOT}/examples/ex{i}_{e['note']}.png")
    rows_imgs.append(row)
    print("wrote", f"examples/ex{i}_{e['note']}.png")

# contact sheet (legend + all rows)
legend_h = 30
W = max(r.width for r in rows_imgs)
H = legend_h + sum(r.height for r in rows_imgs)
sheet = Image.new("RGB", (W, H), (255, 255, 255))
ld = ImageDraw.Draw(sheet)
ld.text((12, 7), "Overlay:  red fill = model prediction     green outline = ground-truth target   (perfect prediction = red fills the green outline)   |   MMR val, 7 examples",
        fill=(0, 0, 0), font=F_LBL)
y = legend_h
for r in rows_imgs:
    sheet.paste(r, (0, y)); y += r.height
sheet.save(f"{ROOT}/mmr_examples_contact_sheet.png")
print("wrote mmr_examples_contact_sheet.png", sheet.size)
