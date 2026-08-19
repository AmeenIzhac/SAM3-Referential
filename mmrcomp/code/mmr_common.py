"""Shared MMR-val GT decoding + gIoU/cIoU metrics, imported by both the
M2SA harness (m2sa env) and the SAM3 harness (sam3 env) so both score the
IDENTICAL ground truth the IDENTICAL way, question-by-question."""
import json
import numpy as np
from pycocotools import mask as mask_util

VAL_JSON = "/mnt/data0/ameen/mmr_data/MMR_val.json"
IMAGE_ROOT = "/mnt/data0/ameen/mmr_data/images"   # + record["file_name"] (e.g. train2017/xxx.jpg)


def load_val(path=VAL_JSON):
    return json.load(open(path))


def shard_records(records, shard=0, nshards=1):
    """Deterministic contiguous-stride shard over images (records)."""
    return [(i, r) for i, r in enumerate(records) if i % nshards == shard]


def decode_rle(seg):
    rle = {"size": seg["size"], "counts": seg["counts"]}
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode()
    return mask_util.decode(rle).astype(bool)


def decode_gt(record, qi):
    """Return (per_target_masks [T,H,W] bool, gt_union [H,W] bool, cat_names)."""
    answer_list = record["answers"][qi]
    masks = []
    names = []
    for tgt in answer_list:
        m = decode_rle(tgt["segmentation"])
        if m.ndim > 2:
            m = m.any(axis=2)
        masks.append(m)
        names.append(tgt.get("category_name", "?"))
    per_target = np.stack(masks, axis=0)
    union = per_target.any(axis=0)
    return per_target, union, names


def fg_inter_union(pred_bin, gt_bin):
    """Foreground (class-1) intersection & union counts for a binary pair."""
    inter = int(np.logical_and(pred_bin, gt_bin).sum())
    union = int(pred_bin.sum()) + int(gt_bin.sum()) - inter
    return inter, union


class UnionMeter:
    """gIoU = mean per-question IoU of (union-of-pred) vs (union-of-GT-targets);
    cIoU = cumulative intersection / cumulative union. Matches the LISA/ReasonSeg
    convention (iou=1 when both empty)."""
    def __init__(self):
        self.inter_sum = 0
        self.union_sum = 0
        self.ious = []

    def add(self, pred_union, gt_union):
        inter, union = fg_inter_union(pred_union, gt_union)
        self.inter_sum += inter
        self.union_sum += union
        self.ious.append(1.0 if union == 0 else inter / union)

    def result(self):
        n = len(self.ious)
        giou = float(np.mean(self.ious)) if n else 0.0
        ciou = float(self.inter_sum / self.union_sum) if self.union_sum else 0.0
        return {"n": n, "gIoU": round(100 * giou, 2), "cIoU": round(100 * ciou, 2),
                "iou50": round(100 * float(np.mean([i >= 0.5 for i in self.ious])), 2) if n else 0.0}


class NativeMeter:
    """Reproduces MMR's official validate(): per-TARGET IoU (teacher-forced [SEG]
    count), gIoU averaged over all targets, cIoU cumulative over all targets."""
    def __init__(self):
        self.inter_sum = 0.0
        self.union_sum = 0.0
        self.acc_iou_sum = 0.0
        self.target_count = 0

    def add_question(self, pred_bins, gt_bins):
        """pred_bins, gt_bins: aligned lists/arrays of per-target binary masks."""
        T = len(gt_bins)
        if T == 0:
            return
        q_inter = 0.0
        q_union = 0.0
        q_acc = 0.0
        for pb, gb in zip(pred_bins, gt_bins):
            inter, union = fg_inter_union(pb, gb)
            q_inter += inter
            q_union += union
            q_acc += (1.0 if union == 0 else inter / (union + 1e-5))
        self.inter_sum += q_inter
        self.union_sum += q_union
        self.acc_iou_sum += q_acc          # sum of per-target IoU (already summed over T)
        self.target_count += T

    def result(self):
        giou = self.acc_iou_sum / self.target_count if self.target_count else 0.0
        ciou = self.inter_sum / (self.union_sum + 1e-10) if self.union_sum else 0.0
        return {"targets": self.target_count, "gIoU": round(100 * giou, 2),
                "cIoU": round(100 * ciou, 2)}


def granularity(cat_names):
    """MMR splits results by object vs part; a part category_name contains ':'
    (e.g. 'cup:handle'). A question is Obj-only, Part-only, or mixed."""
    has_part = any(":" in n for n in cat_names)
    has_obj = any(":" not in n for n in cat_names)
    if has_part and has_obj:
        return "obj&part"
    return "part" if has_part else "obj"
