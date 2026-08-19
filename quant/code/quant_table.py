#!/usr/bin/env python3
"""Build the results table for quant/eval/*.json (markdown, stdout)."""
import glob
import json
import os
import sys

EVAL = "/workspace/reasonseg/quant/eval"
ORDER = ["fp32", "bf16", "int8_g128", "int6_g128", "int5_g128", "int4_g128",
         "int3_g128", "int2_g128", "int8_pc", "int4_pc", "int4_g64", "int4_g32",
         "int4_g128_noemb", "int3_g64", "int3_g32", "int5_pc",
         "int4_g128_bf16rest", "int5_g128_bf16rest", "int4_from_ckpt",
         "int3_vision_only", "int3_language_only", "int3_heads_only",
         "int4_g128_head8", "int3_g128_head8", "int3_g64_head8"]
LABEL = {"fp32": "fp32 (baseline)", "bf16": "bf16 cast",
         "int8_g128": "**int8** g128", "int6_g128": "int6 g128",
         "int5_g128": "**int5** g128", "int4_g128": "**int4** g128",
         "int3_g128": "int3 g128", "int2_g128": "int2 g128",
         "int8_pc": "int8 per-channel", "int4_pc": "int4 per-channel",
         "int4_g64": "int4 g64", "int4_g32": "int4 g32",
         "int4_g128_noemb": "int4 g128, no emb",
         "int3_g64": "int3 g64", "int3_g32": "int3 g32",
         "int5_pc": "int5 per-channel",
         "int4_g128_bf16rest": "int4 g128, bf16 remainder",
         "int5_g128_bf16rest": "int5 g128, bf16 remainder",
         "int4_from_ckpt": "int4 g128, from the artifact",
         "int3_vision_only": "int3, vision backbone only",
         "int3_language_only": "int3, language backbone only",
         "int3_heads_only": "int3, detection heads only",
         "int4_g128_head8": "int4 g128 + **int8 heads**",
         "int3_g128_head8": "int3 g128 + **int8 heads**",
         "int3_g64_head8": "int3 g64 + **int8 heads**"}


# storage per quantized weight: `bits` for the code, plus one bf16 scale (16b)
# and one uint8 zero (8b) shared by each group of `group_size`. Without this the
# group-size rows are not size-comparable to the bit-width rows: int4/g32 costs
# 4.75 bits/weight, which is really competing with int5/g128's 5.19.
def bits_per_weight(bits, group_size, mean_in_features=1500,
                    head_bits=None, n_head=0, n_total=0):
    g = group_size if group_size > 0 else mean_in_features
    body = bits + 24.0 / g
    if not head_bits or not n_head or not n_total:
        return body
    head = head_bits + 24.0 / g
    return ((n_total - n_head) * body + n_head * head) / n_total


def load():
    out = {}
    for f in glob.glob(os.path.join(EVAL, "mmr_*.json")):
        tag = os.path.basename(f)[len("mmr_"):-len(".json")]
        out[tag] = json.load(open(f))
    return out


def main():
    res = load()
    tags = [t for t in ORDER if t in res] + sorted(t for t in res if t not in ORDER)
    if not tags:
        sys.exit("no results in " + EVAL)
    base = res.get("fp32", res[tags[0]])["overall"]["gIoU"]

    print("| weights | bits/weight | gIoU | Δ gIoU | cIoU | IoU@50 | Obj | Part | Obj&Part | n |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for t in tags:
        r = res[t]
        o, g = r["overall"], r["by_gran"]
        i = r.get("build_info") or {}
        if i.get("bits"):
            bpw = f"{bits_per_weight(i['bits'], i.get('group_size', 128), head_bits=i.get('head_bits'), n_head=i.get('head_quantized_params', 0), n_total=i.get('quantized_params', 0)):.2f}"
        else:
            bpw = "32" if t == "fp32" else ("16" if t == "bf16" else "—")
        d = o["gIoU"] - base
        print(f"| {LABEL.get(t, t)} | {bpw} | {o['gIoU']:.2f} | {d:+.2f} | {o['cIoU']:.2f} | "
              f"{o['iou50']:.2f} | {g['obj']['gIoU']:.2f} | {g['part']['gIoU']:.2f} | "
              f"{g['obj&part']['gIoU']:.2f} | {o['n']} |")

    print("\n| weights | single-target gIoU | multi-target gIoU | coverage | rel_err mean | rel_err max | worst tensor |")
    print("|---|--:|--:|--:|--:|--:|---|")
    for t in tags:
        r = res[t]
        i = r.get("build_info") or {}
        cov = i.get("coverage")
        em = f"{i['rel_err_mean']:.4f}" if i.get("rel_err_mean") else "—"
        ex = f"{i['rel_err_max']:.4f}" if i.get("rel_err_max") else "—"
        worst = (i.get("rel_err_max_tensor") or "—").replace("backbone.", "")
        covs = f"{100 * cov:.1f} %" if cov else "—"
        print(f"| {LABEL.get(t, t)} | {r['single_target']['gIoU']:.2f} | "
              f"{r['multi_target']['gIoU']:.2f} | {covs} | {em} | {ex} | {worst} |")


if __name__ == "__main__":
    main()
