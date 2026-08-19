#!/usr/bin/env python3
"""Assemble the MMR data-scaling ladder from the in-flight benchmark dumps.

Reads every mmr_eval/mmrscale_*.json written by bench_scale.sh, sorts the
rungs, and emits a markdown table plus scaling_curve_full.png.

Rung tags look like  pct16_s7299_n7299  ->  16 % of the data, taken at trainer
step 7299, after 7299 images. One image carries 3.378 MMR questions on average,
so the sample count is reported as images x 3.378.
"""
import bisect, glob, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = "/workspace/reasonseg/mmr_eval"
OUTDIR = "/workspace/reasonseg/mmrcomp"
Q_PER_IMG = 154124 / 45618          # 3.378 questions per MMR train image
TOTAL_IMAGES = 45618
TOTAL_SAMPLES = 154124

# published reference points on the same benchmark (see mmrcomp/README.md)
REFS = [(36.0, "M²SA-7B (teacher-forced)", "#b45309"),
        (25.2, "M²SA-7B (question-only)", "#c2410c"),
        (11.1, "SAM3 reasonseg fine-tune", "#9333ea")]
BASE_GIOU, BASE_CIOU = 0.6, 1.5     # original SAM3, zero-shot, question prompt


TRAIN_LOG = "/mnt/data0/ameen/mmr_runs/mmr_scale_train.log"


def grad_skips():
    """Steps the trainer dropped on a non-finite grad norm, so each rung can be
    labelled with how many optimizer updates it ACTUALLY received."""
    if not os.path.exists(TRAIN_LOG):
        return []
    txt = open(TRAIN_LOG, errors="ignore").read()
    return sorted({int(m) for m in
                   re.findall(r"iter (\d+); skipping optimizer step", txt)})


def load():
    skips = grad_skips()
    rows = []
    for f in glob.glob(f"{EVAL}/mmrscale_*.json"):
        d = json.load(open(f))
        m = re.match(r"mmrscale_pct([\d.]+)_s(\d+)_n(\d+)", d["label"])
        if not m:
            continue
        pct, step, imgs = float(m.group(1)), int(m.group(2)), int(m.group(3))
        o = d["overall"]
        nskip = bisect.bisect_right(skips, step)
        rows.append({"pct": pct, "step": step, "images": imgs,
                     "samples": round(imgs * Q_PER_IMG),
                     "gIoU": o["gIoU"], "cIoU": o["cIoU"], "iou50": o["iou50"],
                     "by_gran": {g: v["gIoU"] for g, v in d["by_gran"].items()},
                     "single": d["single_target"]["gIoU"],
                     "multi": d["multi_target"]["gIoU"],
                     "skipped_steps": nskip, "effective_updates": step - nskip,
                     # a rung is only a real data point if nearly every step it
                     # was meant to train on actually produced an update
                     "valid": nskip <= 0.01 * step})
    return sorted(rows, key=lambda r: r["pct"])


def table(rows):
    L = ["| % of data | images | samples | effective updates | gIoU | cIoU | IoU@50 | obj | part | obj&part |",
         "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
         f"| 0 (base SAM3) | 0 | 0 | 0 | {BASE_GIOU} | {BASE_CIOU} | 0.6 | 0.4 | 0.3 | 1.1 |"]
    for r in rows:
        g = r["by_gran"]
        pct = f"{r['pct']:g}" if r["pct"] < 99 else "100"
        eff = f"{r['effective_updates']:,} / {r['step']:,}"
        if r["valid"]:
            L.append(f"| **{pct} %** | {r['images']:,} | {r['samples']:,} | {eff} | "
                     f"**{r['gIoU']}** | {r['cIoU']} | {r['iou50']} | {g.get('obj', 0)} | "
                     f"{g.get('part', 0)} | {g.get('obj&part', 0)} |")
        else:
            L.append(f"| ~~{pct} %~~ ⚠️ | {r['images']:,} | {r['samples']:,} | **{eff}** | "
                     f"~~{r['gIoU']}~~ | ~~{r['cIoU']}~~ | ~~{r['iou50']}~~ | ~~{g.get('obj', 0)}~~ | "
                     f"~~{g.get('part', 0)}~~ | ~~{g.get('obj&part', 0)}~~ |")
    bad = [r for r in rows if not r["valid"]]
    if bad:
        L.append("")
        L.append(f"⚠️ **{len(bad)} rung(s) struck through are NOT valid data points.** The trainer "
                 "dropped those optimizer steps on non-finite gradient norms, so the model did not "
                 "actually train on the extra data they claim. See *The run stopped learning* below.")
    return "\n".join(L)


def plot(rows):
    rows = [r for r in rows if r["valid"]]   # never plot rungs the model did not train
    xs = [r["samples"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(xs, [r["gIoU"] for r in rows], "o-", color="#2563eb", lw=2.4, ms=7,
            label="gIoU", zorder=5)
    ax.plot(xs, [r["cIoU"] for r in rows], "s-", color="#0d9488", lw=2.2, ms=6,
            label="cIoU", zorder=5)
    for r in rows:
        ax.annotate(f"{r['gIoU']:.1f}", (r["samples"], r["gIoU"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8.5, color="#2563eb", fontweight="bold")
    if xs:
        ax.scatter([xs[0] * 0.45], [BASE_GIOU], marker="v", s=55, color="#64748b", zorder=5)
        ax.annotate(f"base SAM3 {BASE_GIOU}", (xs[0] * 0.45, BASE_GIOU),
                    textcoords="offset points", xytext=(4, 8), fontsize=8.5, color="#64748b")
    for y, txt, col in REFS:
        ax.axhline(y, ls="--", lw=1.2, color=col, alpha=.85)
        ax.text(0.995, y, f"{txt} = {y} ", va="bottom", ha="right", fontsize=8.2,
                color=col, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("MMR training samples seen — cumulative, ONE continuously-trained model (log scale)")
    ax.set_ylabel("MMR val score (×100)")
    ax.set_title("Original SAM3 on MMR — data-scaling ladder (question prompt)",
                 fontweight="bold")
    ax.set_ylim(0, max(42, max([r["gIoU"] for r in rows] + [0]) + 8))
    ax.grid(True, which="both", axis="y", alpha=.22)
    ax.grid(True, which="major", axis="x", alpha=.12)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = f"{OUTDIR}/scaling_curve_full.png"
    fig.savefig(p, dpi=155)
    return p


def splice(md, rows):
    """Keep scaling_ladder.md self-contained: drop the current table in at the
    RESULTS-TABLE marker, so the doc reads correctly even mid-run."""
    p = f"{OUTDIR}/scaling_ladder.md"
    if not os.path.exists(p):
        return
    doc = open(p).read()
    done = len(rows) >= 10
    status = ("" if done else
              f"\n*Run in flight — {len(rows)} of 10 rungs landed so far.*\n")
    block = f"<!-- RESULTS-TABLE -->\n{status}\n{md}\n<!-- /RESULTS-TABLE -->"
    if "<!-- RESULTS-TABLE -->" in doc:
        pre = doc.split("<!-- RESULTS-TABLE -->")[0]
        post = doc.split("<!-- /RESULTS-TABLE -->")[-1] if "<!-- /RESULTS-TABLE -->" in doc \
            else doc.split("<!-- RESULTS-TABLE -->")[1]
        open(p, "w").write(pre + block + post)


if __name__ == "__main__":
    rows = load()
    if not rows:
        raise SystemExit("no mmrscale_*.json rungs found yet")
    md = table(rows)
    print(md)
    print()
    best = max(rows, key=lambda r: r["gIoU"])
    print(f"rungs: {len(rows)}   peak gIoU {best['gIoU']} at {best['pct']:g} % "
          f"({best['samples']:,} samples)")
    print("wrote", plot(rows))
    json.dump(rows, open(f"{OUTDIR}/scaling_ladder.json", "w"), indent=1)
    splice(md, rows)
