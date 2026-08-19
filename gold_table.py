#!/usr/bin/env python3
"""Collate SA-Co/Gold per-subset metric dumps into the markdown tables for the write-up.

The paper's headline triple is cgF1 / IL_MCC / positive_micro_F1, averaged over the 7
subsets; we also carry the precision/recall/FPR split because that is where the
fine-tunes actually differ from base."""
import json, sys
from pathlib import Path

ROOT = Path("/mnt/data0/ameen/gold_eval")
SUBSETS = ["metaclip", "sa1b", "crowded", "fg_food", "fg_sports_equipment",
           "attributes", "wiki_common"]
NICE = {"metaclip": "Captioner MetaCLIP", "sa1b": "Captioner SA-1B",
        "crowded": "Crowded", "fg_food": "FG food", "fg_sports_equipment": "FG sport",
        "attributes": "Attributes", "wiki_common": "Wiki common"}
# stock SAM3 as published in facebookresearch/sam3 scripts/eval/gold/README.md
PUBLISHED = {"metaclip": (47.26, 0.81, 58.58), "sa1b": (53.69, 0.86, 62.55),
             "crowded": (61.08, 0.90, 67.73), "fg_food": (53.41, 0.79, 67.28),
             "fg_sports_equipment": (65.52, 0.89, 73.75), "attributes": (54.93, 0.76, 72.00),
             "wiki_common": (42.53, 0.70, 60.85)}
K = "cgF1_eval_segm_"


def load(tag, subset):
    p = ROOT / tag / f"{subset}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def main():
    tags = sys.argv[1:] or ["base", "reasonseg", "mmr"]
    have = {t: {s: load(t, s) for s in SUBSETS} for t in tags}

    print("## Per-subset cgF1\n")
    print("| model | " + " | ".join(NICE[s] for s in SUBSETS) + " | average |")
    print("|---" * (len(SUBSETS) + 2) + "|")
    print("| SAM 3 (published) | " +
          " | ".join(f"{PUBLISHED[s][0]:.2f}" for s in SUBSETS) + " | " +
          f"{sum(PUBLISHED[s][0] for s in SUBSETS)/len(SUBSETS):.2f} |")
    for t in tags:
        cells, vals = [], []
        for s in SUBSETS:
            m = have[t][s]
            if m is None:
                cells.append("–")
            else:
                v = m[K + "cgF1"] * 100
                cells.append(f"{v:.2f}")
                vals.append(v)
        avg = f"{sum(vals)/len(vals):.2f}" if len(vals) == len(SUBSETS) else "–"
        print(f"| {t} | " + " | ".join(cells) + f" | {avg} |")

    for metric, label, scale in [("IL_MCC", "IL_MCC", 1),
                                 ("positive_micro_F1", "positive_micro_F1", 100)]:
        print(f"\n## Per-subset {label}\n")
        print("| model | " + " | ".join(NICE[s] for s in SUBSETS) + " | average |")
        print("|---" * (len(SUBSETS) + 2) + "|")
        idx = 1 if metric == "IL_MCC" else 2
        print("| SAM 3 (published) | " +
              " | ".join(f"{PUBLISHED[s][idx]:.2f}" for s in SUBSETS) + " | " +
              f"{sum(PUBLISHED[s][idx] for s in SUBSETS)/len(SUBSETS):.2f} |")
        for t in tags:
            cells, vals = [], []
            for s in SUBSETS:
                m = have[t][s]
                if m is None:
                    cells.append("–")
                else:
                    v = m[K + metric] * scale
                    cells.append(f"{v:.2f}")
                    vals.append(v)
            avg = f"{sum(vals)/len(vals):.2f}" if len(vals) == len(SUBSETS) else "–"
            print(f"| {t} | " + " | ".join(cells) + f" | {avg} |")

    # a 7-subset average is only comparable if every model has all 7; when one does not,
    # average over the subsets they all share so the columns still line up
    shared = [s for s in SUBSETS if all(have[t][s] for t in tags)]
    if shared and len(shared) < len(SUBSETS):
        print(f"\n## Matched-subset average ({len(shared)} subsets all models completed: "
              f"{', '.join(NICE[s] for s in shared)})\n")
        print("| model | cgF1 | IL_MCC | positive_micro_F1 |")
        print("|---|---|---|---|")
        print("| SAM 3 (published) | " +
              " | ".join(f"{sum(PUBLISHED[s][i] for s in shared)/len(shared):.2f}"
                         if i != 1 else
                         f"{sum(PUBLISHED[s][1] for s in shared)/len(shared):.3f}"
                         for i in (0, 1, 2)) + " |")
        for t in tags:
            c = sum(have[t][s][K + "cgF1"] for s in shared) / len(shared) * 100
            m = sum(have[t][s][K + "IL_MCC"] for s in shared) / len(shared)
            f = sum(have[t][s][K + "positive_micro_F1"] for s in shared) / len(shared) * 100
            print(f"| {t} | {c:.2f} | {m:.3f} | {f:.2f} |")

    print("\n## Diagnostic split (mean over completed subsets)\n")
    rows = ["cgF1", "precision", "recall", "IL_F1", "IL_MCC", "IL_precision",
            "IL_recall", "IL_FPR"]
    print("| metric | " + " | ".join(tags) + " |")
    print("|---" * (len(tags) + 1) + "|")
    for r in rows:
        cells = []
        for t in tags:
            vs = [m[K + r] for m in have[t].values() if m]
            cells.append(f"{sum(vs)/len(vs):.3f}" if vs else "–")
        print(f"| {r} | " + " | ".join(cells) + " |")
    for t in tags:
        n = sum(1 for m in have[t].values() if m)
        print(f"\n<!-- {t}: {n}/{len(SUBSETS)} subsets scored -->")


if __name__ == "__main__":
    main()
