#!/usr/bin/env python3
"""Re-score existing SA-Co/Gold predictions at a range of score thresholds — CPU only, no GPU.

Point of the exercise: CGF1Evaluator constructs CGF1Eval without passing `threshold`, so it
silently uses CGF1Eval's default of **0.5**, not the 0.05 used at inference. Every headline
number is therefore a score>=0.5 number. This sweeps the threshold that actually bites, to
separate "this checkpoint cannot do the task" from "this checkpoint is miscalibrated".
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "/workspace/sam3")
from sam3.eval.cgf1_eval import CGF1Evaluator

ANN = Path("/mnt/data0/ameen/saco_gold/gts")
K = "cgF1_eval_segm_"
ROWS = ["cgF1", "precision", "recall", "positive_micro_F1",
        "IL_MCC", "IL_FPR", "IL_recall", "IL_precision"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--subset", default="sa1b")
    ap.add_argument("--thresholds", default="0.05,0.2,0.35,0.5,0.65,0.8,0.9")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ths = [float(t) for t in args.thresholds.split(",")]
    gts = [str(ANN / f"gold_{args.subset}_merged_{x}_release_test.json") for x in "abc"]
    results = {}
    for t in ths:
        ev = CGF1Evaluator(gt_path=gts, iou_type="segm", verbose=False)
        for ce in ev.coco_evals:          # the knob CGF1Evaluator forgets to expose
            ce.threshold = t
        out = ev.evaluate(args.pred)
        results[f"{t}"] = out
        print(f"threshold {t}: " +
              "  ".join(f"{r}={out[K + r]:.4f}" for r in ROWS), flush=True)
    json.dump(results, open(args.out, "w"), indent=1)

    print("\n| threshold | " + " | ".join(ROWS) + " |")
    print("|---" * (len(ROWS) + 1) + "|")
    for t in ths:
        o = results[f"{t}"]
        print(f"| {t} | " + " | ".join(f"{o[K + r]:.4f}" for r in ROWS) + " |")


if __name__ == "__main__":
    main()
