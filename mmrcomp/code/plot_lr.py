#!/usr/bin/env python3
"""Overlay the LR schedule on the data-scaling ladder.

The run uses one continuous inverse-sqrt schedule across the whole epoch, so the
LR is a function of the step -- and since one step is one image, it is a function
of how much DATA has been seen. Plotting the two together shows how much of the
ladder's flattening coincides with the LR decaying away.
"""
import json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/newsam/sam3")
from sam3.train.optim.schedulers import InverseSquareRootParamScheduler as Sched

OUTDIR = "/workspace/reasonseg/mmrcomp"
TOTAL_STEPS = 45618                  # one image per step, 1 epoch
Q_PER_IMG = 154124 / 45618
WARMUP = COOLDOWN = TIMESCALE = 20
GROUPS = {"transformer": 8e-4 * 0.1,          # 8e-5
          "vision backbone": 2.5e-4 * 0.1,    # 2.5e-5 (x 0.9^depth layer-wise)
          "language backbone": 5e-5 * 0.1}    # 5e-6


def lr_at(base, step):
    s = Sched(base_lr=base, warmup_steps=WARMUP, cooldown_steps=COOLDOWN,
              timescale=TIMESCALE)
    return s(step=step, where=step / TOTAL_STEPS)


def mass(base, n):
    """Cumulative sum of LR over steps -- a proxy for total parameter movement."""
    t = sum(base * min(1.0, k / WARMUP) for k in range(1, min(n, WARMUP) + 1))
    if n > WARMUP:
        t += base * math.sqrt(TIMESCALE) * 2 * (math.sqrt(n) - math.sqrt(WARMUP))
    return t


def main():
    rungs = []
    p = f"{OUTDIR}/scaling_ladder.json"
    if os.path.exists(p):
        rungs = json.load(open(p))

    steps = list(range(1, TOTAL_STEPS + 1, 5))
    xs = [s * Q_PER_IMG for s in steps]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for (name, base), col in zip(GROUPS.items(), ["#dc2626", "#ea580c", "#a16207"]):
        ax.plot(xs, [lr_at(base, s) for s in steps], lw=2, color=col, label=f"LR — {name}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MMR training samples seen (log)")
    ax.set_ylabel("learning rate (log)")
    ax.grid(True, which="major", alpha=.2)
    ax.spines[["top"]].set_visible(False)

    if rungs:
        ax2 = ax.twinx()
        ax2.plot([r["samples"] for r in rungs], [r["gIoU"] for r in rungs],
                 "o-", color="#2563eb", lw=2.4, ms=7, label="MMR val gIoU", zorder=5)
        ax2.set_ylabel("MMR val gIoU (×100)", color="#2563eb")
        ax2.tick_params(axis="y", labelcolor="#2563eb")
        ax2.set_ylim(0, 45)
        ax2.spines[["top"]].set_visible(False)
        for r in rungs:
            ax2.annotate(f"{r['gIoU']:.1f}", (r["samples"], r["gIoU"]),
                         textcoords="offset points", xytext=(0, 9), ha="center",
                         fontsize=8, color="#2563eb", fontweight="bold")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False, fontsize=9)
    else:
        ax.legend(loc="lower left", frameon=False, fontsize=9)

    ax.set_title("LR decays as 1/√(data) — every doubling of data multiplies it by 0.707",
                 fontweight="bold")
    fig.tight_layout()
    out = f"{OUTDIR}/lr_schedule.png"
    fig.savefig(out, dpi=155)
    print("wrote", out)

    tot = mass(1.0, TOTAL_STEPS)
    print(f"\n{'rung':>6} {'step':>7} {'samples':>9} {'transformer LR':>15} "
          f"{'% of peak':>10} {'cum LR-mass':>12}")
    for name, st in [("0.25%", 114), ("0.5%", 228), ("1%", 456), ("2%", 912),
                     ("4%", 1825), ("8%", 3649), ("16%", 7299), ("32%", 14598),
                     ("64%", 29196), ("100%", TOTAL_STEPS)]:
        lr = lr_at(8e-5, st)
        print(f"{name:>6} {st:>7} {round(st*Q_PER_IMG):>9} {lr:>15.3e} "
              f"{100*lr/8e-5:>9.1f}% {100*mass(1.0, st)/tot:>11.1f}%")


if __name__ == "__main__":
    main()
