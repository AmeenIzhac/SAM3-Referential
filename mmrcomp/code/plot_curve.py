#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4]
xt = ["base\n(0)", "1k", "2k", "4k", "8k"]
giou = [0.6, 25.8, 26.38, 25.54, 23.54]
ciou = [1.5, 24.04, 24.51, 26.21, 26.16]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, giou, "o-", color="#2563eb", lw=2.4, ms=8, label="gIoU", zorder=5)
ax.plot(x, ciou, "s-", color="#0d9488", lw=2.4, ms=7, label="cIoU", zorder=5)
for xi, g in zip(x, giou):
    ax.annotate(f"{g:.1f}", (xi, g), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color="#2563eb", fontweight="bold")

# reference lines
for y, txt, col in [(25.2, "M²SA-7B (question-only)", "#b45309"),
                    (11.1, "SAM3 reasonseg fine-tune", "#9333ea")]:
    ax.axhline(y, ls="--", lw=1.3, color=col, alpha=.8)
    ax.text(4.05, y, f" {txt} = {y}", va="center", fontsize=8.5, color=col)

ax.set_xticks(x); ax.set_xticklabels(xt)
ax.set_xlabel("MMR training samples seen (one continuously-trained model)")
ax.set_ylabel("MMR val score (×100)")
ax.set_title("SAM3 on MMR — data-scaling curve (question prompt)", fontweight="bold")
ax.set_ylim(0, 40); ax.set_xlim(-0.2, 4.9)
ax.grid(True, axis="y", alpha=.25); ax.legend(loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("/workspace/reasonseg/mmrcomp/scaling_curve.png", dpi=130)
print("saved scaling_curve.png")
