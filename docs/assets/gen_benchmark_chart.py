#!/usr/bin/env python3
"""Generate benchmark comparison SVG for README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Data
groups = [
    # (label, full_context, naive_rag)
    ("LoCoMo", 61.4, 26.0),
    ("LongMemEval", 46.2, 54.6),
]
mab = [
    ("AR\n(Retrieval)", 94.0, 90.4),
    ("TTL\n(Learning)", 86.0, 44.0),
    ("LRU\n(Underst.)", 82.4, 67.6),
    ("CR\n(Forgetting)", 50.0, 41.0),
]

fig, ax = plt.subplots(figsize=(10, 5))

# Colors
fc_color = "#4A90D9"
rag_color = "#E85D5D"
bar_width = 0.35

# -- Left cluster: LoCoMo + LongMemEval --
left_x = np.array([0, 1])
fc_vals_left = [g[1] for g in groups]
rag_vals_left = [g[2] for g in groups]
labels_left = [g[0] for g in groups]

bars1 = ax.bar(left_x - bar_width/2, fc_vals_left, bar_width,
               color=fc_color, label="Full-context", zorder=3)
bars2 = ax.bar(left_x + bar_width/2, rag_vals_left, bar_width,
               color=rag_color, label="Naive RAG", zorder=3)

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=fc_color)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=rag_color)

# Crossover arrow annotation
ax.annotate("", xy=(1 + bar_width/2, 54.6), xytext=(0 - bar_width/2, 61.4),
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.2,
                            connectionstyle="arc3,rad=0.15"))
ax.text(0.5, 66, "crossover", ha="center", fontsize=7.5, color="#666",
        fontstyle="italic")

# -- Right cluster: MAB competencies (offset by gap) --
gap = 1.5  # visual gap between clusters
right_x = np.array([0, 1, 2, 3]) + len(groups) + gap
fc_vals_right = [g[1] for g in mab]
rag_vals_right = [g[2] for g in mab]
labels_right = [g[0] for g in mab]

bars3 = ax.bar(right_x - bar_width/2, fc_vals_right, bar_width,
               color=fc_color, zorder=3)
bars4 = ax.bar(right_x + bar_width/2, rag_vals_right, bar_width,
               color=rag_color, zorder=3)

for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=fc_color)
for bar in bars4:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=rag_color)

# TTL gap annotation
ttl_x = right_x[1]
ax.annotate("+42pp", xy=(ttl_x, 65), fontsize=8, ha="center",
            fontweight="bold", color="#C0392B",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FADBD8",
                      edgecolor="#C0392B", linewidth=0.8))

# CR annotation
cr_x = right_x[3]
ax.annotate("~chance", xy=(cr_x, 53), fontsize=7, ha="center",
            fontstyle="italic", color="#888")

# Axis setup
all_x = np.concatenate([left_x, right_x])
all_labels = labels_left + labels_right
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_ylim(0, 108)
ax.set_yticks([0, 20, 40, 60, 80, 100])

# Grid
ax.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
ax.set_axisbelow(True)

# Group labels — use data coordinates, placed just below the x-axis
ax.text(0.5, -12, "Retrieval Benchmarks", ha="center", fontsize=9,
        fontweight="bold", color="#555", clip_on=False)
ax.text(np.mean(right_x), -12, "MAB Lifecycle Competencies", ha="center",
        fontsize=9, fontweight="bold", color="#555", clip_on=False)

# Divider line
mid = (left_x[-1] + right_x[0]) / 2
ax.axvline(x=mid, color="#CCC", linestyle=":", linewidth=1, zorder=1)

# Legend
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

# Cleanup
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.subplots_adjust(bottom=0.18)
out_path = os.path.join(os.path.dirname(__file__), "benchmark-chart.svg")
fig.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
print(f"Saved to {out_path}")

# Sanity check
fsize = os.path.getsize(out_path)
print(f"File size: {fsize/1024:.1f} KB")

# Check viewBox height isn't absurd
with open(out_path) as f:
    line4 = [next(f) for _ in range(4)][-1]
    if "viewBox" in line4:
        parts = line4.split("viewBox=")[1].split('"')[1].split()
        h = float(parts[3])
        print(f"SVG viewBox height: {h:.1f}pt")
        if h > 600:
            print("WARNING: viewBox height too large!")
