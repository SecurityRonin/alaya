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

fc_color = "#4A90D9"
rag_color = "#E85D5D"
bar_width = 0.35

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.8),
                                         gridspec_kw={"width_ratios": [1, 1.8],
                                                      "wspace": 0.35})

# ===== LEFT PANEL: Retrieval Crossover (X pattern) =====
left_x = np.array([0, 1])
fc_vals = [g[1] for g in groups]
rag_vals = [g[2] for g in groups]
labels_left = [g[0] for g in groups]

bars1 = ax_left.bar(left_x - bar_width/2, fc_vals, bar_width,
                    color=fc_color, label="Full-context", zorder=3, alpha=0.85)
bars2 = ax_left.bar(left_x + bar_width/2, rag_vals, bar_width,
                    color=rag_color, label="Naive RAG", zorder=3, alpha=0.85)

# The X: connect FC tops and RAG tops with lines — BEHIND bars
ax_left.plot(left_x, fc_vals, color=fc_color, linewidth=2.5, zorder=2,
             marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5,
             linestyle="--", alpha=0.7)
ax_left.plot(left_x, rag_vals, color=rag_color, linewidth=2.5, zorder=2,
             marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5,
             linestyle="--", alpha=0.7)

# Value labels on bars — white background so lines don't obscure
for bar in bars1:
    ax_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=fc_color, zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))
for bar in bars2:
    ax_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=rag_color, zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))

# "X" label at intersection point
# Lines: FC goes 61.4 -> 46.2, RAG goes 26.0 -> 54.6
# Intersection: solve 61.4 + t*(46.2-61.4) = 26.0 + t*(54.6-26.0)
# 61.4 - 15.2t = 26.0 + 28.6t  =>  35.4 = 43.8t  =>  t = 0.808
t = 35.4 / 43.8
cross_x = 0 + t * 1
cross_y = 61.4 + t * (46.2 - 61.4)
ax_left.annotate("crossover", xy=(cross_x, cross_y + 3), fontsize=7.5,
                 ha="center", color="#555", fontstyle="italic",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                           edgecolor="#CCC", linewidth=0.6, alpha=0.9))

ax_left.set_xticks(left_x)
ax_left.set_xticklabels(labels_left, fontsize=9.5)
ax_left.set_ylabel("Accuracy (%)", fontsize=10)
ax_left.set_ylim(0, 78)
ax_left.set_yticks([0, 20, 40, 60])
ax_left.set_title("Retrieval Benchmarks", fontsize=10, fontweight="bold",
                  color="#444", pad=10)
ax_left.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
ax_left.set_axisbelow(True)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)
ax_left.legend(loc="upper center", framealpha=0.9, fontsize=8.5, ncol=2)

# ===== RIGHT PANEL: MAB Lifecycle Competencies =====
right_x = np.arange(len(mab))
fc_vals_r = [g[1] for g in mab]
rag_vals_r = [g[2] for g in mab]
labels_right = [g[0] for g in mab]

bars3 = ax_right.bar(right_x - bar_width/2, fc_vals_r, bar_width,
                     color=fc_color, label="Full-context", zorder=3, alpha=0.85)
bars4 = ax_right.bar(right_x + bar_width/2, rag_vals_r, bar_width,
                     color=rag_color, label="Naive RAG", zorder=3, alpha=0.85)

for bar in bars3:
    ax_right.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                  f"{bar.get_height():.1f}", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color=fc_color)
for bar in bars4:
    ax_right.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                  f"{bar.get_height():.1f}", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color=rag_color)

# TTL gap annotation — draw bracket between the two bars
ttl_x = right_x[1]
ax_right.annotate("+42pp", xy=(ttl_x, 65), fontsize=8.5, ha="center",
                  fontweight="bold", color="#C0392B",
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="#FADBD8",
                            edgecolor="#C0392B", linewidth=0.8))

# CR annotation
cr_x = right_x[3]
ax_right.annotate("~chance", xy=(cr_x, 53), fontsize=7.5, ha="center",
                  fontstyle="italic", color="#888")

ax_right.set_xticks(right_x)
ax_right.set_xticklabels(labels_right, fontsize=9)
ax_right.set_ylim(0, 108)
ax_right.set_yticks([0, 20, 40, 60, 80, 100])
ax_right.set_title("MAB Lifecycle Competencies", fontsize=10,
                   fontweight="bold", color="#444", pad=10)
ax_right.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
ax_right.set_axisbelow(True)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

out_path = os.path.join(os.path.dirname(__file__), "benchmark-chart.svg")
fig.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
print(f"Saved to {out_path}")

fsize = os.path.getsize(out_path)
print(f"File size: {fsize/1024:.1f} KB")

with open(out_path) as f:
    for line in f:
        if "viewBox" in line:
            parts = line.split("viewBox=")[1].split('"')[1].split()
            print(f"SVG viewBox: {parts[2]}x{parts[3]}pt")
            break
