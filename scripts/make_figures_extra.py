# -*- coding: utf-8 -*-
"""Extra figures of the ICISTA paper.

    py -3.13 make_figures_extra.py   -> ../figures/fig_method.pdf, ../figures/fig_cost.pdf
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 7})
INK, MUTED = "#222222", "#555555"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# ---------------------------------------------------------------- Fig. method (single column)
fig, ax = plt.subplots(figsize=(3.45, 1.55))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")


def box(x, y, w, h, text, col, fs=5.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=col + "22", ec=col, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, mutation_scale=6))


box(1, 15, 14, 14, "M2 prompt\nT = 0.3", MUTED)
box(22, 26, 19, 15, "4 local models\nOllama, CPU\n20 runs each", BLUE)
box(22, 3, 19, 15, "GPT-4o (API)\n10 runs,\nre-executed", MUTED)
box(48, 15, 14, 14, "Extract\n3 files", MUTED)
box(70, 26, 29, 15, "pytest in the pinned\nenvironment\n→ pass / failure type", GREEN)
box(70, 3, 29, 15, "Static detectors\nRVH / IIH", GOLD)
arrow(16, 24, 21, 32); arrow(16, 20, 21, 11)
arrow(42, 33, 47, 25); arrow(42, 10, 47, 19)
arrow(63, 25, 69, 32); arrow(63, 19, 69, 11)
fig.savefig("../figures/fig_method.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("../figures/fig_method.png", dpi=200, bbox_inches="tight", pad_inches=0.02)

# ---------------------------------------------------------------- Fig. cost vs quality (single column)
S = json.load(open("statistics_icista.json", encoding="utf-8"))
M = [("qwen2.5-coder:3b", "Qwen2.5-Coder 3B", 1.9), ("qwen2.5-coder:7b", "Qwen2.5-Coder 7B", 4.7),
     ("llama3.1:8b", "Llama 3.1 8B", 4.9), ("mistral:7b", "Mistral 7B", 4.4)]
OFF = {"qwen2.5-coder:3b": (-6, 9), "qwen2.5-coder:7b": (6, 9), "llama3.1:8b": (-6, 0), "mistral:7b": (-7, 11)}
HA = {"llama3.1:8b": "right", "qwen2.5-coder:3b": "right", "mistral:7b": "right"}
fig, ax = plt.subplots(figsize=(3.45, 1.85))
g = S["gpt-4o"]
ax.axhspan(100 * g["wilson95"][0], 100 * g["wilson95"][1], color="#999999", alpha=0.15, lw=0)
ax.axhline(100 * g["pass_rate"], color=MUTED, lw=0.8, ls="--")
ax.text(301, 100 * g["pass_rate"] + 1.5, "GPT-4o (API): 3/10\n95% CI shaded", fontsize=5.8, color=MUTED, ha="right", va="bottom")
for key, name, gb in M:
    s = S[key]; x = s["median_seconds"]; p = 100 * s["pass_rate"]
    lo, hi = 100 * s["wilson95"][0], 100 * s["wilson95"][1]
    col = BLUE if key == "llama3.1:8b" else ORANGE
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=3 + gb * 0.9, color=col, ecolor=col,
                elinewidth=0.8, capsize=2, mec="white", mew=0.6, zorder=3)
    dx, dy = OFF[key]
    ax.annotate(name + f"\n{s['all_pass']}/20, {gb} GB", (x, p), xytext=(dx, dy), textcoords="offset points",
                fontsize=5.4, color=INK, ha=HA.get(key, "left"), va="center")
ax.set_xlim(100, 305); ax.set_ylim(-3, 92)
ax.set_xlabel("Median time per generation on the CPU laptop (s)", fontsize=6.5)
ax.set_ylabel("Generations passing (%)", fontsize=6.5)
ax.tick_params(labelsize=6.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("../figures/fig_cost.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("../figures/fig_cost.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved fig_method + fig_cost")
