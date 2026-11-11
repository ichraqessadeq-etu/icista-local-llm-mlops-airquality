# -*- coding: utf-8 -*-
"""Fig. 1 of the ICISTA paper: outcome of every generation per model (pass or failure category).

    py -3.13 classify_failures.py && py -3.13 make_figure_icista.py   -> ../figures/fig1.pdf
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = json.load(open("failure_categories.json", encoding="utf-8"))
ORDER = ["gpt-4o", "qwen2.5-coder:3b", "qwen2.5-coder:7b", "llama3.1:8b", "mistral:7b"]
LABEL = {"gpt-4o": "GPT-4o (API)", "qwen2.5-coder:3b": "Qwen2.5-Coder 3B", "qwen2.5-coder:7b": "Qwen2.5-Coder 7B",
         "llama3.1:8b": "Llama 3.1 8B", "mistral:7b": "Mistral 7B"}
CATS = [("pass", "All tests pass", "#2a78d6"), ("missing_name", "Missing import/name", "#eb6834"),
        ("mlflow_api", "Non-existent MLflow API", "#1baf7a"), ("data_logic", "Data/logic error", "#eda100"),
        ("collection", "Test file not collected", "#e87ba4")]
INK = "#222222"

plt.rcParams.update({"font.family": "serif", "font.size": 7})
fig, ax = plt.subplots(figsize=(3.45, 2.05))          # IEEE single column (3.5 in), printed at 1:1
for y, m in enumerate(ORDER):
    n = sum(C[m].values()); left = 0.0
    for key, name, col in CATS:
        k = C[m][key]
        if not k:
            continue
        w = 100 * k / n
        ax.barh(y, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.0,
                label=name if y == 0 or key not in [c[0] for c in CATS if C[ORDER[0]][c[0]]] else None)
        if w >= 9:
            ax.text(left + w / 2, y, str(k), ha="center", va="center", fontsize=6.5, color="white" if key == "pass" else INK)
        left += w
    ax.text(101, y, f"n={n}", va="center", fontsize=6, color="#555555")
ax.set_yticks(range(len(ORDER)), [LABEL[m] for m in ORDER], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Generations (%)", fontsize=6.5)
ax.tick_params(axis="x", labelsize=6.5)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CATS]
ax.legend(handles, [n for _, n, _ in CATS], frameon=False, fontsize=5.8, ncol=2, loc="upper center",
          bbox_to_anchor=(0.45, -0.25), handlelength=1.0, columnspacing=0.8)
fig.tight_layout()
fig.savefig("../figures/fig1.pdf"); fig.savefig("../figures/fig1.png", dpi=200)
print("saved ../figures/fig1.pdf")
