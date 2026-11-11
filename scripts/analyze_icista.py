# -*- coding: utf-8 -*-
"""Statistics, Table I and Fig. 1 of the ICISTA paper, from saved runs (no model call).

Every folder runs_*/ with a results.json is read, except dry runs, speed tests and FAILED folders.

    py -3.13 analyze_icista.py      -> statistics_icista.json, table_rows.tex, ../paper/fig1.pdf
"""
import glob, json, math, os
from scipy.stats import fisher_exact

SKIP = ("FAILED", "runs_dry", "speedtest")
REF = "gpt-4o"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load():
    out = {}
    for root in sorted(glob.glob("runs_*")):
        if any(s in root for s in SKIP) or not os.path.exists(f"{root}/results.json"):
            continue
        r = json.load(open(f"{root}/results.json", encoding="utf-8"))
        model = r["meta"]["model"]
        runs = r["runs"]
        secs = [x["cost"].get("seconds") for x in runs if isinstance(x.get("cost"), dict) and x["cost"].get("seconds")]
        toks = [x["cost"].get("completion_tokens") for x in runs if isinstance(x.get("cost"), dict) and x["cost"].get("completion_tokens")]
        hints = {}
        for x in runs:
            h = (x.get("pytest") or {}).get("failure_hint")
            if h and not (x.get("pytest") or {}).get("all_pass"):
                hints[h] = hints.get(h, 0) + 1
        n = len(runs)
        k = sum(1 for x in runs if (x.get("pytest") or {}).get("all_pass"))
        out[model] = {
            "folder": root, "n": n, "all_pass": k, "pass_rate": k / n if n else None,
            "wilson95": wilson(k, n),
            "rvh": sum(1 for x in runs if x.get("is_rvh")), "iih": sum(1 for x in runs if x.get("is_iih")),
            "truncated": sum(1 for x in runs if isinstance(x.get("cost"), dict) and x["cost"].get("truncated")),
            "yaml_ok": sum(1 for x in runs if (x.get("extraction") or {}).get("yaml_parses")),
            "failure_hints": hints,
            "mean_seconds": round(sum(secs) / len(secs), 1) if secs else None,
            "median_seconds": round(sorted(secs)[len(secs) // 2], 1) if secs else None,
            "mean_completion_tokens": round(sum(toks) / len(toks)) if toks else None,
        }
    if REF in out:
        a, n1 = out[REF]["all_pass"], out[REF]["n"]
        for m, s in out.items():
            if m != REF:
                s["fisher_p_vs_gpt4o"] = round(fisher_exact([[s["all_pass"], s["n"] - s["all_pass"]], [a, n1 - a]])[1], 3)
    return out


def main():
    st = load()
    json.dump(st, open("statistics_icista.json", "w", encoding="utf-8"), indent=2)
    rows = []
    for m, s in st.items():
        lo, hi = s["wilson95"]
        rows.append(f"{m} & {s['all_pass']}/{s['n']} & [{lo:.2f}, {hi:.2f}] & {s['rvh']} & {s['iih']} & "
                    f"{s['mean_completion_tokens'] or '--'} & {s.get('median_seconds') or '--'} & "
                    f"{s.get('fisher_p_vs_gpt4o', '--')} \\\\")
        print(f"{m:22s} pass {s['all_pass']:2d}/{s['n']:2d}  CI [{lo:.2f},{hi:.2f}]  RVH {s['rvh']}  IIH {s['iih']}  "
              f"trunc {s['truncated']}  yaml {s['yaml_ok']}/{s['n']}  {s.get('median_seconds')} s (median)  "
              f"p={s.get('fisher_p_vs_gpt4o', '-')}  fails {s['failure_hints']}")
    open("table_rows.tex", "w", encoding="utf-8").write("\n".join(rows) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ms = list(st)
        fig, ax = plt.subplots(figsize=(3.4, 2.1))   # IEEE single column: 3.5 in
        for i, m in enumerate(ms):
            s = st[m]; lo, hi = s["wilson95"]; p = s["pass_rate"]
            ax.bar(i, 100 * p, 0.6, color="#eb6834" if m == REF else "#2a78d6", edgecolor="white")
            ax.errorbar(i, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color="#222222", capsize=2, lw=0.8)
            ax.text(i, 100 * hi + 2, f"{s['all_pass']}/{s['n']}", ha="center", fontsize=6.5)
        ax.set_xticks(range(len(ms)), ms, rotation=25, ha="right", fontsize=6.5)
        ax.set_ylabel("Generations with all tests passing (%)", fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5); ax.set_ylim(0, 110)
        ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        fig.tight_layout()
        os.makedirs("../paper", exist_ok=True)
        fig.savefig("fig_passrate.pdf"); fig.savefig("fig_passrate.png", dpi=200)
        print("saved fig_passrate.pdf (exploratory; the paper figure is made by make_figure_icista.py)")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
