"""Make every figure and the summary table from results/results.json.

    python make_figures.py [results/results.json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference categorical palette (fixed order), text tokens, surfaces
C = {"ccs-adaptive": "#2a78d6", "ccs-global": "#eb6834", "ccs-local": "#1baf7a",
     "random-adaptive": "#eda100"}
GRAY, INK, INK2, GRID = "#9b9a94", "#0b0b0b", "#52514e", "#e6e5e0"
LABEL = {"ccs-adaptive": "CCS, adaptive plasticity", "ccs-global": "CCS, global plasticity",
         "ccs-local": "CCS, local plasticity (Rc only)", "random-adaptive": "Random search, adaptive",
         "innate-only": "No interface", "hand-global": "Hand-designed, global",
         "hand-local": "Hand-designed, local"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(path):
    with open(path) as f:
        return json.load(f)


def fig_search(R, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for m in ("ccs-adaptive", "random-adaptive", "ccs-local", "ccs-global"):
        curves = [np.maximum.accumulate([e["score"] for e in s["log"]])
                  for s in R["searches"] if s["method"] == m]
        A = np.array(curves)
        x = np.arange(1, A.shape[1] + 1)
        ax.plot(x, A.mean(0), color=C[m], lw=2, label=LABEL[m])
        ax.fill_between(x, A.min(0), A.max(0), color=C[m], alpha=0.12, lw=0)
    ax.set_xlabel("environments evaluated (inner-loop runs)")
    ax.set_ylabel("best score so far")
    ax.set_ylim(-0.25, 0.65)
    ax.set_title("Outer-loop search vs. random search")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out); plt.close(fig)


def fig_tradeoff(R, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in ("ccs-global", "ccs-local", "ccs-adaptive"):
        pts = np.array([(e["interference"], e["gain"]) for s in R["searches"]
                        if s["method"] == m for e in s["log"]])
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=C[m], alpha=0.35, lw=0, label=LABEL[m])
    for f in R["finals"]:
        if f["name"] in ("ccs-global", "ccs-local", "ccs-adaptive"):
            g = np.mean([r["gain"] for r in f["runs"]]); i = np.mean([r["interference"] for r in f["runs"]])
            ax.scatter(i, g, s=70, color=C[f["name"]], edgecolor="white", lw=1.5, zorder=5)
    ax.set_xlabel("interference (drop in innate-task R²)")
    ax.set_ylabel("capability gain (Δ few-shot decodability)")
    ax.set_title("Every evaluated environment: gain vs. interference")
    ax.legend(loc="lower right", fontsize=8, markerscale=2)
    ax.text(0.98, 0.98, "large dots = selected winners,\nre-tested on fresh seeds",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    fig.savefig(out); plt.close(fig)


def fig_profile(R, out, world=0):
    f = next(f for f in R["finals"] if f["world"] == world and f["name"] == "ccs-adaptive")
    C0 = np.mean([r["C0"] for r in f["runs"]], 0)
    C1 = np.mean([r["C1"] for r in f["runs"]], 0)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.9), sharey=True)
    for ax, M, title in ((axes[0], C0, "Before (developed learner)"),
                         (axes[1], C1, "After CCS-selected environment")):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(8), [f"z{i}" for i in range(8)])
        ax.set_yticks([0, 1], ["channel only", "sense + channel"])
        ax.grid(False)
        ax.axvline(3.5, color=INK2, lw=1, ls="--")
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else INK)
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.9, label="few-shot R²")
    fig.supxlabel("z0–z3 visible to the innate sense  |  z4–z7 hidden from it",
                  fontsize=7.5, color=INK2, x=0.45, y=-0.08)
    fig.savefig(out); plt.close(fig)


ORDER = ["innate-only", "hand-global", "hand-local", "random-adaptive",
         "ccs-global", "ccs-local", "ccs-adaptive"]


def battery_table(R):
    hidden = np.array(R["hidden_task_mask"])
    rows = defaultdict(lambda: defaultdict(list))
    for f in R["finals"]:
        b = np.array(f["battery"])
        rows[f["name"]]["all"].append(b.mean())
        rows[f["name"]]["visible"].append(b[~hidden].mean())
        rows[f["name"]]["hidden"].append(b[hidden].mean())
        if f["runs"]:
            for k in ("gain", "interference", "score"):
                rows[f["name"]][k].append(np.mean([r[k] for r in f["runs"]]))
    return rows


def fig_battery(R, out):
    rows = battery_table(R)
    names = [n for n in ORDER if n in rows]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for off, key, col, lab in ((-w / 2, "visible", "#9fc3ee", "tasks on visible latents only"),
                               (w / 2, "hidden", "#2a78d6", "tasks needing a hidden latent")):
        vals = [np.mean(rows[n][key]) for n in names]
        lo = [np.mean(rows[n][key]) - np.min(rows[n][key]) for n in names]
        hi = [np.max(rows[n][key]) - np.mean(rows[n][key]) for n in names]
        ax.bar(x + off, vals, w - 0.04, color=col, label=lab, yerr=[lo, hi],
               error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
    SHORT = {"innate-only": "No\ninterface", "hand-global": "Hand-made\nglobal",
             "hand-local": "Hand-made\nlocal", "random-adaptive": "Random\nsearch",
             "ccs-global": "CCS\nglobal", "ccs-local": "CCS\nlocal (Rc)", "ccs-adaptive": "CCS\nadaptive"}
    ax.set_xticks(x, [SHORT[n] for n in names], fontsize=7.5)
    ax.set_ylabel("held-out few-shot R²")
    ax.set_ylim(0, 1)
    ax.set_title("Held-out task battery (never seen by the search)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out); plt.close(fig)


def fig_landscape(R, out):
    """Unbiased sample: random-search logs. Mean gain and interference per gene value."""
    E = [e for s in R["searches"] if s["method"] == "random-adaptive" for e in s["log"]]
    groups = {
        "interface code": lambda g: g["code"],
        "opens R2 (association)": lambda g: "yes" if any(m[2] and st for m, st in g["phases"]) else "no",
        "input during learning": lambda g: g["cond"],
        "objective": lambda g: ("both" if g["w_rec"] and g["w_align"] else
                                "reconstruct" if g["w_rec"] else "align"),
    }
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.5), sharey=True)
    for ax, (gname, fn) in zip(axes, groups.items()):
        d = defaultdict(list)
        for e in E:
            d[fn(e["genome"])].append((e["gain"], e["interference"]))
        keys = sorted(d)
        G = [np.mean([v[0] for v in d[k]]) for k in keys]
        I = [np.mean([v[1] for v in d[k]]) for k in keys]
        x = np.arange(len(keys)); w = 0.38
        ax.bar(x - w / 2, G, w - 0.04, color="#2a78d6", label="mean gain")
        ax.bar(x + w / 2, I, w - 0.04, color="#eb6834", label="mean interference")
        ax.set_xticks(x, keys, fontsize=7.5, rotation=20)
        ax.set_title(gname, fontsize=8.5)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Landscape of the environment space (random sample)", fontsize=10,
                 fontweight="bold", color=INK, y=1.04)
    fig.savefig(out); plt.close(fig)


def summary(R):
    rows = battery_table(R)
    lines = ["| condition | score | gain | interference | battery (all) | battery (visible) | battery (hidden) |",
             "|---|---|---|---|---|---|---|"]
    for n in ORDER:
        if n not in rows: continue
        r = rows[n]
        f = lambda k: f"{np.mean(r[k]):+.3f}" if r[k] else "—"
        lines.append(f"| {LABEL[n]} | {f('score')} | {f('gain')} | {f('interference')} | "
                     f"{np.mean(r['all']):.3f} | {np.mean(r['visible']):.3f} | {np.mean(r['hidden']):.3f} |")
    lines.append("")
    lines.append("Selected environments:")
    for f in R["finals"]:
        if "desc" in f and f["name"].startswith(("ccs", "random")):
            sc = np.mean([r["score"] for r in f["runs"]])
            lines.append(f"- world {f['world']} {f['name']}: score {sc:+.3f} — {f['desc']}")
    return "\n".join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
    R = load(path)
    os.makedirs("figures", exist_ok=True)
    fig_search(R, "figures/fig1_search.png")
    fig_tradeoff(R, "figures/fig2_tradeoff.png")
    fig_profile(R, "figures/fig3_profile.png")
    fig_battery(R, "figures/fig4_battery.png")
    fig_landscape(R, "figures/fig5_landscape.png")
    s = summary(R)
    with open("results/summary.md", "w") as f:
        f.write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
