"""Robustness summary across worlds: confidence intervals, paired tests, winner genomes.

    python analyze_robustness.py results/results_v2.json [results/results.json results/results_v1_extra.json ...]

The first file is the main one (metric v2, 10 worlds). Extra files are merged for the
v1 10-world comparison. Writes results/robustness.md.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), 0.0
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def sign_test(d):
    """Two-sided exact sign test on paired differences (ignores exact zeros)."""
    from math import comb
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    p = sum(comb(n, j) for j in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return min(p, 1.0), k, n


def finals_by(R):
    out = defaultdict(dict)
    for f in R["finals"]:
        out[f["name"]][f["world"]] = f
    return out


def summarise(R, label, lines):
    hidden = np.array(R["hidden_task_mask"])
    F = finals_by(R)
    names = [n for n in ("hand-global", "hand-local", "random-adaptive", "ccs-global", "ccs-adaptive") if n in F]
    worlds = sorted(F["ccs-adaptive"])
    lines.append(f"## {label}: {len(worlds)} worlds, metric {R['config'].get('metric','v1')}\n")
    lines.append("| condition | score (mean ± 95% CI) | gain | interference | innate, channel on | battery hidden |")
    lines.append("|---|---|---|---|---|---|")
    stats = {}
    for n in names:
        per_world = {k: [] for k in ("score", "gain", "interference", "innate_on", "hid")}
        for w in worlds:
            f = F[n][w]
            for k in ("score", "gain", "interference", "innate_on"):
                per_world[k].append(np.mean([r.get(k, np.nan) for r in f["runs"]]))
            per_world["hid"].append(np.array(f["battery"])[hidden].mean())
        stats[n] = per_world
        m, c = ci95(per_world["score"])
        lines.append(f"| {n} | {m:+.3f} ± {c:.3f} | {np.mean(per_world['gain']):.3f} | "
                     f"{np.mean(per_world['interference']):.3f} | {np.nanmean(per_world['innate_on']):.3f} | "
                     f"{np.mean(per_world['hid']):.3f} |")
    lines.append("")
    if "random-adaptive" in stats:
        for n in ("ccs-adaptive", "ccs-global"):
            d = np.array(stats[n]["score"]) - np.array(stats["random-adaptive"]["score"])
            p, k, nn = sign_test(d)
            m, c = ci95(d)
            lines.append(f"- {n} − random: Δscore = {m:+.3f} ± {c:.3f}; CCS wins in {k}/{nn} worlds "
                         f"(sign test p = {p:.3f})")
    d = np.array(stats["ccs-adaptive"]["score"]) - np.array(stats["ccs-global"]["score"])
    p, k, nn = sign_test(d)
    m, c = ci95(d)
    lines.append(f"- adaptive − global: Δscore = {m:+.3f} ± {c:.3f}; adaptive wins in {k}/{nn} worlds (p = {p:.3f})")
    lines.append("")
    # what the winners chose
    for n in ("ccs-global", "ccs-adaptive"):
        gs = [F[n][w]["genome"] for w in worlds]
        cond = Counter(g["cond"] for g in gs); code = Counter(g["code"] for g in gs)
        obj = Counter("rec only" if g["w_align"] == 0 else ("align only" if g["w_rec"] == 0 else "both") for g in gs)
        ncar = np.mean([sum(g["carried"]) for g in gs])
        opens_r1 = sum(any(m[0] and s for m, s in g["phases"]) for g in gs)
        steps = np.mean([sum(s for m, s in g["phases"] if any(m)) for g in gs])
        lines.append(f"- {n} winners: input {dict(cond)}; code {dict(code)}; objective {dict(obj)}; "
                     f"carries {ncar:.1f} latents on average; opens R1 in {opens_r1}/{len(gs)}; "
                     f"mean plastic steps {steps:.0f}")
    lines.append("")
    return stats


def main():
    main_path, extra = sys.argv[1], sys.argv[2:]
    R2 = json.load(open(main_path))
    lines = ["# Robustness analysis\n"]
    s2 = summarise(R2, "Main robustness run", lines)
    if extra:
        merged = None
        for p in extra:
            R = json.load(open(p))
            if merged is None:
                merged = R
            else:
                merged["finals"] += R["finals"]; merged["searches"] += R["searches"]
        # keep only methods present in every world
        F = finals_by(merged); worlds = sorted(F["ccs-adaptive"])
        merged["finals"] = [f for f in merged["finals"] if f["world"] in worlds and
                            (f["name"] in ("ccs-adaptive", "ccs-global", "hand-global", "hand-local", "innate-only")
                             or len(F[f["name"]]) == len(worlds))]
        s1 = summarise(merged, "Original metric (v1), all worlds", lines)
        # v1 -> v2 comparison on shared worlds
        shared = sorted(set(finals_by(R2)["ccs-adaptive"]) & set(worlds))
        lines.append(f"## v1 → v2 on the {len(shared)} shared worlds\n")
        for n in ("ccs-global", "ccs-adaptive"):
            a = [np.mean([r['innate_on'] for r in finals_by(merged)[n][w]['runs']]) for w in shared]
            b = [np.mean([r['innate_on'] for r in finals_by(R2)[n][w]['runs']]) for w in shared]
            ga = [np.mean([r['gain'] for r in finals_by(merged)[n][w]['runs']]) for w in shared]
            gb = [np.mean([r['gain'] for r in finals_by(R2)[n][w]['runs']]) for w in shared]
            lines.append(f"- {n}: innate R² with channel ON {np.mean(a):.3f} → {np.mean(b):.3f}; "
                         f"gain {np.mean(ga):.3f} → {np.mean(gb):.3f}")
    out = "\n".join(lines)
    open("results/robustness.md", "w").write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
