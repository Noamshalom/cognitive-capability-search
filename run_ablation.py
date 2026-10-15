"""Ablation of the search's main discovery: train the new channel while the innate sense is silent.

Takes each world's CCS winners (global and adaptive plasticity), changes ONLY the input
condition (channel -> mixed -> both), and re-evaluates on fresh seeds.

    python run_ablation.py      # needs results/results.json; writes results/ablation.json
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np

from ccs import core
from ccs.search import deserialise
from run_experiments import setup


def run(args):
    world_seed, name, gdict, cond = args
    world, L0, D, base = setup(world_seed)
    g = deserialise(gdict)
    g["cond"] = cond
    runs = [core.evaluate(g, world, L0, D, base, seed=s) for s in (101, 202, 303)]
    return {"world": world_seed, "name": name, "cond": cond,
            **{k: float(np.mean([r[k] for r in runs])) for k in ("gain", "interference", "score")}}


def crosstalk(args):
    """Innate performance with the channel ON - a quantity the search score did NOT include."""
    world_seed, name, gdict = args
    world, L0, D, base = setup(world_seed)
    g = deserialise(gdict)
    on, off = [], []
    for s in (101, 202, 303):
        L = core.acquire(L0, world, g, seed=s)
        on.append(core.innate_r2(L, D, world, g)); off.append(core.innate_r2(L, D))
    return {"world": world_seed, "name": name, "before_on": core.innate_r2(L0, D, world, g),
            "before_off": base, "after_on": float(np.mean(on)), "after_off": float(np.mean(off))}


def main():
    R = json.load(open("results/results.json"))
    jobs = [(f["world"], f["name"], f["genome"], cond)
            for f in R["finals"] if f["name"] in ("ccs-global", "ccs-adaptive")
            for cond in ("channel", "mixed", "both")]
    with Pool() as p:
        out = p.map(run, jobs)
    json.dump(out, open("results/ablation.json", "w"), indent=1)
    print("| winner | input during learning | gain | interference | score |")
    print("|---|---|---|---|---|")
    for name in ("ccs-global", "ccs-adaptive"):
        for cond in ("channel", "mixed", "both"):
            rs = [o for o in out if o["name"] == name and o["cond"] == cond]
            m = lambda k: np.mean([o[k] for o in rs])
            print(f"| {name} | {cond} | {m('gain'):+.3f} | {m('interference'):+.3f} | {m('score'):+.3f} |")

    jobs = [(f["world"], f["name"], f["genome"]) for f in R["finals"] if f.get("genome")]
    with Pool() as p:
        ct = p.map(crosstalk, jobs)
    json.dump(ct, open("results/crosstalk.json", "w"), indent=1)
    print("\n| condition | innate R2 before | after, channel OFF | after, channel ON |")
    print("|---|---|---|---|")
    for name in ("hand-global", "hand-local", "random-adaptive", "ccs-global", "ccs-local", "ccs-adaptive"):
        rs = [o for o in ct if o["name"] == name]
        m = lambda k: np.mean([o[k] for o in rs])
        print(f"| {name} | {m('before_off'):.3f} | {m('after_off'):.3f} | {m('after_on'):.3f} |")


if __name__ == "__main__":
    main()
