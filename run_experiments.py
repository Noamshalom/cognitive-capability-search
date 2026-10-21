"""Run every experiment in the paper and write results/results.json.

    python run_experiments.py            # full run (~15-25 min on 2 CPU cores)
    python run_experiments.py --quick    # smoke test (~1 min)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from ccs import core
from ccs.learner import develop
from ccs.search import Evaluator, genetic_search, random_search, serialise, deserialise
from ccs.world import World

ALL_METHODS = ("ccs-adaptive", "ccs-global", "ccs-local", "random-adaptive")
METHODS = ALL_METHODS

HAND = {  # the "obvious" interface a human designer might pick first
    "carried": np.ones(8, dtype=int), "code": "linear", "noise": 0.05,
    "w_rec": 1.0, "w_align": 0.0, "cond": "both", "lr": 3e-3,
}


def _init(metric):
    core.METRIC = metric


def setup(world_seed):
    world = World(world_seed)
    L0 = develop(world, seed=world_seed)
    D = core.EvalData(world, seed=world_seed + 7)
    return world, L0, D, core.innate_r2(L0, D)


def job(args):
    world_seed, method, pop, gens = args
    t = time.time()
    world, L0, D, base = setup(world_seed)
    ev = Evaluator(world, L0, D, base, seed=world_seed)
    rng = np.random.default_rng(1000 * world_seed + ALL_METHODS.index(method))
    regime = method.split("-")[1]
    if method.startswith("ccs"):
        best, _, _ = genetic_search(ev, regime, rng, pop=pop, gens=gens)
    else:
        best, _, _ = random_search(ev, regime, rng, budget=pop * gens)
    print(f"  world {world_seed} {method:16s} done in {time.time() - t:5.0f}s  "
          f"best={ev.cache[core.genome_key(best)]['score']:+.3f}", flush=True)
    return {"world": world_seed, "method": method, "best": serialise(best), "log": ev.log}


def reevaluate(args):
    """Re-test a genome with fresh inner-loop seeds + the held-out battery."""
    world_seed, name, gdict = args
    world, L0, D, base = setup(world_seed)
    battery = core.Battery()
    if gdict is None:  # no interface at all
        b = battery.score(L0, world, None, D)
        return {"world": world_seed, "name": name, "battery": b.tolist(), "runs": []}
    g = deserialise(gdict)
    runs, bat = [], []
    for s in (101, 202, 303):
        r = core.evaluate(g, world, L0, D, base, seed=s, keep=True)
        bat.append(battery.score(r.pop("learner"), world, g, D))
        runs.append(r)
    return {"world": world_seed, "name": name, "genome": gdict, "desc": core.describe(g),
            "runs": runs, "battery": np.mean(bat, 0).tolist()}


def main():
    global METHODS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--first-world", type=int, default=0)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--metric", choices=("v1", "v2"), default="v1",
                    help="v1: interference with channel silent (original). v2: also with channel on.")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    core.METRIC = a.metric
    METHODS = tuple(a.methods.split(","))
    pop, gens, worlds = (8, 3, 1) if a.quick else (20, 12, a.worlds)
    world_ids = list(range(a.first_world, a.first_world + worlds))
    os.makedirs("results", exist_ok=True)

    print(f"[1/2] search: worlds {world_ids} x {len(METHODS)} methods x {pop * gens} evaluations, metric {a.metric}")
    jobs = [(w, m, pop, gens) for w in world_ids for m in METHODS]
    with Pool(a.procs, initializer=_init, initargs=(a.metric,)) as pool:
        searches = pool.map(job, jobs)

    print("[2/2] re-evaluating winners and baselines on fresh seeds + held-out battery")
    re_jobs = []
    for w in world_ids:
        re_jobs.append((w, "innate-only", None))
        for regime, phases in (("hand-global", [((1, 1, 1), 600), ((0, 0, 0), 0)]),
                               ("hand-local", [((0, 1, 0), 600), ((0, 0, 0), 0)])):
            re_jobs.append((w, regime, serialise({**HAND, "phases": phases})))
        for s in searches:
            if s["world"] == w:
                re_jobs.append((w, s["method"], s["best"]))
    with Pool(a.procs, initializer=_init, initargs=(a.metric,)) as pool:
        finals = pool.map(reevaluate, re_jobs)

    out = {"config": {"pop": pop, "gens": gens, "worlds": world_ids, "beta": core.BETA,
                      "gamma": core.GAMMA, "n_shot": core.N_SHOT, "metric": a.metric, "methods": METHODS},
           "hidden_task_mask": core.Battery().hidden_mask().tolist(),
           "searches": searches, "finals": finals}
    path = a.out or ("results/results_quick.json" if a.quick else "results/results.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
