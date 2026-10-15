"""Outer loop: search over environments (genomes) for what the learner can become."""
from __future__ import annotations

import numpy as np

from . import core


class Evaluator:
    """Caches evaluations so a genome is never trained twice in one run."""
    def __init__(self, world, L0, D, base_innate, seed: int = 0):
        self.world, self.L0, self.D, self.base = world, L0, D, base_innate
        self.seed, self.cache, self.log = seed, {}, []

    def __call__(self, g):
        k = core.genome_key(g)
        if k not in self.cache:
            r = core.evaluate(g, self.world, self.L0, self.D, self.base, seed=self.seed)
            self.cache[k] = r
        r = self.cache[k]
        self.log.append({"key": k, "genome": serialise(g), **{x: r[x] for x in
                         ("score", "gain", "interference", "innate", "cost")}})
        return r["score"]


def serialise(g):
    return {**g, "carried": [int(x) for x in g["carried"]],
            "phases": [[list(m), s] for m, s in g["phases"]]}


def deserialise(d):
    return {**d, "carried": np.array(d["carried"], dtype=int),
            "phases": [(tuple(m), s) for m, s in d["phases"]]}


def genetic_search(ev: Evaluator, regime: str, rng, pop: int = 20, gens: int = 12, elite: int = 2):
    P = [core.random_genome(rng, regime) for _ in range(pop)]
    F = [ev(g) for g in P]
    for _ in range(gens - 1):
        order = np.argsort(F)[::-1]
        new = [P[i] for i in order[:elite]]
        while len(new) < pop:
            def pick():
                i, j = rng.integers(pop, size=2)
                return P[i] if F[i] > F[j] else P[j]
            child = core.crossover(pick(), pick(), rng) if rng.random() < 0.5 else pick()
            new.append(core.mutate(child, rng, regime))
        P = new
        F = [ev(g) for g in P]
    best = int(np.argmax(F))
    return P[best], P, F


def random_search(ev: Evaluator, regime: str, rng, budget: int = 240):
    G = [core.random_genome(rng, regime) for _ in range(budget)]
    F = [ev(g) for g in G]
    best = int(np.argmax(F))
    return G[best], G, F
