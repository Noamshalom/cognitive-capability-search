"""Why does training with the innate sense silent spare the innate readout?

Two measurements on each world's CCS-global winner, trained under its own condition
(channel only) and under the ablated condition (both inputs present):

1. Revert test. After acquisition, put ONE parameter group back to its developed value
   and re-measure the innate task (channel silent). The group whose revert restores the
   most is the group carrying the damage.

2. Perturbation geometry. The change to the association region's pre-activation on
   innate-only input, dz(s) = h1'(s) W2a' + b2' - (h1(s) W2a + b2), splits into a constant
   part (its mean over s) and an input-dependent part. A constant shift is a bias the
   readout can partly absorb; an input-dependent part rewrites the sensory map itself.

    python run_mechanism.py    # needs results/results.json; writes results/mechanism.json
"""
from __future__ import annotations

import json

import numpy as np

from ccs import core
from ccs.search import deserialise
from run_experiments import setup

GROUPS = {"b1": ["b1"], "W1": ["W1"], "W2a": ["W2a"], "b2": ["b2"], "W2b": ["W2b"], "Wc": ["Wc"]}


def analyse(world_seed, gdict, cond, seed=101):
    world, L0, D, base = setup(world_seed)
    g = deserialise(gdict); g["cond"] = cond
    L = core.acquire(L0, world, g, seed=seed)
    full = core.innate_r2(L, D)
    revert = {}
    for name, keys in GROUPS.items():
        Lr = L.clone()
        for k in keys:
            Lr.p[k] = L0.p[k].copy()
        revert[name] = core.innate_r2(Lr, D) - full          # how much reverting restores
    # perturbation geometry on innate-only input
    s = D.s_test
    h1_old = np.tanh(s @ L0.p["W1"] + L0.p["b1"]); h1_new = np.tanh(s @ L.p["W1"] + L.p["b1"])
    pre_old = h1_old @ L0.p["W2a"] + L0.p["b2"]; pre_new = h1_new @ L.p["W2a"] + L.p["b2"]
    dz = pre_new - pre_old
    const = dz.mean(0, keepdims=True); var_part = dz - const
    signal_sd = float(pre_old.std())
    sv = np.linalg.svd(L.p["W2a"] - L0.p["W2a"], compute_uv=False)
    return {"world": world_seed, "cond": cond, "innate_after": full, "innate_before": base,
            "revert_gain": revert,
            "const_rms": float(np.sqrt((const ** 2).mean())) / signal_sd,
            "varying_rms": float(np.sqrt((var_part ** 2).mean())) / signal_sd,
            "dW2a_top_sv_share": float(sv[0] ** 2 / (sv ** 2).sum()),
            "dW2a_rel": float(np.linalg.norm(L.p["W2a"] - L0.p["W2a"]) / np.linalg.norm(L0.p["W2a"])),
            "dW1_rel": float(np.linalg.norm(L.p["W1"] - L0.p["W1"]) / np.linalg.norm(L0.p["W1"]))}


def main():
    R = json.load(open("results/results.json"))
    out = []
    for f in R["finals"]:
        if f["name"] != "ccs-global":
            continue
        for cond in ("channel", "both"):
            out.append(analyse(f["world"], f["genome"], cond))
    json.dump(out, open("results/mechanism.json", "w"), indent=1)
    for cond in ("channel", "both"):
        rs = [o for o in out if o["cond"] == cond]
        m = lambda f: np.mean([f(o) for o in rs])
        print(f"\n== input during learning: {cond} ==  innate after {m(lambda o:o['innate_after']):.3f}")
        print("  restored by reverting:", {k: round(m(lambda o: o["revert_gain"][k]), 3) for k in GROUPS})
        print(f"  pre-activation shift on innate input (in units of signal sd): "
              f"constant {m(lambda o:o['const_rms']):.2f}, input-dependent {m(lambda o:o['varying_rms']):.2f}")
        print(f"  |dW1|/|W1| {m(lambda o:o['dW1_rel']):.2f}   |dW2a|/|W2a| {m(lambda o:o['dW2a_rel']):.2f}   "
              f"top-SV share of dW2a {m(lambda o:o['dW2a_top_sv_share']):.2f}")


if __name__ == "__main__":
    main()
