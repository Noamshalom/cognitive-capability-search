"""A small modular learner with region-level plasticity control (pure NumPy).

Architecture (fixed; nothing about it is chosen by the search):

    s  (innate sense, 16) --R1--> h1 (32) --\
                                             R2 --> h2 (32) --R3--> innate readout (z[0:4])
    c  (new channel, 16) --Rc--> hc (32) --/

Regions:  R1 = sensory encoder, Rc = interface ("new") region,
          R2 = association region, R3 = innate readout (always frozen after development).

"Development" pretrains R1, R2, R3 on the innate task with the new channel silent,
so the channel enters h2 only through random, untrained Rc / W2b weights.

During capability acquisition only regions listed in a plasticity mask are updated;
everything else is frozen. That lets the outer loop decide *where* the learner is
plastic and *for how long*.
"""
from __future__ import annotations

import copy
import numpy as np

H = 32
W2B_INIT = 0.15       # initial strength of the interface -> association pathway
NEURAL_NOISE = 0.2    # intrinsic noise in the association region
REGIONS = ("R1", "Rc", "R2")          # regions the outer loop may unlock
PARAMS_OF = {"R1": ("W1", "b1"), "Rc": ("Wc",), "R2": ("W2a", "W2b", "b2"),
             "R3": ("W3", "b3")}


class Learner:
    def __init__(self, s_dim: int, c_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        g = lambda i, o: rng.normal(size=(i, o)) / np.sqrt(i)
        self.p = {
            "W1": g(s_dim, H), "b1": np.zeros(H),
            "Wc": g(c_dim, H),
            # the new channel reaches the association region only weakly at birth
            "W2a": g(H, H), "W2b": g(H, H) * W2B_INIT, "b2": np.zeros(H),
            "W3": g(H, out_dim), "b3": np.zeros(out_dim),
        }

    def clone(self) -> "Learner":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ forward
    def forward(self, s, c, rng=None):
        """rng=None -> deterministic noise draw (fixed seed), so evaluations are comparable."""
        p = self.p
        h1 = np.tanh(s @ p["W1"] + p["b1"])
        hc = np.tanh(c @ p["Wc"])
        r = rng if rng is not None else np.random.default_rng(len(s))
        eta = NEURAL_NOISE * r.normal(size=(len(s), H))
        h2 = np.tanh(h1 @ p["W2a"] + hc @ p["W2b"] + p["b2"] + eta)
        y = h2 @ p["W3"] + p["b3"]
        return {"s": s, "c": c, "h1": h1, "hc": hc, "h2": h2, "y": y}

    def backward_h2(self, f, dh2):
        """Gradients of all region params given dL/dh2."""
        p = self.p
        da2 = dh2 * (1 - f["h2"] ** 2)
        g = {"W2a": f["h1"].T @ da2, "W2b": f["hc"].T @ da2, "b2": da2.sum(0)}
        da1 = (da2 @ p["W2a"].T) * (1 - f["h1"] ** 2)
        g["W1"], g["b1"] = f["s"].T @ da1, da1.sum(0)
        dac = (da2 @ p["W2b"].T) * (1 - f["hc"] ** 2)
        g["Wc"] = f["c"].T @ dac
        return g


class Adam:
    def __init__(self, params: dict, lr: float):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict, allowed: set):
        self.t += 1
        b1, b2 = 0.9, 0.999
        for k, g in grads.items():
            if k not in allowed:
                continue
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def develop(world, seed: int = 0, steps: int = 4000, batch: int = 128) -> Learner:
    """'Evolution + development': train R1, R2, R3 on the innate task (predict z[0:4] from s)."""
    from .world import S_DIM, C_DIM, N_INNATE
    rng = np.random.default_rng(seed + 1000)
    L = Learner(S_DIM, C_DIM, N_INNATE, seed=seed)
    opt = Adam(L.p, lr=3e-3)
    allowed = {"W1", "b1", "W2a", "b2", "W3", "b3"}
    zc = np.zeros((batch, C_DIM))
    for _ in range(steps):
        z = world.sample_z(rng, batch)
        s = world.sense(z, rng)
        f = L.forward(s, zc, rng)
        dy = 2 * (f["y"] - z[:, :N_INNATE]) / (batch * N_INNATE)
        g = L.backward_h2(f, dy @ L.p["W3"].T)
        g["W3"], g["b3"] = f["h2"].T @ dy, dy.sum(0)
        opt.step(L.p, g, allowed)
    return L
