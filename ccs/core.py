"""Cognitive Capability Search: the inner loop, the metrics and the score.

An *environment* e (a genome) specifies everything the outer loop controls:

    interface   : carried latents, code, noise           (what the learner is exposed to)
    objective   : weights on two self-supervised signals  (what it is pushed to do)
                    w_rec   - reconstruct the channel through h2
                    w_align - "recycling": make h2(channel only) match the
                              representation the innate sense produces
    condition   : channel | both | mixed                  (is the innate sense present?)
    plasticity  : two phases, each = (set of unlocked regions, number of steps)
                  -> unlock -> learn -> (selection by the outer loop) -> consolidate (freeze)
    lr          : learning rate

None of these names a capability. The score is computed against a *generic* probe
bank - how much more of the world's latent structure becomes cheaply readable from
the learner's association region - not against a target task.
"""
from __future__ import annotations

import numpy as np

from .learner import Adam, REGIONS, PARAMS_OF
from .world import World, N_LATENT, N_INNATE, C_DIM, S_DIM, CODES

NOISES = (0.05, 0.3, 1.0)
WEIGHTS = (0.0, 0.5, 1.0)
CONDS = ("channel", "both", "mixed")
STEPS = (0, 150, 300, 600)
LRS = (1e-3, 3e-3)
MAX_STEPS = 2 * max(STEPS)

BETA = 1.0     # interference penalty
GAMMA = 0.02   # training-time cost (per MAX_STEPS)
# Interference metric version. v1 (original study): innate task measured with the new
# channel silent. v2: the worse of channel-silent and channel-on, so crosstalk through
# the new pathway is penalised too (closes the gap the v1 search exploited).
METRIC = "v1"
N_SHOT = 64    # samples for a few-shot linear probe ("cheaply readable")


# ---------------------------------------------------------------------- genomes
def random_genome(rng, regime: str = "adaptive") -> dict:
    carried = (rng.random(N_LATENT) < 0.5).astype(int)
    if carried.sum() == 0:
        carried[rng.integers(N_LATENT)] = 1
    w_rec, w_align = 0.0, 0.0
    while w_rec == 0 and w_align == 0:
        w_rec, w_align = rng.choice(WEIGHTS), rng.choice(WEIGHTS)
    g = {
        "carried": carried,
        "code": str(rng.choice(CODES)),
        "noise": float(rng.choice(NOISES)),
        "w_rec": float(w_rec), "w_align": float(w_align),
        "cond": str(rng.choice(CONDS)),
        "phases": [(tuple(int(x) for x in (rng.random(3) < 0.5)), int(rng.choice(STEPS)))
                   for _ in range(2)],
        "lr": float(rng.choice(LRS)),
    }
    return constrain(g, regime)


def constrain(g: dict, regime: str) -> dict:
    """Fix the plasticity genes for the non-adaptive baselines."""
    if regime == "global":
        g["phases"] = [((1, 1, 1), 600), ((0, 0, 0), 0)]
    elif regime == "local":
        g["phases"] = [((0, 1, 0), 600), ((0, 0, 0), 0)]
    return g


def mutate(g: dict, rng, regime: str, p: float = 0.2) -> dict:
    g = {k: (v.copy() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else v))
         for k, v in g.items()}
    for i in range(N_LATENT):
        if rng.random() < p / 2:
            g["carried"][i] ^= 1
    if g["carried"].sum() == 0:
        g["carried"][rng.integers(N_LATENT)] = 1
    if rng.random() < p: g["code"] = str(rng.choice(CODES))
    if rng.random() < p: g["noise"] = float(rng.choice(NOISES))
    if rng.random() < p: g["w_rec"] = float(rng.choice(WEIGHTS))
    if rng.random() < p: g["w_align"] = float(rng.choice(WEIGHTS))
    if g["w_rec"] == 0 and g["w_align"] == 0:
        g["w_rec"] = 1.0
    if rng.random() < p: g["cond"] = str(rng.choice(CONDS))
    if rng.random() < p: g["lr"] = float(rng.choice(LRS))
    phases = []
    for mask, steps in g["phases"]:
        mask = tuple(int(m ^ (rng.random() < p / 2)) for m in mask)
        if rng.random() < p:
            steps = int(rng.choice(STEPS))
        phases.append((mask, steps))
    g["phases"] = phases
    return constrain(g, regime)


def crossover(a: dict, b: dict, rng) -> dict:
    child = {}
    for k in a:
        src = a if rng.random() < 0.5 else b
        v = src[k]
        child[k] = v.copy() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else v)
    return child


def genome_key(g: dict) -> str:
    ph = "|".join(f"{''.join(map(str, m))}x{s}" for m, s in g["phases"])
    return (f"{''.join(map(str, g['carried']))}-{g['code']}-n{g['noise']}-rec{g['w_rec']}"
            f"-al{g['w_align']}-{g['cond']}-{ph}-lr{g['lr']}")


def describe(g: dict) -> str:
    carried = [i for i in range(N_LATENT) if g["carried"][i]]
    ph = []
    for mask, steps in g["phases"]:
        regs = [r for r, m in zip(REGIONS, mask) if m]
        if steps and regs:
            ph.append(f"open {'+'.join(regs)} for {steps}")
    return (f"carries z{carried} | {g['code']} code, noise {g['noise']} | "
            f"objective rec={g['w_rec']}, align={g['w_align']} | input={g['cond']} | "
            f"{'; '.join(ph) or 'no plasticity'} | lr={g['lr']}")


# ---------------------------------------------------------------------- data
class EvalData:
    """Fixed evaluation sets shared by every genome (same z, same noise draws)."""
    def __init__(self, world: World, seed: int = 7, n_test: int = 1000):
        rng = np.random.default_rng(seed)
        self.z_train = world.sample_z(rng, N_SHOT * 4)
        self.z_test = world.sample_z(rng, n_test)
        self.s_train = world.sense(self.z_train, rng)
        self.s_test = world.sense(self.z_test, rng)
        self.noise_rng_seed = seed + 1


def _channel(world, g, z, seed):
    r = np.random.default_rng(seed)
    return world.channel(z, g["carried"], g["code"], g["noise"], r)


def _inputs(cond, s, c, rng=None):
    if cond == "channel":
        return np.zeros_like(s), c
    if cond == "both":
        return s, c
    # mixed: half the rows lose the innate sense
    s = s.copy()
    idx = (rng.random(len(s)) < 0.5) if rng is not None else (np.arange(len(s)) % 2 == 0)
    s[idx] = 0.0
    return s, c


# ---------------------------------------------------------------------- metrics
def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-1):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    P = Xte @ W
    ss_res = ((Yte - P) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9
    return 1 - ss_res / ss_tot


def capability_matrix(L, world, g, D: EvalData, n_shot: int = N_SHOT, reps: int = 4):
    """C[cond, latent]: few-shot linear decodability of each latent from h2.
    cond 0 = channel only, cond 1 = innate sense + channel."""
    c_tr = _channel(world, g, D.z_train, D.noise_rng_seed)
    c_te = _channel(world, g, D.z_test, D.noise_rng_seed + 1)
    C = np.zeros((2, N_LATENT))
    for ci, cond in enumerate(("channel", "both")):
        Htr = L.forward(*_inputs(cond, D.s_train, c_tr))["h2"]
        Hte = L.forward(*_inputs(cond, D.s_test, c_te))["h2"]
        r2 = np.zeros(N_LATENT)
        for k in range(reps):
            sl = slice(k * n_shot, (k + 1) * n_shot)
            r2 += ridge_r2(Htr[sl], D.z_train[sl], Hte, D.z_test)
        C[ci] = np.clip(r2 / reps, 0, 1)
    return C


def innate_r2(L, D: EvalData, world=None, g=None):
    """Innate-task R^2 through the frozen readout R3. By default the new channel is
    silent; pass world and g to measure it with the channel switched on (crosstalk)."""
    c = (np.zeros((len(D.s_test), C_DIM)) if g is None
         else _channel(world, g, D.z_test, D.noise_rng_seed + 2))
    f = L.forward(D.s_test, c)
    y, t = f["y"], D.z_test[:, :N_INNATE]
    return float(np.mean(1 - ((t - y) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)))


# ---------------------------------------------------------------------- inner loop
def acquire(L0, world, g, seed: int = 0, batch: int = 128, trace: bool = False):
    """Inner loop: expose a copy of the learner to environment g under its plasticity schedule."""
    L = L0.clone()
    rng = np.random.default_rng(seed)
    # objective head: linear decoder h2 -> c (discarded afterwards)
    Wd = rng.normal(size=(L.p["W2a"].shape[1], C_DIM)) * 0.1
    bd = np.zeros(C_DIM)
    opt = Adam(L.p, g["lr"])
    dopt = Adam({"Wd": Wd, "bd": bd}, g["lr"])
    dparams = {"Wd": Wd, "bd": bd}
    hist = []
    for mask, steps in g["phases"]:
        allowed = {p for r, m in zip(REGIONS, mask) if m for p in PARAMS_OF[r]}
        if not allowed:
            continue  # a phase that opens nothing is just waiting
        for _ in range(steps):
            z = world.sample_z(rng, batch)
            s = world.sense(z, rng)
            c = world.channel(z, g["carried"], g["code"], g["noise"], rng)
            grads = {k: np.zeros_like(v) for k, v in L.p.items()}
            loss = 0.0
            if g["w_rec"] > 0:
                f = L.forward(*_inputs(g["cond"], s, c, rng), rng)
                pred = f["h2"] @ dparams["Wd"] + dparams["bd"]
                err = pred - c
                loss += g["w_rec"] * float((err ** 2).mean())
                dpred = g["w_rec"] * 2 * err / err.size
                dgr = {"Wd": f["h2"].T @ dpred, "bd": dpred.sum(0)}
                dopt.step(dparams, dgr, {"Wd", "bd"})
                for k, v in L.backward_h2(f, dpred @ dparams["Wd"].T).items():
                    grads[k] += v
            if g["w_align"] > 0:
                ref = L0.forward(s, np.zeros_like(c), rng)["h2"]     # consolidated innate code
                f = L.forward(np.zeros_like(s), c, rng)               # channel only
                err = f["h2"] - ref
                loss += g["w_align"] * float((err ** 2).mean())
                for k, v in L.backward_h2(f, g["w_align"] * 2 * err / err.size).items():
                    grads[k] += v
            opt.step(L.p, grads, allowed)
            if trace:
                hist.append(loss)
    return (L, hist) if trace else L


def total_steps(g) -> int:
    return sum(steps for mask, steps in g["phases"] if any(mask))


def evaluate(g, world, L0, D, base_innate, seed: int = 0, keep=False):
    C0 = capability_matrix(L0, world, g, D)
    L = acquire(L0, world, g, seed=seed)
    C1 = capability_matrix(L, world, g, D)
    inn = innate_r2(L, D)
    inn_on = innate_r2(L, D, world, g)
    gain = float((C1 - C0).mean())
    worst = inn if METRIC == "v1" else min(inn, inn_on)
    interference = max(0.0, base_innate - max(worst, 0.0))
    cost = total_steps(g) / MAX_STEPS
    score = gain - BETA * interference - GAMMA * cost
    out = {"gain": gain, "interference": interference, "innate": inn, "innate_on": inn_on, "cost": cost,
           "score": score, "C0": C0.tolist(), "C1": C1.tolist()}
    if keep:
        out["learner"] = L
    return out


# ---------------------------------------------------------------------- held-out battery
class Battery:
    """Held-out downstream tasks never seen by the search: random nonlinear functions of
    2-3 latents each. Measured by few-shot linear readout from h2 (innate sense + channel)."""
    def __init__(self, n_tasks: int = 40, seed: int = 99):
        rng = np.random.default_rng(seed)
        self.tasks = []
        for _ in range(n_tasks):
            k = int(rng.integers(2, 4))
            idx = rng.choice(N_LATENT, size=k, replace=False)
            W = rng.normal(size=(k, 12))
            b = rng.normal(size=12) * 0.5
            v = rng.normal(size=12)
            self.tasks.append((idx, W, b, v))

    def targets(self, z):
        return np.stack([np.tanh(z[:, idx] @ W + b) @ v for idx, W, b, v in self.tasks], 1)

    def hidden_mask(self):
        return np.array([bool(np.any(idx >= N_INNATE)) for idx, *_ in self.tasks])

    def score(self, L, world, g, D: EvalData, reps: int = 4):
        """Per-task few-shot R^2. g=None means the channel is silent."""
        Ytr, Yte = self.targets(D.z_train), self.targets(D.z_test)
        if g is None:
            c_tr = np.zeros((len(D.z_train), C_DIM)); c_te = np.zeros((len(D.z_test), C_DIM))
        else:
            c_tr = _channel(world, g, D.z_train, D.noise_rng_seed)
            c_te = _channel(world, g, D.z_test, D.noise_rng_seed + 1)
        Htr = L.forward(D.s_train, c_tr)["h2"]
        Hte = L.forward(D.s_test, c_te)["h2"]
        r2 = np.zeros(len(self.tasks))
        for k in range(reps):
            sl = slice(k * N_SHOT, (k + 1) * N_SHOT)
            r2 += ridge_r2(Htr[sl], Ytr[sl], Hte, Yte)
        return np.clip(r2 / reps, 0, 1)
