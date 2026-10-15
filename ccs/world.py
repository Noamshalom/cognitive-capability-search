"""The toy world: latent causes, an innate sense, and a space of candidate interfaces.

World
-----
z ~ N(0, I_8) are the hidden causes of the world.

The learner's *innate sense* s sees only z[0:4] (through a fixed random nonlinear
mixing). Latents z[4:8] are invisible to it: from the learner's point of view they
are aspects of the world it has no access to.

Interfaces
----------
A candidate interface is a new input channel c (16 dims) that carries some subset
of latents through some code. This is the analogue of a cultural invention such as
writing: a new stimulus stream that a fixed-architecture learner may or may not be
able to make use of.

    carried : which latents the channel carries (binary mask over 8)
    code    : linear | population | phase | hashed
    noise   : additive noise level relative to channel std
"""
from __future__ import annotations

import numpy as np

N_LATENT = 8
N_INNATE = 4          # latents visible to the innate sense
S_DIM = 16            # innate sense dimensionality
C_DIM = 16            # new-channel dimensionality

CODES = ("linear", "population", "phase", "hashed")


class World:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.seed = seed
        # innate sense: s = tanh(A_s z[:4]) + noise
        self.A_s = rng.normal(size=(N_INNATE, S_DIM)) / np.sqrt(N_INNATE) * 1.5
        # per-code random parameters, generated over all 8 latents so a code is
        # a fixed property of the world, independent of which latents it carries
        self.A_lin = rng.normal(size=(N_LATENT, C_DIM))
        self.A_pop = rng.normal(size=(N_LATENT, C_DIM))
        self.mu_pop = rng.normal(size=C_DIM) * 1.0
        self.A_phase = rng.normal(size=(N_LATENT, C_DIM))
        self.phi_phase = rng.uniform(0, 2 * np.pi, size=C_DIM)
        self.hash_table = rng.normal(size=(4096, C_DIM))
        self.hash_mult = rng.integers(1, 2**31 - 1, size=N_LATENT)

    # ------------------------------------------------------------------ sampling
    @staticmethod
    def sample_z(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=(n, N_LATENT))

    def sense(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        s = np.tanh(z[:, :N_INNATE] @ self.A_s)
        return s + 0.05 * rng.normal(size=s.shape)

    def channel(self, z: np.ndarray, carried: np.ndarray, code: str, noise: float,
                rng: np.random.Generator) -> np.ndarray:
        """Encode the carried latents into a 16-d channel."""
        m = carried.astype(float)
        k = max(m.sum(), 1.0)
        zm = z * m  # zero out latents the channel does not carry
        if code == "linear":
            c = zm @ self.A_lin / np.sqrt(k)
        elif code == "population":
            # Gaussian tuning curves along random preferred directions (smooth, nonlinear)
            u = zm @ self.A_pop / np.sqrt(k)
            c = np.exp(-0.5 * (u - self.mu_pop) ** 2)
        elif code == "phase":
            # high-frequency oscillatory code: locally informative, globally ambiguous
            u = zm @ self.A_phase / np.sqrt(k)
            c = np.sin(3.0 * u + self.phi_phase)
        elif code == "hashed":
            # coarse bins of the carried latents, hashed to random vectors:
            # informative in principle, but with no smooth structure to learn
            bins = np.clip(np.floor((z + 3.0) / 1.5), 0, 3).astype(np.int64)  # 4 bins/latent
            key = (bins * self.hash_mult * carried.astype(np.int64)).sum(axis=1) % 4096
            c = self.hash_table[key]
        else:
            raise ValueError(code)
        mean, std = self._channel_stats(carried, code)
        c = (c - mean) / std
        return c + noise * rng.normal(size=c.shape)

    def _channel_stats(self, carried: np.ndarray, code: str):
        """Fixed per-interface normalisation (estimated once from a large sample)."""
        if not hasattr(self, "_stats"):
            self._stats = {}
        key = (code, tuple(int(x) for x in carried))
        if key not in self._stats:
            self._stats[key] = (0.0, 1.0)  # identity while estimating
            r = np.random.default_rng(12345)
            c = self.channel(self.sample_z(r, 20000), carried, code, 0.0, r)
            self._stats[key] = (c.mean(0), c.std(0) + 1e-6)
        return self._stats[key]
