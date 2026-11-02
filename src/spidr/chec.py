"""ChEC-seq observation model: occupancy -> tethered MNase cuts -> unstranded cut counts.

expected(x) = depth_scale * sat(rate(x) * mnase_bias(x)) * pcr_gc_bias(x)
  rate(x)        = sum_i occupancy(i) * K(x - i), K = bimodal tether kernel with a protected footprint
  mnase_bias(x)  = exp(gamma * (AT fraction of dinucleotide x,x+1 - 0.5)); MNase prefers A/T, which
                   pushes against a GC-driven signal
  sat(y)         = 1 - exp(-k * y): digestion-time saturation (k = 0 gives the linear regime)
  pcr_gc_bias(x) = Gaussian in local GC around 0.5 (library amplification bias); None disables
observed ~ NegBin(mean=expected, size=nb_dispersion), independently per bp and replicate.
No free-MNase or other non-specific background is modelled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import convolve1d

from .seq import is_gc


@dataclass
class ChecConfig:
    tether_offset: float = 25.0
    tether_sd: float = 12.0
    footprint_half: int = 8
    mnase_at_pref: float = 1.0
    saturation: float = 0.3  # k * (99.99th percentile of rate*bias); 0 = linear
    gc_bias_sd: float | None = 0.25
    gc_bias_window: int = 101
    total_cuts: float = 5e6  # expected cuts per replicate, genome-wide
    nb_dispersion: float = 5.0
    n_replicates: int = 2

    @classmethod
    def from_dict(cls, d: dict | None) -> "ChecConfig":
        return cls(**(d or {}))

    def to_dict(self) -> dict:
        return asdict(self)


def tether_kernel(cfg: ChecConfig) -> np.ndarray:
    R = int(np.ceil(cfg.tether_offset + 4 * cfg.tether_sd))
    d = np.arange(-R, R + 1)
    k = np.exp(-((np.abs(d) - cfg.tether_offset) ** 2) / (2 * cfg.tether_sd**2))
    k[np.abs(d) <= cfg.footprint_half] = 0
    return k / k.sum()


def mnase_bias(seq: np.ndarray, gamma: float) -> np.ndarray:
    at = (~is_gc(seq)).astype(np.float64)
    di = np.empty_like(at)
    di[:, :-1] = (at[:, :-1] + at[:, 1:]) / 2
    di[:, -1] = at[:, -1]
    return np.exp(gamma * (di - 0.5))


def pcr_gc_bias(seq: np.ndarray, sd: float | None, window: int) -> np.ndarray:
    if sd is None:
        return np.ones(seq.shape)
    k = np.full(window, 1.0 / window)
    g = convolve1d(is_gc(seq).astype(np.float64), k, axis=-1, mode="constant", cval=0.0)
    return np.exp(-((g - 0.5) ** 2) / (2 * sd**2))



def sample_counts(expected: np.ndarray, cfg: ChecConfig, rng: np.random.Generator) -> np.ndarray:
    r = cfg.nb_dispersion
    g = rng.gamma(r, 1.0 / r, size=expected.shape)
    return rng.poisson(expected * g)
