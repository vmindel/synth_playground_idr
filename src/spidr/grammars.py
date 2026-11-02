"""Planted binding grammars as additive free-energy terms.

Every position i is a candidate site centre. A grammar returns a score s(i) in kT units
(higher = tighter binding), and occupancy is expit(s(i) + mu) with mu set genome-wide.

Components:
  pwm           DBD-like motif term; both strands combined as logaddexp (orientation partition sum).
  composition   linear in G/C fraction in a window; no sequence order information at all.
  kmer_avidity  multivalent weak-site term: log(1 + sum_j exp(e_j)) over a window, where e_j is the
                energy of a match (up to max_mm mismatches) to a low-complexity k-mer.

Note the built-in near-degeneracy: GGGAGGG is one mismatch from GGGCGGG, the GC-box core, and both
are enriched in GC-rich DNA. Separating these hypotheses is the point of the simulator.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d

from .seq import encode, is_gc, revcomp


def pwm_from_consensus(consensus: str, p_cons: list[float] | None = None) -> np.ndarray:
    """Probability matrix (W, 4) from a consensus. Default per-position consensus probabilities
    are higher in the core than at the edges. This is a stand-in, not a measured SP/KLF matrix;
    load a JASPAR matrix with `load_pwm` for that."""
    W = len(consensus)
    if p_cons is None:
        x = np.abs(np.arange(W) - (W - 1) / 2) / ((W - 1) / 2)
        p_cons = 0.95 - 0.35 * x**2
    p = np.zeros((W, 4))
    for k, (b, pc) in enumerate(zip(encode(consensus), p_cons)):
        p[k] = (1 - pc) / 3
        p[k, b] = pc
    return p


def load_pwm(path: str | Path) -> np.ndarray:
    """Read a JASPAR-format matrix (rows 'A [ ... ]' etc.) or a plain 4-row numeric matrix,
    rows in A,C,G,T order. Returns column-normalised probabilities (W, 4)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">") or not line.strip():
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if nums:
            rows.append([float(v) for v in nums])
    m = np.asarray(rows, float)
    if m.shape[0] != 4:
        raise ValueError(f"expected 4 rows (A,C,G,T), got {m.shape[0]}")
    m = m.T
    return m / m.sum(1, keepdims=True)


def pwm_logodds(prob: np.ndarray, pseudocount: float = 0.01, bg: float = 0.25) -> np.ndarray:
    p = prob + pseudocount
    p /= p.sum(1, keepdims=True)
    return np.log(p / bg)


def pwm_site_scores(seq: np.ndarray, logodds: np.ndarray) -> np.ndarray:
    """(B, L) centre-indexed scores, both strands via logaddexp; -inf where the site does not fit."""
    B, L = seq.shape
    W = logodds.shape[0]
    n = L - W + 1
    fwd = np.zeros((B, n))
    rc = np.zeros((B, n))
    lo_rc = logodds[::-1, ::-1]
    for k in range(W):
        sl = seq[:, k : k + n]
        fwd += logodds[k][sl]
        rc += lo_rc[k][sl]
    out = np.full((B, L), -np.inf)
    out[:, W // 2 : W // 2 + n] = np.logaddexp(fwd, rc)
    return out


def kmer_mismatches(seq: np.ndarray, kmer: np.ndarray) -> np.ndarray:
    """(B, L-k+1) Hamming distance of each k-length window to `kmer` (forward strand only)."""
    B, L = seq.shape
    k = len(kmer)
    n = L - k + 1
    match = np.zeros((B, n), dtype=np.int16)
    for t, b in enumerate(kmer):
        match += seq[:, t : t + n] == b
    return k - match


def kmer_hits(seq: np.ndarray, kmer: str, max_mm: int) -> np.ndarray:
    """Boolean (B, L) mask of positions covered by a <=max_mm match on either strand."""
    km = encode(kmer)
    k = len(km)
    B, L = seq.shape
    cover = np.zeros((B, L), bool)
    for m in {kmer: km, "rc": revcomp(km)}.values():
        h = kmer_mismatches(seq, m) <= max_mm
        for t in range(k):
            cover[:, t : t + h.shape[1]] |= h
    return cover


class PWMComponent:
    def __init__(self, consensus: str = "GGGGCGGGG", scale: float = 0.7, pwm_file: str | None = None,
                 name: str = "dbd"):
        prob = load_pwm(pwm_file) if pwm_file else pwm_from_consensus(consensus)
        self.logodds = pwm_logodds(prob)
        self.scale = scale
        self.name = name
        self.radius = self.logodds.shape[0]

    def score(self, seq: np.ndarray) -> np.ndarray:
        return self.scale * pwm_site_scores(seq, self.logodds)


class CompositionComponent:
    def __init__(self, window: int = 201, beta: float = 10.0, gc_ref: float = 0.42, name: str = "idr_comp"):
        if window % 2 == 0:
            raise ValueError("window must be odd so the kernel is centred")
        self.window, self.beta, self.gc_ref, self.name = window, beta, gc_ref, name
        self.radius = window // 2

    def score(self, seq: np.ndarray) -> np.ndarray:
        k = np.full(self.window, 1.0 / self.window)
        gcw = convolve1d(is_gc(seq).astype(np.float64), k, axis=-1, mode="constant", cval=0.0)
        return self.beta * (gcw - self.gc_ref)


class KmerAvidityComponent:
    def __init__(self, motifs=("GGGAGGG",), max_mm: int = 1, eps_site: float = 2.0, lam: float = 2.0,
                 window: int = 201, beta: float = 1.0, name: str = "idr_kmer"):
        if window % 2 == 0:
            raise ValueError("window must be odd so the kernel is centred")
        self.motifs = list(motifs)
        self.max_mm, self.eps_site, self.lam = max_mm, eps_site, lam
        self.window, self.beta, self.name = window, beta, name
        self.kmax = max(len(m) for m in self.motifs)
        self.radius = window // 2 + self.kmax

    def site_weights(self, seq: np.ndarray) -> np.ndarray:
        B, L = seq.shape
        w = np.zeros((B, L))
        for m in self.motifs:
            fwd = encode(m)
            strands = [fwd] if m == _rc_str(m) else [fwd, revcomp(fwd)]
            k = len(fwd)
            for km in strands:
                h = kmer_mismatches(seq, km)
                e = np.where(h <= self.max_mm, np.exp(self.eps_site - self.lam * h), 0.0)
                w[:, k // 2 : k // 2 + e.shape[1]] += e
        return w

    def score(self, seq: np.ndarray) -> np.ndarray:
        agg = convolve1d(self.site_weights(seq), np.ones(self.window), axis=-1, mode="constant", cval=0.0)
        return self.beta * np.log1p(agg)


def _rc_str(s: str) -> str:
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


_TYPES = {"pwm": PWMComponent, "composition": CompositionComponent, "kmer_avidity": KmerAvidityComponent}


class Grammar:
    def __init__(self, name: str, components: list):
        self.name = name
        self.components = components
        self.radius = max(c.radius for c in components)

    @classmethod
    def from_spec(cls, name: str, spec: list[dict]) -> "Grammar":
        comps = []
        for d in spec:
            d = dict(d)
            comps.append(_TYPES[d.pop("type")](**d))
        return cls(name, comps)

    def score_components(self, seq: np.ndarray) -> dict[str, np.ndarray]:
        return {c.name: c.score(seq) for c in self.components}

    def score(self, seq: np.ndarray) -> np.ndarray:
        seq = np.atleast_2d(seq)
        return sum(c.score(seq) for c in self.components)
