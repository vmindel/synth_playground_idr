"""Sequence perturbations that separate composition, low-complexity avidity and motif hypotheses.

Each design rewrites a core (default 300 bp, oligo-scale, around the window summit) in its
native context. Predicted effects under each planted grammar give a grammar x design matrix;
designs whose effects differ most between grammars are candidates for an MPBA library.
"""
from __future__ import annotations

import numpy as np

from .grammars import kmer_hits, pwm_from_consensus, pwm_logodds, pwm_site_scores
from .seq import encode, gc_fraction, is_gc, revcomp

LOWCOMP = "GGGAGGG"
GCBOX = "GGGGCGGGG"
_GCBOX_LO = pwm_logodds(pwm_from_consensus(GCBOX))
_GCBOX_THR = 0.6 * _GCBOX_LO.max(1).sum()


def lowcomp_mask(seq: np.ndarray, motif: str = LOWCOMP, max_mm: int = 1) -> np.ndarray:
    return kmer_hits(seq[None], motif, max_mm)[0]


def gcbox_centres(seq: np.ndarray) -> np.ndarray:
    return np.flatnonzero(pwm_site_scores(seq[None], _GCBOX_LO)[0] >= _GCBOX_THR)


def gcbox_mask(seq: np.ndarray) -> np.ndarray:
    m = np.zeros(len(seq), bool)
    W = _GCBOX_LO.shape[0]
    for c in gcbox_centres(seq):
        m[max(0, c - W // 2) : c + W - W // 2] = True
    return m


def _sample(n: int, gc: float, rng) -> np.ndarray:
    strong = rng.random(n) < gc
    coin = rng.random(n) < 0.5
    return np.where(strong, np.where(coin, 1, 2), np.where(coin, 0, 3)).astype(np.uint8)


def gc_swap(seq, a, b, rng):
    """Swap C<->G at a random half of core G/C positions. GC count is preserved exactly; G-runs break."""
    x = seq.copy()
    core = x[a:b]
    m = is_gc(core) & (rng.random(b - a) < 0.5)
    core[m] = 3 - core[m]
    return x


def block_shuffle(seq, a, b, rng, block: int = 25):
    """Shuffle bases within consecutive blocks: local composition kept, k-mers destroyed."""
    x = seq.copy()
    for s in range(a, b, block):
        e = min(s + block, b)
        x[s:e] = rng.permutation(x[s:e])
    return x


def kmer_scrub(seq, a, b, rng, motif: str = LOWCOMP, max_mm: int = 1, max_iter: int = 100):
    """Remove low-complexity k-mer matches from the core with GC-class-preserving swaps
    (G<->C, A<->T), so composition is unchanged."""
    x = seq.copy()
    for _ in range(max_iter):
        m = lowcomp_mask(x, motif, max_mm)
        m[:a] = False
        m[b:] = False
        idx = np.flatnonzero(m)
        if idx.size == 0:
            break
        pick = idx[rng.random(idx.size) < 0.3]
        if pick.size == 0:
            pick = idx[:1]
        x[pick] = 3 - x[pick]  # A<->T, C<->G: stays within GC class
    return x


def gcbox_ko(seq, a, b, rng, max_iter: int = 50):
    """Knock out GC-box (DBD) matches in the core by turning the two central bases into A/T."""
    x = seq.copy()
    for _ in range(max_iter):
        cs = [c for c in gcbox_centres(x) if a <= c < b - 1]
        if not cs:
            break
        for c in cs:
            x[c : c + 2] = rng.choice(np.array([0, 3], np.uint8), 2)
    return x


def at_backfill(seq, a, b, rng, gc: float = 0.35):
    """Keep low-complexity and GC-box matches, resample everything else in the core at low GC."""
    x = seq.copy()
    keep = lowcomp_mask(x) | gcbox_mask(x)
    idx = np.arange(a, b)[~keep[a:b]]
    x[idx] = _sample(idx.size, gc, rng)
    return x


def gc_backfill_scrubbed(seq, a, b, rng, gc: float = 0.65):
    """Resample the core at high GC, then remove k-mer and GC-box matches: composition without words."""
    x = seq.copy()
    x[a:b] = _sample(b - a, gc, rng)
    return gcbox_ko(kmer_scrub(x, a, b, rng), a, b, rng)


def kmer_implant(seq, a, b, rng, n: int = 3, motif: str = LOWCOMP):
    """Resample the core, scrub matches, then implant n exact k-mer copies. The backfill GC is lowered
    so that core GC after implanting matches the native core (the k-mers are G-rich)."""
    x = seq.copy()
    km = encode(motif)
    k = len(km)
    L = b - a
    target = float(gc_fraction(seq[a:b])) * L
    fill_gc = np.clip((target - n * is_gc(km).sum()) / (L - n * k), 0.05, 0.95)
    x[a:b] = _sample(L, fill_gc, rng)
    x = gcbox_ko(kmer_scrub(x, a, b, rng), a, b, rng)
    slots = np.linspace(a, b - k, n + 2)[1:-1].astype(int)
    for s in slots:
        x[s : s + k] = km if rng.random() < 0.5 else revcomp(km)
    return x


DESIGNS = {
    "gc_swap": gc_swap,
    "block_shuffle": block_shuffle,
    "kmer_scrub": kmer_scrub,
    "gcbox_ko": gcbox_ko,
    "at_backfill": at_backfill,
    "gc_backfill_scrubbed": gc_backfill_scrubbed,
    "kmer_implant": kmer_implant,
}


def core_bounds(expected_central: np.ndarray, flank: int, input_len: int, core_len: int = 300):
    summit = flank + int(np.argmax(expected_central))
    a = int(np.clip(summit - core_len // 2, 0, input_len - core_len))
    return a, a + core_len


def design_matrix(oracles: dict, seqs: np.ndarray, core_len: int = 300, seed: int = 0,
                  ref_oracle: str | None = None):
    """Median log2 fold change in predicted total signal, grammar x design, on the same sequences.
    Cores are centred on each window's summit under `ref_oracle`, or by default on the summit of the
    per-window-normalised profiles summed across grammars, so all grammars see identical sequences.
    Returns (grammars, designs, matrix, designed_sequences dict)."""
    rng = np.random.default_rng(seed)
    any_or = next(iter(oracles.values()))
    flank, Lin = any_or.flank, any_or.input_len
    names = list(oracles)
    dnames = list(DESIGNS)
    # cores are fixed per window so every grammar sees identical designed sequences
    base = {g: o(seqs) for g, o in oracles.items()}
    if ref_oracle is None:
        prof = sum(b / b.sum(1, keepdims=True) for b in base.values())
    else:
        prof = base[ref_oracle]
    bounds = [core_bounds(p, flank, Lin, core_len) for p in prof]
    designed = {d: np.stack([DESIGNS[d](s, a, b, rng) for s, (a, b) in zip(seqs, bounds)]) for d in dnames}
    M = np.zeros((len(names), len(dnames)))
    for i, g in enumerate(names):
        tot = base[g].sum(1)
        for j, d in enumerate(dnames):
            M[i, j] = np.median(np.log2(oracles[g](designed[d]).sum(1) / tot))
    return names, dnames, M, designed
