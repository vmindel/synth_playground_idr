"""Attribution analyses: seqlet calling and word enrichment, diffuseness, and agreement between
attribution maps (oracle vs model, or oracle grammar A vs grammar B)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .grammars import kmer_mismatches
from .ism import centred, contributions
from .seq import decode, encode, is_gc, revcomp


def call_seqlets(contrib: np.ndarray, region: tuple[int, int], smooth: int = 7, quantile: float = 0.99,
                 min_sep: int = 10) -> np.ndarray:
    """Positive-contribution seqlet centres, (M, 2) array of (window, position). The threshold is a
    quantile of smoothed contribution pooled over all windows, as a motif caller would use."""
    a, b = region
    sm = uniform_filter1d(contrib.astype(np.float64), smooth, axis=1)
    thr = np.quantile(sm[:, a:b], quantile)
    out = []
    for n in range(len(sm)):
        pk, _ = find_peaks(sm[n, a:b], height=thr, distance=min_sep)
        out += [(n, a + p) for p in pk]
    return np.asarray(out, int).reshape(-1, 2)


def _extract(seqs: np.ndarray, centres: np.ndarray, half: int) -> np.ndarray:
    idx = np.clip(centres[:, 1][:, None] + np.arange(-half, half + 1), 0, seqs.shape[1] - 1)
    return seqs[centres[:, 0][:, None], idx]


def _has(wins: np.ndarray, word: str, max_mm: int) -> np.ndarray:
    km = encode(word)
    if wins.shape[1] < len(km):
        return np.zeros(len(wins), bool)
    h = np.minimum(kmer_mismatches(wins, km), kmer_mismatches(wins, revcomp(km)))
    return (h <= max_mm).any(1)


def _canonical_kmer_counts(wins: np.ndarray, k: int) -> np.ndarray:
    n = wins.shape[1] - k + 1
    pw = 4 ** np.arange(k - 1, -1, -1)
    fwd = np.zeros((len(wins), n), np.int64)
    rc = np.zeros((len(wins), n), np.int64)
    comp = 3 - wins
    for t in range(k):
        fwd += wins[:, t : t + n].astype(np.int64) * pw[t]
        rc += comp[:, t : t + n].astype(np.int64) * pw[k - 1 - t]
    return np.bincount(np.minimum(fwd, rc).ravel(), minlength=4**k)


def _kmer_str(i: int, k: int) -> str:
    return decode(np.array([(i // 4 ** (k - 1 - t)) % 4 for t in range(k)], np.uint8))


def seqlet_report(seqs: np.ndarray, contrib: np.ndarray, region: tuple[int, int], k: int = 6,
                  half: int = 7, quantile: float = 0.99, n_top: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a, b = region
    sl = call_seqlets(contrib, region, quantile=quantile)
    nbg = max(5 * len(sl), 2000)
    bg = np.stack([rng.integers(0, len(seqs), nbg), rng.integers(a + half, b - half, nbg)], 1)
    ws, wb = _extract(seqs, sl, half), _extract(seqs, bg, half)
    cs, cb = _canonical_kmer_counts(ws, k), _canonical_kmer_counts(wb, k)
    enr = np.log2(((cs + 1) / (cs.sum() + 4**k)) / ((cb + 1) / (cb.sum() + 4**k)))
    enr[cs < 5] = -np.inf
    top = np.argsort(enr)[::-1][:n_top]
    rep = {
        "n_seqlets": int(len(sl)),
        "seqlet_gc": float(is_gc(ws).mean()) if len(sl) else float("nan"),
        "bg_gc": float(is_gc(wb).mean()),
        "gcbox_core_rate": float(_has(ws, "GGGCGG", 0).mean()) if len(sl) else float("nan"),
        "gcbox_core_bg": float(_has(wb, "GGGCGG", 0).mean()),
        "lowcomp_rate": float(_has(ws, "GGGAGGG", 1).mean()) if len(sl) else float("nan"),
        "lowcomp_bg": float(_has(wb, "GGGAGGG", 1).mean()),
        "top_kmers": [(_kmer_str(int(i), k), float(enr[i]), int(cs[i])) for i in top if np.isfinite(enr[i])],
    }
    return rep


def concentration(contrib: np.ndarray, region: tuple[int, int], top: float = 0.05) -> float:
    """Median over windows of the share of |contribution| carried by the top `top` fraction of
    positions. Motif grammars concentrate attribution; composition grammars spread it."""
    a, b = region
    x = np.abs(contrib[:, a:b])
    n = max(1, int(round(top * x.shape[1])))
    s = np.sort(x, axis=1)[:, ::-1]
    return float(np.median(s[:, :n].sum(1) / np.maximum(s.sum(1), 1e-12)))


def gc_attribution(contrib: np.ndarray, seqs: np.ndarray, region: tuple[int, int]) -> dict:
    a, b = region
    c, s = contrib[:, a:b], seqs[:, a:b]
    m = is_gc(s)
    return {"mean_contrib_GC": float(c[m].mean()), "mean_contrib_AT": float(c[~m].mean())}


def _rowcorr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x**2).sum(1) * (y**2).sum(1))
    return np.where(den > 0, (x * y).sum(1) / np.maximum(den, 1e-30), np.nan)


def agreement(hyp_a: np.ndarray, hyp_b: np.ndarray, seqs: np.ndarray, region: tuple[int, int],
              top: float = 0.05) -> dict:
    a, b = region
    ca, cb = contributions(hyp_a, seqs)[:, a:b], contributions(hyp_b, seqs)[:, a:b]
    ha = centred(hyp_a)[:, a:b].reshape(len(seqs), -1)
    hb = centred(hyp_b)[:, a:b].reshape(len(seqs), -1)
    n = max(1, int(round(top * (b - a))))
    ta = np.argsort(-np.abs(ca), 1)[:, :n]
    tb = np.argsort(-np.abs(cb), 1)[:, :n]
    jac = [len(np.intersect1d(x, y)) / len(np.union1d(x, y)) for x, y in zip(ta, tb)]
    return {
        "contrib_pearson_median": float(np.nanmedian(_rowcorr(ca, cb))),
        "hyp_pearson_median": float(np.nanmedian(_rowcorr(ha, hb))),
        "top5pct_jaccard_median": float(np.median(jac)),
    }
