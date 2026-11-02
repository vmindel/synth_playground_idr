"""Model-agnostic in silico saturation mutagenesis.

`predict_fn` maps integer-encoded sequences (B, L) uint8 to a non-negative profile (B, L_out),
e.g. the oracle, or a wrapper around a trained model:

    def predict_fn(x):                      # x: (B, 2114) uint8
        oh = spidr.seq.onehot(x)            # (B, 2114, 4); transpose if the model wants (B, 4, L)
        logits, logcounts = model(torch.from_numpy(oh))
        return softmax(logits) * exp(logcounts)   # (B, 1000) expected counts, as numpy

Output is the hypothetical ISM tensor (N, L, 4): delta log(total predicted counts) for each
substitution, 0 at the reference base and outside the mutated region.
"""
from __future__ import annotations

import numpy as np


def ism(predict_fn, seqs: np.ndarray, start: int = 0, end: int | None = None,
        batch_size: int = 2048, eps: float = 1e-8) -> np.ndarray:
    seqs = np.atleast_2d(seqs).astype(np.uint8)
    N, L = seqs.shape
    end = L if end is None else end
    P = end - start
    pos = np.repeat(np.arange(start, end), 3)
    shift = np.tile(np.array([1, 2, 3], np.uint8), P)
    out = np.zeros((N, L, 4), np.float32)
    rows = np.arange(3 * P)
    for n in range(N):
        s = seqs[n]
        ref = np.log(predict_fn(s[None]).sum(-1) + eps)[0]
        alts = (s[pos] + shift) % 4
        var = np.repeat(s[None], 3 * P, axis=0)
        var[rows, pos] = alts
        vals = np.concatenate([np.log(predict_fn(var[i : i + batch_size]).sum(-1) + eps)
                               for i in range(0, 3 * P, batch_size)])
        out[n, pos, alts] = vals - ref
    return out


def contributions(hyp: np.ndarray, seqs: np.ndarray) -> np.ndarray:
    """Per-position contribution of the observed base: minus the mean substitution effect
    (positive = the base supports binding)."""
    return -hyp.sum(-1) / 4.0


def centred(hyp: np.ndarray) -> np.ndarray:
    return hyp - hyp.mean(-1, keepdims=True)


def to_modisco(seqs: np.ndarray, hyp: np.ndarray):
    """(ohe, attr) in (N, 4, L) layout as used by tfmodisco-lite's `modisco motifs -s/-a` inputs."""
    from .seq import onehot
    return onehot(seqs).transpose(0, 2, 1), centred(hyp).transpose(0, 2, 1).astype(np.float32)
