"""Training-set export for seq2profile models (default 2114 bp in -> 1000 bp out, 1 bp, unstranded).

HDF5 layout (one file per sampling mode), per split group `train`/`val`/`test`:
  seq       (N, input_len)  uint8   A=0 C=1 G=2 T=3
  counts    (N, R, output_len) int32  observed cut counts per replicate
  expected  (N, output_len) float32  noise-free ground truth
  chrom     (N,) int16 index into attrs['chroms'];  start (N,) int64 = output-window start (0-based)
  is_peak   (N,) bool (all True in tile mode)

Modes:
  tile             every window on a fixed stride; composition signal is present in the contrast.
  peaks_gcmatched  top-signal windows plus GC-matched non-peak windows (BPNet/ChromBPNet-style).
                   GC matching removes between-window GC differences from the peak/background
                   contrast by construction, so a composition grammar is partly hidden from the model.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

from .genome import Genome
from .seq import gc_fraction
from .simulate import load_tracks


def output_starts(chrom_len: int, input_len: int, output_len: int, stride: int) -> np.ndarray:
    flank = (input_len - output_len) // 2
    return np.arange(flank, chrom_len - output_len - flank + 1, stride)


def gather(genome: Genome, run_dir, chrom: str, starts: np.ndarray, input_len: int, output_len: int,
           n_reps: int):
    flank = (input_len - output_len) // 2
    seq = genome.seqs[chrom]
    names = ["expected"] + [f"counts_rep{r + 1}" for r in range(n_reps)]
    tr = load_tracks(run_dir, chrom, names)
    idx_in = starts[:, None] - flank + np.arange(input_len)
    idx_out = starts[:, None] + np.arange(output_len)
    counts = np.stack([tr[f"counts_rep{r + 1}"][idx_out] for r in range(n_reps)], axis=1)
    return seq[idx_in], counts, tr["expected"][idx_out]


def _gc_matched(gc: np.ndarray, is_peak: np.ndarray, ratio: float, rng, bins=np.linspace(0.2, 0.8, 25)):
    pb = np.digitize(gc, bins)
    keep = list(np.flatnonzero(is_peak))
    for b in np.unique(pb[is_peak]):
        want = int(round(ratio * np.sum(is_peak & (pb == b))))
        pool = np.flatnonzero(~is_peak & (pb == b))
        keep += list(rng.choice(pool, min(want, len(pool)), replace=False))
    return np.sort(np.asarray(keep, dtype=int))


def export_dataset(genome: Genome, run_dir, splits: dict[str, list[str]], out_path, mode: str = "tile",
                   train_stride: int = 500, peak_quantile: float = 0.9, neg_ratio: float = 1.0,
                   seed: int = 0) -> dict:
    params = json.loads((Path(run_dir) / "params.json").read_text())
    Lin, Lout, R = params["input_len"], params["output_len"], params["chec"]["n_replicates"]
    rng = np.random.default_rng(seed)
    chroms = genome.chroms
    summary = {"mode": mode}
    with h5py.File(out_path, "w") as h5:
        h5.attrs.update({"encoding": "ACGT", "input_len": Lin, "output_len": Lout, "mode": mode,
                         "chroms": json.dumps(chroms), "grammar": params["grammar_name"]})
        for split, cs in splits.items():
            stride = train_stride if split == "train" else Lout
            parts = []
            for c in cs:
                st = output_starts(len(genome.seqs[c]), Lin, Lout, stride)
                seq, cnt, exp = gather(genome, run_dir, c, st, Lin, Lout, R)
                parts.append((np.full(len(st), chroms.index(c), np.int16), st, seq, cnt, exp))
            ci, st, seq, cnt, exp = (np.concatenate(x) for x in zip(*parts))
            tot = exp.sum(1)
            gc = gc_fraction(seq)
            if mode == "tile":
                is_peak = np.ones(len(st), bool)
                sel = np.arange(len(st))
            elif mode == "peaks_gcmatched":
                peak = tot >= np.quantile(tot, peak_quantile)
                sel = _gc_matched(gc, peak, neg_ratio, rng)
                is_peak = peak[sel]
            else:
                raise ValueError(f"unknown mode {mode}")
            g = h5.create_group(split)
            g.create_dataset("seq", data=seq[sel], compression="gzip")
            g.create_dataset("counts", data=cnt[sel].astype(np.int32), compression="gzip")
            g.create_dataset("expected", data=exp[sel].astype(np.float32), compression="gzip")
            g.create_dataset("chrom", data=ci[sel])
            g.create_dataset("start", data=st[sel])
            g.create_dataset("is_peak", data=is_peak)
            rho = spearmanr(gc[sel], np.log(tot[sel] + 1e-6)).statistic
            summary[split] = {"n": int(len(sel)), "n_peak": int(is_peak.sum()),
                              "spearman_gc_logsignal": float(rho)}
    return summary


def write_bedgraph(genome: Genome, run_dir, track: str, out_path) -> None:
    """Run-length bedGraph of a track (e.g. counts_rep1), zero runs omitted. Convert with
    bedGraphToBigWig using genome.fa.fai.sizes from the genome directory."""
    with gzip.open(out_path, "wt") as fh:
        for c in genome.chroms:
            a = load_tracks(run_dir, c, [track])[track]
            brk = np.flatnonzero(np.diff(a) != 0) + 1
            s = np.r_[0, brk]
            e = np.r_[brk, len(a)]
            v = a[s]
            for i in np.flatnonzero(v != 0):
                fh.write(f"{c}\t{s[i]}\t{e[i]}\t{v[i]:g}\n")
