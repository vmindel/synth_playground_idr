"""Synthetic background genome with mammalian-like composition structure.

Layers, in order of application:
  1. Isochores: ~300 kb segments whose GC is drawn from a five-family mixture
     (approximate human-like L1..H3 spread; not fitted to hg38).
  2. Local GC field: kb-scale smooth fluctuation on top of the isochore mean, so that
     composition varies at the 100-300 bp scale an IDR term would read.
  3. CpG depletion outside CpG islands, by CG->TG / CG->CA conversion (deamination-like).
  4. CpG islands, preferentially in GC-rich isochores, undepleted, ~66% GC.
  5. Imperfect simple repeats / low-complexity tracts, e.g. (GGGA)n, (GA)n, poly-A.
  6. Planted SP/KLF-like GC-box sites, enriched in CGIs (promoter-like), sparse elsewhere.

Without 1-4 a GC signal would be trivially separable from background; without 5-6 the
low-complexity and motif hypotheses would have no natural decoys.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .seq import encode, mutate, read_fasta, revcomp, write_fasta


@dataclass
class GenomeConfig:
    n_chrom: int = 5
    chrom_len: int = 2_000_000
    isochore_mean_len: int = 300_000
    isochore_gc_means: tuple = (0.37, 0.41, 0.46, 0.50, 0.55)
    isochore_gc_weights: tuple = (0.19, 0.37, 0.24, 0.12, 0.08)
    isochore_gc_sd: float = 0.01
    local_gc_sd: float = 0.04
    local_gc_scale: int = 2000
    cpg_oe: float = 0.25
    cgi_per_mb: float = 10.0
    cgi_len_mean: int = 1000
    cgi_gc: float = 0.66
    gcbox_consensus: str = "GGGGCGGGG"
    gcbox_per_cgi: float = 2.0
    gcbox_outside_per_mb: float = 20.0
    gcbox_mut_rate: float = 0.1
    repeat_per_mb: float = 60.0
    repeat_units: tuple = ("GGGA", "GA", "GGA", "CA", "A", "AT")
    repeat_len_mean: int = 40
    repeat_mut_rate: float = 0.05
    seed: int = 1

    @classmethod
    def from_dict(cls, d: dict | None) -> "GenomeConfig":
        d = dict(d or {})
        for k in ("isochore_gc_means", "isochore_gc_weights", "repeat_units"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class Genome:
    seqs: dict[str, np.ndarray]
    annotations: list[tuple] = field(default_factory=list)  # (chrom, start, end, type, name)

    @property
    def chroms(self) -> list[str]:
        return list(self.seqs)

    @property
    def total_len(self) -> int:
        return sum(len(s) for s in self.seqs.values())

    def save(self, out_dir: str | Path, cfg: GenomeConfig | None = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_fasta(out / "genome.fa", self.seqs)
        with open(out / "annotations.bed", "w") as fh:
            for c, s, e, t, n in sorted(self.annotations, key=lambda x: (x[0], x[1])):
                fh.write(f"{c}\t{s}\t{e}\t{t}\t{n}\n")
        with open(out / "genome.fa.fai.sizes", "w") as fh:
            for c, a in self.seqs.items():
                fh.write(f"{c}\t{len(a)}\n")
        if cfg is not None:
            (out / "genome_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    @classmethod
    def load(cls, out_dir: str | Path) -> "Genome":
        out = Path(out_dir)
        seqs = read_fasta(out / "genome.fa")
        ann = []
        bed = out / "annotations.bed"
        if bed.exists():
            for line in bed.read_text().splitlines():
                c, s, e, t, n = line.split("\t")
                ann.append((c, int(s), int(e), t, n))
        return cls(seqs, ann)


def _gc_field(L: int, cfg: GenomeConfig, rng: np.random.Generator) -> np.ndarray:
    iso = np.empty(L)
    pos = 0
    sigma = 0.5
    mu_ln = np.log(cfg.isochore_mean_len) - sigma**2 / 2
    w = np.asarray(cfg.isochore_gc_weights, float)
    w /= w.sum()
    while pos < L:
        seg = max(20_000, int(rng.lognormal(mu_ln, sigma)))
        fam = rng.choice(len(w), p=w)
        iso[pos : pos + seg] = rng.normal(cfg.isochore_gc_means[fam], cfg.isochore_gc_sd)
        pos += seg
    # smooth on a 100 bp grid and interpolate; a direct bp-level filter at this scale is very slow
    step = 100
    n = L // step + 2
    coarse = gaussian_filter1d(rng.standard_normal(n), sigma=max(cfg.local_gc_scale / step, 1.0))
    noise = np.interp(np.arange(L), np.arange(n) * step, coarse)
    noise /= noise.std()
    return np.clip(iso + cfg.local_gc_sd * noise, 0.25, 0.75)


def _sample_iid(gc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    strong = rng.random(gc.shape) < gc
    coin = rng.random(gc.shape) < 0.5
    return np.where(strong, np.where(coin, 1, 2), np.where(coin, 0, 3)).astype(np.uint8)


def _deplete_cpg(seq: np.ndarray, allowed: np.ndarray, oe: float, rng: np.random.Generator) -> None:
    idx = np.flatnonzero((seq[:-1] == 1) & (seq[1:] == 2) & allowed[:-1] & allowed[1:])
    sel = idx[rng.random(idx.size) < 1 - oe]
    to_tg = rng.random(sel.size) < 0.5
    seq[sel[to_tg]] = 3  # CG -> TG
    seq[sel[~to_tg] + 1] = 0  # CG -> CA


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _place_cgis(L: int, gc: np.ndarray, cfg: GenomeConfig, rng: np.random.Generator):
    n = rng.poisson(cfg.cgi_per_mb * L / 1e6)
    bins = gc[::1000]
    w = bins**8
    w /= w.sum()
    starts = rng.choice(len(bins), n, p=w) * 1000 + rng.integers(0, 1000, n)
    lens = np.clip(rng.lognormal(np.log(cfg.cgi_len_mean) - 0.125, 0.5, n), 300, 4000).astype(int)
    ivs = _merge([(int(s), int(min(s + l, L))) for s, l in zip(starts, lens)])
    mask = np.zeros(L, bool)
    for s, e in ivs:
        mask[s:e] = True
    return mask, ivs


def generate_genome(cfg: GenomeConfig) -> Genome:
    rng = np.random.default_rng(cfg.seed)
    seqs: dict[str, np.ndarray] = {}
    ann: list[tuple] = []
    site = encode(cfg.gcbox_consensus)
    W = len(site)
    units = [encode(u) for u in cfg.repeat_units]
    for ci in range(cfg.n_chrom):
        name = f"chr{ci + 1}"
        L = cfg.chrom_len
        gc = _gc_field(L, cfg, rng)
        # inflate target GC to offset the G/C lost to CpG depletion
        seq = _sample_iid(np.clip(gc + (1 - cfg.cpg_oe) * (gc / 2) ** 2, 0, 1), rng)
        cgi_mask, cgis = _place_cgis(L, gc, cfg, rng)
        _deplete_cpg(seq, ~cgi_mask, cfg.cpg_oe, rng)
        for s, e in cgis:
            seq[s:e] = _sample_iid(np.full(e - s, cfg.cgi_gc), rng)
            ann.append((name, s, e, "CGI", "CGI"))

        n_rep = rng.poisson(cfg.repeat_per_mb * L / 1e6)
        for _ in range(n_rep):
            ui = rng.integers(len(units))
            length = int(np.clip(rng.lognormal(np.log(cfg.repeat_len_mean) - 0.125, 0.5), 12, 300))
            tract = np.tile(units[ui], length // len(units[ui]) + 1)[:length]
            if rng.random() < 0.5:
                tract = revcomp(tract)
            tract = mutate(tract, cfg.repeat_mut_rate, rng)
            s = int(rng.integers(0, L - length))
            seq[s : s + length] = tract
            ann.append((name, s, s + length, "repeat", f"({cfg.repeat_units[ui]})n"))

        positions = []
        for s, e in cgis:
            k = rng.poisson(cfg.gcbox_per_cgi)
            if e - s > W:
                positions += list(rng.integers(s, e - W, k))
        n_out = rng.poisson(cfg.gcbox_outside_per_mb * L / 1e6)
        cand = rng.integers(0, L - W, n_out * 2)
        positions += [int(p) for p in cand[~cgi_mask[cand]][:n_out]]
        for p in positions:
            x = site if rng.random() < 0.5 else revcomp(site)
            seq[p : p + W] = mutate(x, cfg.gcbox_mut_rate, rng)
            ann.append((name, int(p), int(p) + W, "gcbox", cfg.gcbox_consensus))
        seqs[name] = seq
    return Genome(seqs, ann)


def cpg_oe(seq: np.ndarray) -> float:
    c = (seq == 1).mean()
    g = (seq == 2).mean()
    cg = ((seq[:-1] == 1) & (seq[1:] == 2)).mean()
    return float(cg / (c * g)) if c * g > 0 else float("nan")
