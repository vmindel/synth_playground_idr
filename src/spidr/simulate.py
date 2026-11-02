"""Grammar + ChEC model -> genome-wide tracks, calibration, and an exact window oracle."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import expit

from .chec import ChecConfig, mnase_bias, pcr_gc_bias, sample_counts, tether_kernel
from .genome import Genome
from .grammars import Grammar


class Simulator:
    def __init__(self, grammar: Grammar, chec: ChecConfig, mu: float | None = None,
                 k_sat: float | None = None, depth_scale: float | None = None):
        self.grammar = grammar
        self.chec = chec
        self.kernel = tether_kernel(chec)
        self.mu, self.k_sat, self.depth_scale = mu, k_sat, depth_scale

    @property
    def radius(self) -> int:
        """Context (bp) on each side that determines expected cuts at a position."""
        R = len(self.kernel) // 2
        return max(self.grammar.radius + R, self.chec.gc_bias_window // 2, 1)

    def _occupancy(self, score: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(score), expit(score + self.mu), 0.0)

    def _rate_biased(self, seq: np.ndarray, occ: np.ndarray) -> np.ndarray:
        rate = convolve1d(occ, self.kernel, axis=-1, mode="constant", cval=0.0)
        return rate * mnase_bias(seq, self.chec.mnase_at_pref)

    def _finish(self, seq: np.ndarray, x: np.ndarray) -> np.ndarray:
        c = -np.expm1(-self.k_sat * x) if self.k_sat > 0 else x
        return self.depth_scale * c * pcr_gc_bias(seq, self.chec.gc_bias_sd, self.chec.gc_bias_window)

    def tracks(self, seq: np.ndarray) -> dict[str, np.ndarray]:
        seq = np.atleast_2d(seq)
        comps = self.grammar.score_components(seq)
        score = sum(comps.values())
        occ = self._occupancy(score)
        expected = self._finish(seq, self._rate_biased(seq, occ))
        out = {"score": score, "occupancy": occ, "expected": expected}
        out.update({f"score_{k}": v for k, v in comps.items()})
        return out

    def expected(self, seq: np.ndarray) -> np.ndarray:
        seq = np.atleast_2d(seq)
        occ = self._occupancy(self.grammar.score(seq))
        return self._finish(seq, self._rate_biased(seq, occ))

    def calibrate(self, genome: Genome, occupancy_per_mb: float) -> None:
        """Set mu so expected total occupancy matches the target (equalises binding mass across
        grammars), k_sat from the saturation setting, and depth_scale from total_cuts."""
        scores = {c: self.grammar.score(s[None])[0] for c, s in genome.seqs.items()}
        target = occupancy_per_mb * genome.total_len / 1e6

        def total(mu):
            return sum(np.where(np.isfinite(v), expit(v + mu), 0.0).sum() for v in scores.values())

        lo, hi = -80.0, 40.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if total(mid) < target else (lo, mid)
        self.mu = (lo + hi) / 2

        xs = {}
        for c, s in genome.seqs.items():
            xs[c] = self._rate_biased(s[None], self._occupancy(scores[c][None]))[0]
        allx = np.concatenate(list(xs.values()))
        q = np.quantile(allx, 0.9999)
        self.k_sat = self.chec.saturation / q if self.chec.saturation > 0 and q > 0 else 0.0
        self.depth_scale = 1.0
        tot = sum(self._finish(genome.seqs[c][None], xs[c][None]).sum() for c in genome.seqs)
        self.depth_scale = self.chec.total_cuts / tot

    def params(self) -> dict:
        return {"mu": self.mu, "k_sat": self.k_sat, "depth_scale": self.depth_scale, "radius": self.radius}


class Oracle:
    """Noise-free expected cut profile over the central output window of an input window.
    Exact w.r.t. the genome-wide simulation as long as simulator.radius <= flank."""

    def __init__(self, sim: Simulator, input_len: int = 2114, output_len: int = 1000):
        self.sim, self.input_len, self.output_len = sim, input_len, output_len
        self.flank = (input_len - output_len) // 2
        if sim.radius > self.flank:
            raise ValueError(f"model context radius {sim.radius} bp exceeds flank {self.flank} bp")

    def __call__(self, seqs: np.ndarray, batch_size: int = 512) -> np.ndarray:
        seqs = np.atleast_2d(seqs)
        out = np.empty((len(seqs), self.output_len))
        for i in range(0, len(seqs), batch_size):
            e = self.sim.expected(seqs[i : i + batch_size])
            out[i : i + batch_size] = e[:, self.flank : self.flank + self.output_len]
        return out

    @classmethod
    def from_run(cls, run_dir: str | Path) -> "Oracle":
        p = json.loads((Path(run_dir) / "params.json").read_text())
        sim = Simulator(Grammar.from_spec(p["grammar_name"], p["grammar"]), ChecConfig.from_dict(p["chec"]),
                        p["mu"], p["k_sat"], p["depth_scale"])
        return cls(sim, p["input_len"], p["output_len"])


def run_simulation(genome: Genome, grammar_name: str, grammar_spec: list[dict], chec: ChecConfig,
                   out_dir: str | Path, occupancy_per_mb: float = 2000.0, seed: int = 0,
                   input_len: int = 2114, output_len: int = 1000) -> Simulator:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim = Simulator(Grammar.from_spec(grammar_name, grammar_spec), chec)
    if sim.radius > (input_len - output_len) // 2:
        raise ValueError(f"context radius {sim.radius} exceeds flank {(input_len - output_len) // 2}")
    sim.calibrate(genome, occupancy_per_mb)
    rng = np.random.default_rng(seed)
    with h5py.File(out / "tracks.h5", "w") as h5:
        for c, s in genome.seqs.items():
            g = h5.create_group(c)
            tr = sim.tracks(s[None])
            for k, v in tr.items():
                v = v[0]
                if k.startswith("score"):
                    v = np.where(np.isfinite(v), v, np.nan)
                g.create_dataset(k, data=v.astype(np.float32), compression="gzip")
            for r in range(chec.n_replicates):
                cnt = sample_counts(tr["expected"][0], chec, rng)
                g.create_dataset(f"counts_rep{r + 1}", data=cnt.astype(np.int32), compression="gzip")
    params = {"grammar_name": grammar_name, "grammar": grammar_spec, "chec": chec.to_dict(),
              "occupancy_per_mb": occupancy_per_mb, "seed": seed, "input_len": input_len,
              "output_len": output_len, **sim.params()}
    (out / "params.json").write_text(json.dumps(params, indent=2))
    return sim


def load_tracks(run_dir: str | Path, chrom: str, names: list[str] | None = None) -> dict[str, np.ndarray]:
    with h5py.File(Path(run_dir) / "tracks.h5", "r") as h5:
        g = h5[chrom]
        return {k: g[k][:] for k in (names or list(g))}
