"""End-to-end v0 experiment: genome -> grammars -> datasets -> oracle ISM -> analyses -> report."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from .analysis import agreement, concentration, gc_attribution, seqlet_report
from .chec import ChecConfig
from .dataset import export_dataset, output_starts, write_bedgraph
from .designs import design_matrix
from .genome import Genome, GenomeConfig, generate_genome
from .ism import contributions, ism, to_modisco
from .simulate import Oracle, load_tracks, run_simulation


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ism_chunk(args):
    run_dir, seqs, a, b = args
    return ism(Oracle.from_run(run_dir), seqs, a, b)


def oracle_ism(run_dir, seqs: np.ndarray, region: tuple[int, int], n_jobs: int = 1) -> np.ndarray:
    chunks = np.array_split(np.arange(len(seqs)), max(1, min(n_jobs, len(seqs))))
    jobs = [(str(run_dir), seqs[c], *region) for c in chunks if len(c)]
    if n_jobs <= 1:
        return np.concatenate([_ism_chunk(j) for j in jobs])
    with ProcessPoolExecutor(n_jobs) as ex:
        return np.concatenate(list(ex.map(_ism_chunk, jobs)))


def select_ism_windows(genome: Genome, run_dirs: dict, chrom: str, input_len: int, output_len: int,
                       n_top_each: int, n_random: int, seed: int = 0) -> dict:
    """Shared windows (same sequences for every grammar): top windows of each grammar on a held-out
    chromosome, plus random windows."""
    rng = np.random.default_rng(seed)
    starts = output_starts(len(genome.seqs[chrom]), input_len, output_len, output_len)
    chosen, label = [], []
    for g, rd in run_dirs.items():
        e = load_tracks(rd, chrom, ["expected"])["expected"]
        tot = e[starts[:, None] + np.arange(output_len)].sum(1)
        picked = 0
        for i in np.argsort(-tot):
            if i not in chosen:
                chosen.append(int(i))
                label.append(f"top_{g}")
                picked += 1
            if picked == n_top_each:
                break
    rest = np.setdiff1d(np.arange(len(starts)), chosen)
    for i in rng.choice(rest, min(n_random, len(rest)), replace=False):
        chosen.append(int(i))
        label.append("random")
    st = starts[chosen]
    flank = (input_len - output_len) // 2
    seqs = genome.seqs[chrom][st[:, None] - flank + np.arange(input_len)]
    return {"seqs": seqs, "start": st, "chrom": chrom, "label": np.array(label)}


def run_all(config_path: str | Path, n_jobs: int = 4, skip_existing: bool = True) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["out_dir"])
    res_dir = Path(cfg.get("results_dir", out / "results"))
    out.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    Lin, Lout = cfg.get("input_len", 2114), cfg.get("output_len", 1000)
    flank = (Lin - Lout) // 2

    gdir = out / "genome"
    gcfg = GenomeConfig.from_dict(cfg.get("genome"))
    if skip_existing and (gdir / "genome.fa").exists():
        genome = Genome.load(gdir)
        _log(f"loaded genome {gdir}")
    else:
        genome = generate_genome(gcfg)
        genome.save(gdir, gcfg)
        _log(f"genome: {genome.total_len / 1e6:.1f} Mb -> {gdir}")

    chec = ChecConfig.from_dict(cfg.get("chec"))
    simc = cfg.get("simulation", {})
    splits = cfg["splits"]
    exp_cfg = cfg.get("export", {})
    run_dirs, summary = {}, {"grammars": {}}
    for gname, spec in cfg["grammars"].items():
        rd = out / "grammars" / gname
        run_dirs[gname] = rd
        if not (skip_existing and (rd / "params.json").exists()):
            run_simulation(genome, gname, spec, chec, rd, simc.get("occupancy_per_mb", 50.0),
                           simc.get("seed", 0), Lin, Lout)
            _log(f"simulated {gname}")
        gs = {"params": json.loads((rd / "params.json").read_text())}
        for mode in exp_cfg.get("modes", ["tile"]):
            path = rd / f"data_{mode}.h5"
            if not (skip_existing and path.exists() and (rd / f"data_{mode}.json").exists()):
                s = export_dataset(genome, rd, splits, path, mode, exp_cfg.get("train_stride", 500),
                                   exp_cfg.get("peak_quantile", 0.9), exp_cfg.get("neg_ratio", 1.0),
                                   simc.get("seed", 0))
                (rd / f"data_{mode}.json").write_text(json.dumps(s, indent=2))
                _log(f"exported {gname}/{path.name}")
            gs[mode] = json.loads((rd / f"data_{mode}.json").read_text())
        if exp_cfg.get("bedgraph", False):
            for r in range(chec.n_replicates):
                bg = rd / f"counts_rep{r + 1}.bedgraph.gz"
                if not (skip_existing and bg.exists()):
                    write_bedgraph(genome, rd, f"counts_rep{r + 1}", bg)
        summary["grammars"][gname] = gs

    ic = cfg.get("ism", {})
    test_chrom = splits["test"][0]
    wpath = out / "ism_windows.npz"
    if skip_existing and wpath.exists():
        W = dict(np.load(wpath, allow_pickle=True))
    else:
        W = select_ism_windows(genome, run_dirs, test_chrom, Lin, Lout, ic.get("n_top_each", 12),
                               ic.get("n_random", 40), ic.get("seed", 0))
        np.savez_compressed(wpath, **W)
    seqs = W["seqs"]
    radius = max(json.loads((rd / "params.json").read_text())["radius"] for rd in run_dirs.values())
    region = (flank - radius, flank + Lout + radius)
    hyps = {}
    for gname, rd in run_dirs.items():
        p = rd / "oracle_ism.npz"
        if skip_existing and p.exists():
            hyps[gname] = np.load(p)["hyp"]
            continue
        t = time.time()
        hyps[gname] = oracle_ism(rd, seqs, region, n_jobs)
        np.savez_compressed(p, hyp=hyps[gname], region=np.array(region))
        ohe, attr = to_modisco(seqs, hyps[gname])
        np.savez_compressed(rd / "modisco_ohe.npz", ohe)
        np.savez_compressed(rd / "modisco_attr.npz", attr)
        _log(f"oracle ISM {gname}: {len(seqs)} windows in {time.time() - t:.0f}s")

    # analyses are scored over the output window, where attribution is meaningful for the target
    out_region = (flank, flank + Lout)
    an = {}
    for gname in run_dirs:
        c = contributions(hyps[gname], seqs)
        an[gname] = {"seqlets": seqlet_report(seqs, c, out_region),
                     "concentration_top5pct": concentration(c, out_region),
                     **gc_attribution(c, seqs, out_region)}
    names = list(run_dirs)
    S = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            S[i, j] = agreement(hyps[a], hyps[b], seqs, region)["contrib_pearson_median"]
    oracles = {g: Oracle.from_run(rd) for g, rd in run_dirs.items()}
    top = np.array([lbl.startswith("top_") for lbl in W["label"]])
    dc = cfg.get("designs", {})
    gn, dn, M, designed = design_matrix(oracles, seqs[top], dc.get("core_len", 300), dc.get("seed", 0))
    np.savez_compressed(out / "design_sequences.npz", original=seqs[top],
                        **{f"design_{d}": v for d, v in designed.items()})

    summary.update({"analysis": an, "identifiability": {"grammars": names, "contrib_pearson": S.tolist()},
                    "designs": {"grammars": gn, "designs": dn, "median_log2fc": M.tolist()},
                    "ism_region": list(region), "n_ism_windows": int(len(seqs))})
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    from .report import make_report
    make_report(genome, run_dirs, W, hyps, summary, res_dir, Lin, Lout, test_chrom)
    _log(f"report -> {res_dir}")
    return summary
