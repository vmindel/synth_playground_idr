"""Command line entry point: `spidr <command>`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(prog="spidr", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("run-all", help="full experiment from a YAML config")
    a.add_argument("config")
    a.add_argument("--jobs", type=int, default=4)
    a.add_argument("--force", action="store_true", help="recompute steps whose outputs exist")

    a = sub.add_parser("oracle-ism", help="oracle ISM for a run on arbitrary windows (npz with 'seqs')")
    a.add_argument("run_dir")
    a.add_argument("windows")
    a.add_argument("out")
    a.add_argument("--jobs", type=int, default=4)

    a = sub.add_parser("compare", help="agree model ISM with oracle ISM on the same windows")
    a.add_argument("windows", help="ism_windows.npz used for both")
    a.add_argument("oracle", help="oracle_ism.npz (key 'hyp')")
    a.add_argument("model", help="model ISM npz, key 'hyp' (N, L, 4): delta log total counts, 0 at ref")
    a.add_argument("--output-only", action="store_true", help="score only the central output window")

    a = sub.add_parser("seqlets", help="word enrichment in attribution seqlets")
    a.add_argument("windows")
    a.add_argument("ism", help="npz with key 'hyp'")

    args = p.parse_args(argv)

    if args.cmd == "run-all":
        from .pipeline import run_all
        run_all(args.config, n_jobs=args.jobs, skip_existing=not args.force)
        return

    W = np.load(args.windows, allow_pickle=True)
    seqs = W["seqs"]
    L = seqs.shape[1]
    if args.cmd == "oracle-ism":
        from .pipeline import oracle_ism
        from .simulate import Oracle
        o = Oracle.from_run(args.run_dir)
        r = o.sim.radius
        region = (o.flank - r, o.flank + o.output_len + r)
        hyp = oracle_ism(args.run_dir, seqs, region, args.jobs)
        np.savez_compressed(args.out, hyp=hyp, region=np.array(region))
        return

    from .analysis import agreement, seqlet_report
    from .ism import contributions
    flank = (L - 1000) // 2
    if args.cmd == "compare":
        ho, hm = np.load(args.oracle)["hyp"], np.load(args.model)["hyp"]
        if hm.shape != ho.shape:
            raise SystemExit(f"shape mismatch: model {hm.shape} vs oracle {ho.shape}")
        region = (flank, flank + 1000) if args.output_only else tuple(np.load(args.oracle)["region"])
        print(json.dumps(agreement(ho, hm, seqs, region), indent=2))
    elif args.cmd == "seqlets":
        h = np.load(args.ism)["hyp"]
        rep = seqlet_report(seqs, contributions(h, seqs), (flank, flank + 1000))
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
