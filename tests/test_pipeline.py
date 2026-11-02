import json

import numpy as np
import yaml

from spidr.cli import main
from spidr.pipeline import run_all


def test_tiny_pipeline(tmp_path):
    cfg = {
        "out_dir": str(tmp_path / "run"), "results_dir": str(tmp_path / "res"),
        "genome": {"n_chrom": 3, "chrom_len": 150_000, "cgi_per_mb": 40, "seed": 2},
        "simulation": {"occupancy_per_mb": 50, "seed": 1},
        "splits": {"train": ["chr1"], "val": ["chr2"], "test": ["chr3"]},
        "export": {"modes": ["tile", "peaks_gcmatched"], "train_stride": 1000, "bedgraph": True},
        "ism": {"n_top_each": 1, "n_random": 1},
        "grammars": {
            "M0": [{"type": "pwm", "consensus": "GGGGCGGGG", "scale": 1.5}],
            "M1": [{"type": "composition", "window": 201, "beta": 20.0, "gc_ref": 0.42}],
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    s = run_all(p, n_jobs=2)
    res = tmp_path / "res"
    for f in ("summary.md", "summary.json", "fig_locus.png", "fig_designs.png", "fig_identifiability.png"):
        assert (res / f).exists()
    S = np.array(s["identifiability"]["contrib_pearson"])
    assert np.allclose(np.diag(S), 1.0)
    import h5py
    with h5py.File(tmp_path / "run/grammars/M0/data_tile.h5") as h5:
        assert h5["train/seq"].shape[1] == 2114 and h5["train/counts"].shape[1:] == (2, 1000)
    # compare CLI: oracle against itself gives r = 1
    run = tmp_path / "run"
    main(["compare", str(run / "ism_windows.npz"), str(run / "grammars/M0/oracle_ism.npz"),
          str(run / "grammars/M0/oracle_ism.npz")])
