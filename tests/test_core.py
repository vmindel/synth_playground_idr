import numpy as np
import pytest

from spidr import designs
from spidr.chec import ChecConfig
from spidr.genome import GenomeConfig, cpg_oe, generate_genome
from spidr.grammars import Grammar
from spidr.ism import contributions, ism
from spidr.seq import decode, encode, gc_fraction, revcomp
from spidr.simulate import Oracle, Simulator

SPECS = {
    "M0": [{"type": "pwm", "consensus": "GGGGCGGGG", "scale": 1.5}],
    "M1": [{"type": "composition", "window": 201, "beta": 20.0, "gc_ref": 0.42}],
    "M2": [{"type": "kmer_avidity", "motifs": ["GGGAGGG"], "max_mm": 1, "window": 201, "beta": 1.5}],
}


@pytest.fixture(scope="module")
def genome():
    return generate_genome(GenomeConfig(n_chrom=1, chrom_len=400_000, seed=3))


def test_encode_roundtrip():
    s = "ACGTTGCA"
    assert decode(encode(s)) == s
    assert decode(revcomp(encode("AACG"))) == "CGTT"


def test_genome_composition(genome):
    s = genome.seqs["chr1"]
    assert 0.38 < gc_fraction(s) < 0.52
    mask = np.ones(len(s), bool)
    for _, a, b, t, _ in genome.annotations:
        if t == "CGI":
            mask[a:b] = False
    assert cpg_oe(s[mask]) < 0.45  # depleted outside CGIs
    assert any(t == "CGI" for *_, t, _ in genome.annotations)


@pytest.mark.parametrize("name", list(SPECS))
def test_oracle_matches_genome_wide(genome, name):
    sim = Simulator(Grammar.from_spec(name, SPECS[name]), ChecConfig())
    sim.calibrate(genome, 50)
    full = sim.expected(genome.seqs["chr1"][None])[0]
    orc = Oracle(sim)
    for s0 in (10_000, 123_457, 300_001):
        w = genome.seqs["chr1"][s0 : s0 + 2114]
        np.testing.assert_allclose(orc(w[None])[0], full[s0 + 557 : s0 + 1557], rtol=1e-6, atol=1e-12)


def test_calibration_hits_target(genome):
    from scipy.special import expit
    sim = Simulator(Grammar.from_spec("M1", SPECS["M1"]), ChecConfig())
    sim.calibrate(genome, 50)
    s = sim.grammar.score(genome.seqs["chr1"][None])
    assert abs(expit(s + sim.mu).sum() - 50 * 0.4) / (50 * 0.4) < 1e-3
    assert abs(sim.expected(genome.seqs["chr1"][None]).sum() - ChecConfig().total_cuts) / 5e6 < 1e-6


def test_ism_finds_planted_motif(genome):
    sim = Simulator(Grammar.from_spec("M0", SPECS["M0"]), ChecConfig())
    sim.calibrate(genome, 50)
    rng = np.random.default_rng(0)
    x = np.where(rng.random(2114) < 0.5, 0, 3).astype(np.uint8)  # A/T background
    x[1050:1059] = encode("GGGGCGGGG")
    h = ism(Oracle(sim), x[None], 1000, 1110)
    c = contributions(h, x[None])[0]
    assert np.argmax(c) in range(1050, 1059)
    assert c[1050:1059].mean() > 10 * np.abs(np.r_[c[1000:1040], c[1070:1110]]).mean()


def test_designs_preserve_what_they_claim():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 4, 2114).astype(np.uint8)
    x[1000:1007] = encode("GGGAGGG")
    a, b = 900, 1200
    y = designs.gc_swap(x, a, b, rng)
    assert gc_fraction(y[a:b]) == gc_fraction(x[a:b]) and (y[:a] == x[:a]).all()
    y = designs.block_shuffle(x, a, b, rng)
    for s in range(a, b, 25):
        assert sorted(y[s : s + 25]) == sorted(x[s : s + 25])
    y = designs.kmer_scrub(x, a, b, rng)
    assert gc_fraction(y[a:b]) == gc_fraction(x[a:b])
    assert not designs.lowcomp_mask(y)[a:b].any()
    y = designs.gcbox_ko(np.concatenate([x[:1100], encode("GGGGCGGGG"), x[1109:]]), a, b, rng)
    assert not any(a <= c < b - 1 for c in designs.gcbox_centres(y))
