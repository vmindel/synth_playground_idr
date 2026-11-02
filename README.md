# spidr: planted-grammar playground for IDR-directed binding

A synthetic genome where we control the binding rules, a ChEC-seq observation model on top, and
an exact oracle. It asks one question: **if IDR-directed targeting runs through DNA composition
(elevated GC, SP/KLF-like low-complexity words such as GGGAGGG) rather than a discrete motif, can a
seq2profile model plus ISM show that, or do the attributions report a motif regardless?**

The simulator plants competing rules that produce similar binding maps, because in real sequence
"GC-rich", "many weak GGGAGGG sites" and "GC-box motif" are close to the same signal (GGGAGGG is
one mismatch from the GGGCGG core, and all three are enriched in CpG islands).

## Model

**Genome** (`genome.py`). Isochore-scale GC (five-family mixture, approximate human spread),
kb-scale local GC fluctuation, CpG depletion by CG->TG/CA outside CpG islands, CGIs biased towards
GC-rich isochores, imperfect simple repeats including (GGGA)n and (GA)n, and planted GC-box sites
enriched in CGIs. Default 5 x 2 Mb.

**Grammars** (`grammars.py`). Every position is a candidate site centre with score s(i) in kT,
occupancy = expit(s(i) + mu), and mu calibrated so total occupancy is equal across grammars.

| term | form | stands for |
|---|---|---|
| `pwm` | scale x PWM log-odds, strands combined by logaddexp | DBD motif |
| `composition` | beta x (GC fraction in a 201 bp window - gc_ref) | IDR reads bulk composition |
| `kmer_avidity` | beta x log(1 + sum over window of exp(eps - lam x mismatches)) for matches to GGGAGGG with <=1 mm | IDR samples many weak low-complexity sites (multivalent) |

Terms add in free energy, so `M3` grammars (DBD + IDR) multiply affinities.

**ChEC** (`chec.py`). Occupancy convolved with a bimodal tether kernel (cuts 25 +/- 12 bp from the
site, 8 bp protected footprint), MNase A/T cutting preference (pushes against a GC signal), saturation
with digestion time, PCR GC bias, negative-binomial counts, unstranded, two replicates.
No free-MNase background.

**Oracle** (`simulate.Oracle`). For any 2114 bp input it returns the noise-free expected cuts over
the central 1000 bp. The full context radius (grammar window + tether kernel) is checked to fit
inside the 557 bp flank, so the oracle is exact with respect to the genome-wide simulation
(tested). Oracle ISM is the ceiling any trained model's ISM can reach.

## Run

```bash
pip install -e .[test]
```

```bash
spidr run-all configs/v0.yaml --jobs 4
```

Outputs go to `runs/v0/` (data, not committed) and `results/v0/` (figures + `summary.md`).
Steps whose outputs exist are skipped; `--force` recomputes.

```bash
pytest -q
```

## Training Cherimoya on it

Each grammar gets `runs/v0/grammars/<name>/data_tile.h5` and `data_peaks_gcmatched.h5`:

```
<split>/seq       (N, 2114) uint8, A=0 C=1 G=2 T=3   (one-hot with np.eye(4)[seq])
<split>/counts    (N, 2, 1000) int32, observed unstranded cuts per replicate
<split>/expected  (N, 1000) float32, noise-free truth
<split>/chrom, start, is_peak
attrs: input_len, output_len, chroms (JSON), grammar, mode
```

Splits by chromosome: chr1-3 train, chr4 val, chr5 test. `genome/genome.fa` and
`genome.fa.fai.sizes` are there too, and `export.bedgraph: true` writes per-replicate bedGraphs if
your loader wants FASTA + bigWig.

The two sampling modes are an experiment in themselves. `peaks_gcmatched` (top 10% windows +
GC-matched background, the BPNet/ChromBPNet recipe) removes between-window GC differences from the
contrast, so a model trained that way sees less of a composition rule than one trained on tiles.
`summary.md` reports GC-signal correlation in each set.

## Scoring a trained model against the truth

Run ISM on `runs/v0/ism_windows.npz["seqs"]` (same windows, same order) with the summary
**delta log(total predicted counts over the 1000 bp output)** for every substitution, 0 at the
reference base, shape (N, 2114, 4), saved as `np.savez(path, hyp=...)`. `spidr.ism.ism` does this for
any `predict_fn(uint8 (B, 2114)) -> (B, 1000) counts`; see the docstring for a model wrapper.

```bash
spidr compare runs/v0/ism_windows.npz runs/v0/grammars/M1_composition/oracle_ism.npz my_m1_ism.npz
```

```bash
spidr seqlets runs/v0/ism_windows.npz my_m1_ism.npz
```

`modisco_ohe.npz` / `modisco_attr.npz` per grammar hold the oracle attributions in (N, 4, L) layout
for tfmodisco-lite, to check what a standard motif caller reports on a pure composition rule.

## Perturbation designs

`designs.py` rewrites a 300 bp core around each summit in native context: `gc_swap` (random C<->G,
GC exact, G-runs broken), `block_shuffle` (25 bp blocks), `kmer_scrub` (removes GGGAGGG<=1mm with
GC-class-preserving swaps), `gcbox_ko`, `at_backfill` (keep words, drop GC), `gc_backfill_scrubbed`
(raise GC, no words), `kmer_implant` (3 x GGGAGGG at native GC). The grammar x design matrix shows which
constructs separate the hypotheses; `runs/v0/design_sequences.npz` holds the sequences for scoring
with a trained model, and the best-separating classes are MPBA library candidates.

## Caveats

- Every position is a site and overlapping sites are not excluded, so occupancy is phenomenological,
  not a lattice model. Fine for relative signal, not for absolute numbers of molecules.
- The GC-box PWM is consensus-derived, not a measured SP/KLF matrix. `pwm_file:` takes a JASPAR file.
- No chromatin accessibility or nucleosomes yet. Nucleosome occupancy also depends on GC, so adding it
  is the next confound worth planting.
- Depth defaults (5 M cuts over 10 Mb) are far above real per-bp ChEC depth over a mammalian genome;
  sweep `chec.total_cuts` to see where recovery breaks.
