"""Figures and a markdown summary for a pipeline run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter1d

from .designs import gcbox_centres, lowcomp_mask
from .ism import contributions
from .seq import is_gc
from .simulate import load_tracks

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASE = {0: "#109648", 1: "#255c99", 2: "#f7b32b", 3: "#d62839"}  # A C G T, sequence-logo convention
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#86b6ef", "#f0efec", "#ec9a9a", "#a83232"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blu", ["#f0efec", "#9ec5f4", "#3987e5", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _colors(names):
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}


def fig_locus(genome, run_dirs, chrom, out, span=30_000):
    ann = [a for a in genome.annotations if a[0] == chrom]
    cgis = [a for a in ann if a[3] == "CGI"]
    gggas = [a for a in ann if a[4] == "(GGGA)n"]
    centre = len(genome.seqs[chrom]) // 2
    if cgis and gggas:
        g_mid = np.array([(a[1] + a[2]) // 2 for a in gggas])
        best = min(cgis, key=lambda c: np.abs(g_mid - (c[1] + c[2]) // 2).min())
        near = g_mid[np.argmin(np.abs(g_mid - (best[1] + best[2]) // 2))]
        centre = ((best[1] + best[2]) // 2 + near) // 2
    s = max(0, centre - span // 2)
    e = min(len(genome.seqs[chrom]), s + span)
    names = list(run_dirs)
    col = _colors(names)
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(9, 1.1 * (len(names) + 1) + 0.6), sharex=True)
    x = np.arange(s, e) / 1e3
    gc = uniform_filter1d(is_gc(genome.seqs[chrom][s:e]).astype(float), 201)
    ax = axes[0]
    ax.plot(x, gc, color=INK, lw=1)
    ax.set_ylabel("GC\n(201 bp)", rotation=0, ha="right", va="center")
    ax.set_title(f"{chrom}:{s:,}-{e:,}   simulated ChEC, replicate 1, 50 bp bins")
    kinds = {"CGI": ("#52514e", "CGI"), "gcbox": (SERIES[0], "GC-box"), "repeat": (SERIES[2], "repeat")}
    for c, a, b, t, n in ann:
        if b > s and a < e and t in kinds:
            y = {"CGI": 0.18, "gcbox": 0.12, "repeat": 0.06}[t]
            lw = 6 if t == "CGI" else 4
            colr = SERIES[1] if n == "(GGGA)n" else kinds[t][0]
            ax.plot([max(a, s) / 1e3, min(b, e) / 1e3], [y, y], color=colr, lw=lw, solid_capstyle="butt",
                    transform=ax.get_xaxis_transform())
    ax.text(1.0, 1.02, "bars: CGI (grey) · GC-box (blue) · (GGGA)n (orange) · other repeats (green)",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7)
    for ax, g in zip(axes[1:], names):
        cnt = load_tracks(run_dirs[g], chrom, ["counts_rep1"])["counts_rep1"][s:e]
        nb = len(cnt) // 50
        b = cnt[: nb * 50].reshape(nb, 50).sum(1)
        xb = (s + np.arange(nb) * 50 + 25) / 1e3
        ax.fill_between(xb, 0, b, step="mid", color=col[g], lw=0)
        ax.set_ylabel(g, rotation=0, ha="right", va="center", color=INK)
        ax.set_ylim(0, max(b.max(), 1) * 1.05)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    axes[-1].set_xlabel("position (kb)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_attribution(W, hyps, out, flank, output_len, zoom=250):
    seqs = W["seqs"]
    names = list(hyps)
    lo, hi = flank, flank + output_len
    best, best_score = 0, -1
    for i, s in enumerate(seqs):
        gb = [c for c in gcbox_centres(s) if lo <= c < hi]
        lc = np.flatnonzero(lowcomp_mask(s)[lo:hi]) + lo
        score = (len(gb) > 0) + (len(lc) > 0) + 0.01 * len(gb)
        if score > best_score:
            best, best_score = i, score
    s = seqs[best]
    gb = [c for c in gcbox_centres(s) if lo <= c < hi]
    lc = np.flatnonzero(lowcomp_mask(s)[lo:hi]) + lo
    anchors = gb[:1] + list(lc[:1])
    mid = int(np.mean(anchors)) if anchors else flank + output_len // 2
    a = int(np.clip(mid - zoom // 2, lo, hi - zoom))
    b = a + zoom
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.15 * len(names) + 0.6), sharex=True)
    x = np.arange(a, b)
    for ax, g in zip(np.atleast_1d(axes), names):
        c = contributions(hyps[g][best : best + 1], seqs[best : best + 1])[0][a:b]
        ax.bar(x, c, width=1.0, color=[BASE[int(v)] for v in s[a:b]], lw=0)
        ax.axhline(0, color="#c3c2b7", lw=0.6)
        for cc in gb:
            if a <= cc < b:
                ax.axvspan(cc - 4.5, cc + 4.5, color=SERIES[0], alpha=0.08, lw=0)
        m = lowcomp_mask(s)
        for p in np.flatnonzero(m[a:b]) + a:
            ax.axvspan(p - 0.5, p + 0.5, color=SERIES[1], alpha=0.10, lw=0)
        ax.set_ylabel(g, rotation=0, ha="right", va="center")
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    axes[0].set_title(f"Oracle ISM contributions, same sequence under each grammar "
                      f"({W['chrom']}, window {best}); shading: GC-box (blue), GGGAGGG<=1mm (orange)")
    axes[0].text(1.0, 1.02, "bars coloured A green · C blue · G yellow · T red", transform=axes[0].transAxes,
                 ha="right", va="bottom", color=MUTED, fontsize=7)
    axes[-1].set_xlabel("position in input window (bp)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_seqlets(summary, out):
    an = summary["analysis"]
    names = list(an)
    col = _colors(names)
    panels = [("gcbox_core_rate", "gcbox_core_bg", "seqlets containing GGGCGG"),
              ("lowcomp_rate", "lowcomp_bg", "seqlets containing GGGAGGG (<=1 mm)"),
              ("seqlet_gc", "bg_gc", "GC fraction of seqlets")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    x = np.arange(len(names))
    for ax, (k, kb, title) in zip(axes, panels):
        v = [an[g]["seqlets"][k] for g in names]
        vb = [an[g]["seqlets"][kb] for g in names]
        ax.bar(x - 0.19, v, 0.36, color=[col[g] for g in names], label="seqlets")
        ax.bar(x + 0.19, vb, 0.36, color="#c3c2b7", label="random positions")
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Words recovered from oracle attributions (top 1% smoothed contribution peaks, ±7 bp)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _heatmap(ax, M, rows, cols, cmap, vmin, vmax, fmt="{:.2f}"):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            dark = abs(v - (vmin + vmax) / 2) > 0.35 * (vmax - vmin)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark else INK)
    ax.spines[:].set_visible(False)
    return im


def fig_identifiability(summary, out):
    d = summary["identifiability"]
    M = np.asarray(d["contrib_pearson"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = _heatmap(ax, M, d["grammars"], d["grammars"], SEQUENTIAL, 0, 1)
    fig.colorbar(im, ax=ax, shrink=0.8, label="median per-window Pearson r")
    ax.set_title("How alike are perfect attributions\nunder different grammars?")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_designs(summary, out):
    d = summary["designs"]
    M = np.asarray(d["median_log2fc"])
    lim = max(1.0, np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = _heatmap(ax, M, d["grammars"], d["designs"], DIVERGING.reversed(), -lim, lim, "{:+.2f}")
    fig.colorbar(im, ax=ax, shrink=0.85, label="median log2 FC, total signal")
    ax.set_title("Predicted effect of 300 bp core perturbations (same sequences, each grammar)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_markdown(summary, res_dir: Path):
    g = summary["grammars"]
    an = summary["analysis"]
    L = ["# spidr v0 results", "",
         f"ISM windows: {summary['n_ism_windows']} shared windows on the test chromosome "
         f"(top windows per grammar + random). Mutated region {summary['ism_region']}; "
         "analyses scored on the 1000 bp output window.", "",
         "## Grammars", "", "| grammar | components | mu | k_sat | radius (bp) |", "|---|---|---|---|---|"]
    for n, s in g.items():
        p = s["params"]
        comps = "; ".join(", ".join(f"{k}={v}" for k, v in c.items()) for c in p["grammar"])
        L.append(f"| {n} | {comps} | {p['mu']:.2f} | {p['k_sat']:.3g} | {p['radius']} |")
    L += ["", "## Training sets: does the sampling regime hide composition?", "",
          "Spearman rho between input-window GC and log total signal, per split.", "",
          "| grammar | mode | train n | train rho | test n | test rho |", "|---|---|---|---|---|---|"]
    for n, s in g.items():
        for mode in ("tile", "peaks_gcmatched"):
            if mode in s:
                t = s[mode]
                L.append(f"| {n} | {mode} | {t['train']['n']} | {t['train']['spearman_gc_logsignal']:+.2f} | "
                         f"{t['test']['n']} | {t['test']['spearman_gc_logsignal']:+.2f} |")
    L += ["", "## Oracle attributions", "",
          "Concentration = share of |contribution| in the top 5% of positions (motif-like = high).", "",
          "| grammar | concentration | contrib at G/C | contrib at A/T | seqlets | GGGCGG in seqlets (bg) | "
          "GGGAGGG<=1mm in seqlets (bg) | top 6-mers (log2 enr) |", "|---|---|---|---|---|---|---|---|"]
    for n, a in an.items():
        s = a["seqlets"]
        tk = ", ".join(f"{k} ({e:.1f})" for k, e, _ in s["top_kmers"][:5])
        L.append(f"| {n} | {a['concentration_top5pct']:.2f} | {a['mean_contrib_GC']:+.4f} | "
                 f"{a['mean_contrib_AT']:+.4f} | {s['n_seqlets']} | {s['gcbox_core_rate']:.2f} "
                 f"({s['gcbox_core_bg']:.2f}) | {s['lowcomp_rate']:.2f} ({s['lowcomp_bg']:.2f}) | {tk} |")
    d = summary["identifiability"]
    L += ["", "## Identifiability: oracle-vs-oracle attribution agreement", "",
          "| | " + " | ".join(d["grammars"]) + " |", "|---" * (len(d["grammars"]) + 1) + "|"]
    for n, row in zip(d["grammars"], d["contrib_pearson"]):
        L.append(f"| {n} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    d = summary["designs"]
    L += ["", "## Design discrimination (median log2 FC of total signal)", "",
          "| | " + " | ".join(d["designs"]) + " |", "|---" * (len(d["designs"]) + 1) + "|"]
    for n, row in zip(d["grammars"], d["median_log2fc"]):
        L.append(f"| {n} | " + " | ".join(f"{v:+.2f}" for v in row) + " |")
    L += ["", "## Figures", "", "![locus](fig_locus.png)", "", "![attribution](fig_attribution.png)", "",
          "![seqlets](fig_seqlets.png)", "", "![identifiability](fig_identifiability.png)", "",
          "![designs](fig_designs.png)", ""]
    (res_dir / "summary.md").write_text("\n".join(L))


def make_report(genome, run_dirs, W, hyps, summary, res_dir, input_len, output_len, test_chrom):
    res_dir = Path(res_dir)
    flank = (input_len - output_len) // 2
    fig_locus(genome, run_dirs, test_chrom, res_dir / "fig_locus.png")
    fig_attribution(W, hyps, res_dir / "fig_attribution.png", flank, output_len)
    fig_seqlets(summary, res_dir / "fig_seqlets.png")
    fig_identifiability(summary, res_dir / "fig_identifiability.png")
    fig_designs(summary, res_dir / "fig_designs.png")
    write_markdown(summary, res_dir)
