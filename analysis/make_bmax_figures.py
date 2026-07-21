"""Figures for the bMax selection-rule study (analysis/bMax_analysis.md).

Consumes the bMax-study sweeps (run_sweeps.py --only
bmax_fine,sigma_collapse,bmax_vs_L) plus the Python objective replica
(lcd_distance.py) for the information-vs-scale curves.

Usage:
    python analysis/make_bmax_figures.py --results <dir-with-results.csv> \
        --outdir <figure-dir>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcd_distance as lcd  # noqa: E402

# same palette conventions as make_figures.py
BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"
GREEN = "#008300"
VIOLET = "#4a3aa7"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e8e6"
BAND = "#dcebfb"          # shaded recommended-bMax band

SIGMA_COLORS = [BLUE, AQUA, GREEN, VIOLET, MAGENTA, ORANGE]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10,
    "lines.linewidth": 2.0,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_samples(row, results_dir):
    name = os.path.basename(str(row["samples_file"]).replace("\\", "/"))
    cand = os.path.join(results_dir, "samples", name)
    path = cand if os.path.exists(cand) else str(row["samples_file"])
    return np.loadtxt(path, delimiter=",")


# ---------------------------------------------------------------------------
# 1. What the kernel scale b actually "sees"
# ---------------------------------------------------------------------------

def fig_role(df, outdir, results_dir):
    r = df[(df.experiment == "bmax_fine") & (df.bMax == 100)
           & (df.seed == 42)].iloc[0]
    x = load_samples(r, results_dir)
    m = np.linspace(-6, 6, 600)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=False)
    for ax, b in zip(axes, (0.3, 2.0, 20.0)):
        fg = lcd.lcd_gaussian(m, b, [1.0, 1.0])
        fd = lcd.lcd_dirac(m, b, x)
        ax.plot(m, fg, color=INK2, lw=2.2, label="target Gaussian")
        ax.plot(m, fd, color=BLUE, lw=1.8, label="40-sample Dirac mixture")
        reldiff = np.max(np.abs(fg - fd)) / np.max(fg)
        ax.set_title(f"kernel width  b = {b:g}·σ\n"
                     f"max difference ≈ {100 * reldiff:.2g}% of peak",
                     fontsize=10)
        ax.set_xlabel("position along an axis")
        style(ax)
    axes[0].set_ylabel("smoothed mass seen at scale b")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("The distance compares the two distributions smoothed at every "
                 "kernel width b from 0 to bMax\n"
                 "fine scales see individual samples — by b ≈ 20σ the two are "
                 "indistinguishable, so scales beyond that add (almost) nothing",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p = os.path.join(outdir, "fig_bmax_role.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 2. Information vs cutoff: discrimination curves from the objective replica
# ---------------------------------------------------------------------------

def fig_information(df, outdir, results_dir):
    r = df[(df.experiment == "bmax_fine") & (df.bMax == 100)
           & (df.seed == 42)].iloc[0]
    x = load_samples(r, results_dir)
    rng = np.random.default_rng(0)
    jit = 0.03 * rng.standard_normal(x.shape)
    jit -= jit.mean(axis=0, keepdims=True)     # keep the perturbation mean-free
    perts = [
        ("shrunk ×0.8  (covariance −36%)", x * 0.8, RED),
        ("inflated ×1.05  (covariance +10%)", x * 1.05, VIOLET),
        ("jittered ±0.03  (clumpier, same moments)", x + jit, AQUA),
    ]

    Bs = np.logspace(np.log10(0.5), np.log10(1200), 32)
    deltas, asymptote = {}, {}
    for name, xp, _ in perts:
        deltas[name] = np.array(
            [lcd.distance(xp, [1, 1], B) - lcd.distance(x, [1, 1], B)
             for B in Bs])
        # far-field reference, well beyond the plotted range
        asymptote[name] = (lcd.distance(xp, [1, 1], 6000.0)
                           - lcd.distance(x, [1, 1], 6000.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # (a) raw penalty of each wrong set vs cutoff B
    ax = axes[0]
    for name, _, colr in perts:
        ax.plot(Bs, deltas[name], color=colr, label=name)
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel("integration cutoff  bMax  (σ = 1)")
    ax.set_ylabel("D(wrong set) − D(optimal set)")
    ax.set_title("Penalty the objective assigns to a wrong point set\n"
                 "negative = the WRONG set is preferred", fontsize=10)
    i_neg = np.where(deltas[perts[0][0]] < 0)[0]
    if len(i_neg):
        ax.annotate("bMax ≲ 4σ prefers the\ncollapsed set → points\nshrink "
                    "inward (failure mode 1)",
                    xy=(Bs[i_neg[-1]], deltas[perts[0][0]][i_neg[-1]]),
                    xytext=(9, -2e-3), fontsize=9, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.legend(frameon=False, fontsize=8.5, loc="upper center")
    style(ax)

    # (b) share of the discriminating signal still missing at cutoff B
    # (plotted for 3sigma..300sigma: below that the penalty overshoots its
    # asymptote so a "missing fraction" reading makes no sense; above it the
    # curve crosses zero and |.| would show a meaningless sign-flip dip)
    ax = axes[1]
    lo = (Bs >= 3.0) & (Bs <= 300.0)
    for name, _, colr in perts[1:]:            # mean-free, sign-stable curves
        missing = np.abs(asymptote[name] - deltas[name]) / abs(asymptote[name])
        ax.plot(Bs[lo], np.clip(missing[lo], 1e-7, None), color=colr,
                label=name)
    bb = Bs[lo].copy()
    ax.plot(bb, 40.0 / bb ** 2, color=INK2, ls="--", lw=1.4,
            label="∝ 1/bMax²  (theory: tail of the b-integral)")
    d = deltas[perts[1][0]]
    for Bmark, dx, dy in ((50, 0.28, 12), (100, 1.8, 3)):
        i = np.argmin(np.abs(Bs - Bmark))
        val = abs(asymptote[perts[1][0]] - d[i]) / abs(asymptote[perts[1][0]])
        ax.annotate(f"bMax = {Bmark}σ:\n≈{100 * val:.1f}% still missing",
                    xy=(Bs[i], val), xytext=(Bs[i] * dx, val * dy),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("integration cutoff  bMax  (σ = 1)")
    ax.set_ylabel("share of the penalty still missing")
    ax.set_title("Discriminating information beyond cutoff bMax\n"
                 "decays as 1/bMax² once bMax ≫ σ", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    style(ax)

    fig.suptitle("Why moderate bMax is enough: the useful signal saturates\n"
                 "(L = 40 optimal set, seed 42; Python objective replica "
                 "matches the C++ code to 1e-14)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    p = os.path.join(outdir, "fig_bmax_information.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 3. Measured quality vs bMax at L=40 (fine grid)
# ---------------------------------------------------------------------------

def fig_quality(df, outdir):
    f = df[df.experiment == "bmax_fine"]
    g = f.groupby("bMax").agg(
        nncv=("nn_cv_u", "mean"), covE=("cov_err_fro", "mean"),
        iters=("iterations", "mean"), ms=("elapsedMicro", "mean"))
    g["ms"] /= 1000.0
    g["rel"] = g.covE / 2.0     # relative to tr(target) = 2

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))

    ax = axes[0]
    ax.plot(g.index, g.rel, color=BLUE, marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhspan(0.015, 0.018, color=GRID, zorder=0)
    ax.text(1.3, 0.0155, "plateau = intrinsic ≈1/L\nshrinkage, not a bMax "
            "error", fontsize=8.5, color=INK2, va="center")
    ax.set_title("Covariance error (relative)\nsaturates — flat from ≈50σ "
                 "to 1000σ", fontsize=10)
    ax.set_ylabel("‖C − I‖$_F$ / tr(I)")

    ax = axes[1]
    ax.plot(g.index, g.nncv, color=BLUE, marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.6)
    ax.set_title("Non-uniformity NN-CV(u)\ncatastrophic only below ≈5σ",
                 fontsize=10)

    ax = axes[2]
    ax.plot(g.index, g.iters, color=BLUE, marker="o", ms=4,
            label="BFGS iterations")
    ax.plot(g.index, 100 * g.ms / g.ms.max(), color=INK2, ls=":", marker="o",
            ms=3, label="runtime (% of max)")
    ax.set_xscale("log")
    ax.set_title("Optimizer effort\nfalls beyond ≈150σ: quadrature noise "
                 "(∝ bMax²·εrel)\ntriggers early no-progress stops",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    for ax in axes:
        ax.axvspan(50, 100, color=BAND, zorder=0)
        ax.set_xlabel("bMax   (σ = 1, log scale)")
        style(ax)
    axes[0].annotate("recommended\n50–100σ", xy=(70, 0.25), fontsize=8.5,
                     color=INK2, ha="center",
                     xycoords=("data", "axes fraction"))

    fig.suptitle("Measured sample quality vs bMax   (2D standard normal, "
                 "L = 40, strict stop, mean over 4 seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(outdir, "fig_bmax_quality.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 4. Scale invariance: only bMax/σ matters
# ---------------------------------------------------------------------------

def fig_scale_invariance(df, outdir):
    s = df[df.experiment == "sigma_collapse"].copy()
    s["sig"] = s["sigma"].str.split(";").str[0].astype(float)
    s["rel"] = s.cov_err_fro / (2.0 * s.sig ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True)

    ax = axes[0]
    for colr, (sig, d) in zip(SIGMA_COLORS, s.groupby("sig")):
        g = d.groupby("bMax").rel.mean()
        ax.plot(g.index, g.values, color=colr, marker="o", ms=4,
                label=f"σ = {sig:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("raw bMax  (log scale)")
    ax.set_ylabel("relative covariance error")
    ax.set_title("Same sweeps plotted against raw bMax:\nevery σ needs a "
                 "different bMax", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    style(ax)

    ax = axes[1]
    for colr, (sig, d) in zip(SIGMA_COLORS, s.groupby("sig")):
        d = d.assign(ratio=d.bMax / d.sig)
        g = d.groupby("ratio").rel.mean()
        ax.plot(g.index, g.values, color=colr, marker="o", ms=4,
                label=f"σ = {sig:g}")
    ax.set_xscale("log")
    ax.set_xlabel("bMax / σ  (log scale)")
    ax.set_title("Plotted against bMax/σ: all six curves collapse\n"
                 "onto one — only the RATIO matters", fontsize=10)
    style(ax)

    fig.suptitle("Scale invariance: quality depends on bMax/σ only   "
                 "(L = 20, strict stop, σ from 0.1 to 10, 2 seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p = os.path.join(outdir, "fig_bmax_scale_invariance.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 5. The 1/bMax² convergence law and the resulting bMax(L) rule
# ---------------------------------------------------------------------------

def knee(covE_by_B, factor=1.10, plateau_range=(100, 300)):
    """Smallest measured bMax whose cov error is within `factor` of plateau."""
    plate = covE_by_B[(covE_by_B.index >= plateau_range[0])
                      & (covE_by_B.index <= plateau_range[1])].mean()
    ok = covE_by_B[covE_by_B <= factor * plate]
    return float(ok.index.min()), float(plate)


def fig_convergence_law(df, outdir):
    f = df[df.experiment == "bmax_fine"]
    g = f.groupby("bMax").cov_trace_ratio.mean()
    tr_inf = g[(g.index >= 150) & (g.index <= 1000)].mean()
    d = (tr_inf - g[g.index <= 100]).clip(lower=0)
    d = d[d > 0]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    ax.plot(d.index, d.values, color=BLUE, marker="o", ms=5, ls="none",
            label="measured  tr(C)∞ − tr(C)(bMax)")
    bb = np.logspace(0, 2.1, 50)
    ax.plot(bb, 6.0 / bb ** 2, color=INK2, ls="--", lw=1.5,
            label="6·(σ/bMax)²  (fit, slope −2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bMax  (σ = 1)")
    ax.set_ylabel("covariance-trace deficit due to finite bMax")
    ax.set_title("The bMax-induced bias vanishes as 1/bMax²\n"
                 "(L = 40, mean over 4 seeds)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    style(ax)

    # knee(L): where covariance error is within 10% of its plateau
    ax = axes[1]
    knees = {}
    v = df[df.experiment == "bmax_vs_L"]
    for L, dL in v.groupby("L"):
        knees[int(L)] = knee(dL.groupby("bMax").cov_err_fro.mean())[0]
    knees[40] = knee(f.groupby("bMax").cov_err_fro.mean())[0]
    Ls = sorted(knees)
    ax.plot(Ls, [knees[L] for L in Ls], color=BLUE, marker="o", ms=7,
            ls="none", label="measured knee (cov error within\n10% of its "
            "plateau; grid resolution ≈ ×1.5)")
    lg = np.linspace(10, 260, 100)
    ax.fill_between(lg, 10 * np.sqrt(lg), 15 * np.sqrt(lg), color=BAND,
                    zorder=0, label="bMax = (10–15)·σ·√L")
    ax.plot(lg, 10 * np.sqrt(lg), color=INK2, ls="--", lw=1.2)
    ax.plot(lg, 15 * np.sqrt(lg), color=INK2, ls="--", lw=1.2)
    ax.set_xlabel("number of samples  L")
    ax.set_ylabel("bMax needed to saturate quality  (units of σ)")
    ax.set_title("Larger L resolves finer structure and needs\nlarger bMax — "
                 "consistent with (10–15)·σ·√L", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)

    fig.suptitle("From measurement to rule: residual bias ≈ 6σ²/bMax² must "
                 "drop below the intrinsic ≈1/L deficit  ⇒  bMax ≳ σ·√(60·L)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p = os.path.join(outdir, "fig_bmax_convergence_law.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 6. Failure gallery: what the samples look like
# ---------------------------------------------------------------------------

def _density(ax, lim):
    res = 300
    axis = np.linspace(-lim, lim, res)
    xx, yy = np.meshgrid(axis, axis)
    dens = np.exp(-0.5 * (xx ** 2 + yy ** 2)) / (2 * np.pi)
    ax.imshow(dens, origin="lower", extent=[-lim, lim, -lim, lim],
              vmin=0.0, aspect="equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_failure_gallery(df, outdir, results_dir):
    rows = [
        ("bmax_fine", 40, [1, 3, 100],
         "L = 40:  bMax too small collapses the samples"),
        ("bmax_vs_L", 200, [30, 100, 500],
         "L = 200:  very large bMax degrades late-stage optimization "
         "(quadrature noise)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.2))
    for r_i, (exp, L, bvals, rowtitle) in enumerate(rows):
        for c_i, b in enumerate(bvals):
            r = df[(df.experiment == exp) & (df.L == L) & (df.bMax == b)
                   & (df.seed == 42)].iloc[0]
            x = load_samples(r, results_dir)
            ax = axes[r_i][c_i]
            _density(ax, 3.6)
            ax.scatter(x[:, 0], x[:, 1], s=26 if L == 40 else 13,
                       c="#e8000b", edgecolors="white",
                       linewidths=0.5, zorder=3)
            ax.set_title(
                f"bMax = {b}σ\ncov. captured: "
                f"{100 * float(r.cov_trace_ratio):.0f}%   "
                f"NN-CV(u): {float(r.nn_cv_u):.2f}", fontsize=10)
        axes[r_i][1].text(0.5, 1.22, rowtitle, transform=axes[r_i][1].transAxes,
                          ha="center", fontsize=11.5)
    fig.suptitle("Failure modes at both ends of the bMax range   "
                 "(seed 42, strict stop)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(outdir, "fig_bmax_failure_gallery.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.results, "results.csv"))

    fig_role(df, args.outdir, args.results)
    fig_information(df, args.outdir, args.results)
    fig_quality(df, args.outdir)
    fig_scale_invariance(df, args.outdir)
    fig_convergence_law(df, args.outdir)
    fig_failure_gallery(df, args.outdir, args.results)


if __name__ == "__main__":
    main()
