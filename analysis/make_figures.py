"""Generate the meeting figure set from the sweep results.

Homogeneity is shown via the nearest-neighbour distance CV of the uniformized
samples (NN-CV(u)), accuracy via the covariance error, and optimizer effort
via the iteration count.

Usage:
    python analysis/make_figures.py --results <dir-with-results.csv> \
        --outdir <figure-dir>

Figures written:
    fig_quality_vs_L.png          iterations / NN-CV(u) / cov-error vs L
    fig_quality_vs_bmax.png       NN-CV(u) / cov-error / iterations vs bMax
    fig_scatter_before_after.png  x-space scatters, old default vs fixed stop
    fig_nncv_explainer.png        what NN-CV measures (self-contained)
    fig_gaussian_to_uniform.png   uniformization view + the fix (L=100)
    fig_optimizer_stops_early.png gradient-norm trajectory + stop points
    fig_objective_offset.png      why a relative f-tolerance fails here
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import special

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402  (only NN / uniformize / moment helpers are used)

# palette: RED = old default stop (ftolRel=1e-10), BLUE = fixed/strict stop.
BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e8e6"
DOT = "#26324a"       # neutral point colour for explainer scatters
SEG = "#8fa3bf"       # nearest-neighbour segment colour

LBL_OLD = "old default stop (ftolRel=1e-10)"
LBL_FIX = "fixed stop (ftolRel=0)"

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
    """Resolve a run's sample CSV, preferring <results_dir>/samples/<name>."""
    name = os.path.basename(str(row["samples_file"]).replace("\\", "/"))
    cand = os.path.join(results_dir, "samples", name)
    path = cand if os.path.exists(cand) else str(row["samples_file"])
    return np.loadtxt(path, delimiter=",")


def nn_cv(points):
    """NN-distance coefficient of variation of a point set."""
    return metrics.nn_stats(np.asarray(points))["nn_cv"]


# ---------------------------------------------------------------------------
# TASK 1 — quality vs L  (iterations, NN-CV(u), covariance error)
# ---------------------------------------------------------------------------

def fig_quality_vs_L(df, outdir, results_dir):
    base = df[df.experiment == "baseline_L"].groupby("L").mean(numeric_only=True)
    strict = df[df.experiment == "strict_L"].groupby("L").mean(numeric_only=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    panels = [
        ("iterations", "Optimizer effort\n(BFGS iterations until stop)", False),
        ("nn_cv_u", "Non-uniformity\n(NN-distance CV, lower = more even)", False),
        ("cov_err_fro", "Covariance error\n‖C − I‖$_F$ (lower = better)", True),
    ]
    for ax, (col, title, logy) in zip(axes, panels):
        ax.plot(base.index, base[col], color=RED, marker="o", ms=4, label=LBL_OLD)
        ax.plot(strict.index, strict[col], color=BLUE, marker="o", ms=4,
                label=LBL_FIX)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("number of samples  L")
        ax.set_title(title, fontsize=10)
        style(ax)
    axes[0].annotate("default optimizes\nless as L grows",
                     xy=(120, base.loc[100, "iterations"]),
                     xytext=(14, 90), fontsize=8.5, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    axes[0].legend(frameon=False, fontsize=8.5, loc="center right")
    fig.suptitle("Sample quality vs number of samples L   "
                 "(2D standard normal, bMax = 100, mean over 4 seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(outdir, "fig_quality_vs_L.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# TASK 2 — quality vs bMax  (NN-CV(u), covariance error, iterations)
# ---------------------------------------------------------------------------

def fig_quality_vs_bmax(df, outdir, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    panels = [
        ("nn_cv_u", "Non-uniformity\n(NN-distance CV, lower = more even)", False),
        ("cov_err_fro", "Covariance error\n‖C − I‖$_F$ (lower = better)", True),
        ("iterations", "Optimizer effort\n(BFGS iterations until stop)", False),
    ]
    for ax, (col, title, logy) in zip(axes, panels):
        for exp, colr, lbl in (("bmax_default", RED, LBL_OLD),
                               ("bmax_strict", BLUE, LBL_FIX)):
            d = df[(df.experiment == exp) & (df.L == 40)]
            g = d.groupby("bMax")[col].mean()
            ax.plot(g.index, g.values, color=colr, marker="o", ms=4, label=lbl)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("bMax   (σ = 1)")
        ax.set_title(title, fontsize=10)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Sample quality vs bMax   "
                 "(2D standard normal, L = 40, mean over 4 seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(outdir, "fig_quality_vs_bmax.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# before/after scatter
# ---------------------------------------------------------------------------

def _density(ax, faint=False):
    lim, res = 4.0, 400
    axis = np.linspace(-lim, lim, res)
    xx, yy = np.meshgrid(axis, axis)
    dens = np.exp(-0.5 * (xx ** 2 + yy ** 2)) / (2 * np.pi)
    if faint:
        ax.imshow(dens, origin="lower", extent=[-lim, lim, -lim, lim],
                  vmin=0.0, cmap="Greys", alpha=0.55, aspect="equal")
    else:
        ax.imshow(dens, origin="lower", extent=[-lim, lim, -lim, lim],
                  vmin=0.0, aspect="equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(range(-4, 5, 2))
    ax.set_yticks(range(-4, 5, 2))
    ax.grid(False)


def fig_before_after(df, outdir, results_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.6))
    for col, L in enumerate((10, 20, 40)):
        for row, (exp, label) in enumerate((
                ("baseline_L", "old default stop"),
                ("strict_L", "fixed stop (ftolRel=0)"))):
            r = df[(df.experiment == exp) & (df.L == L) & (df.seed == 42)]
            x = load_samples(r.iloc[0], results_dir)
            ax = axes[row][col]
            _density(ax)
            ax.scatter(x[:, 0], x[:, 1], s=30, c="#e8000b",
                       edgecolors="white", linewidths=0.6, zorder=3)
            ax.set_title(
                f"L = {L}, {label}\nNN-CV(u) = {float(r.nn_cv_u.iloc[0]):.2f}, "
                f"iterations = {int(r.iterations.iloc[0])}", fontsize=10)
    fig.suptitle("Standard normal, seed 42, bMax = 100:  "
                 "old default stop (top) vs fixed stop (bottom)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, "fig_scatter_before_after.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# TASK 5 — NN-CV explainer (self-contained synthetic point sets)
# ---------------------------------------------------------------------------

def _nn_index_dist(pts):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    j = d.argmin(axis=1)
    return j, d[np.arange(len(pts)), j]


def _make_regular(rng, n_side=8):
    g = (np.arange(n_side) + 0.5) / n_side
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pts += rng.uniform(-0.16, 0.16, size=pts.shape) / n_side
    return np.clip(pts, 0.01, 0.99)


def _make_random(rng, n=64):
    return rng.uniform(0.0, 1.0, size=(n, 2))


def _make_clumpy(rng, n=64):
    centers = rng.uniform(0.15, 0.85, size=(6, 2))
    n_clustered = n - 8
    idx = rng.integers(0, len(centers), size=n_clustered)
    clustered = centers[idx] + rng.normal(0, 0.028, size=(n_clustered, 2))
    stragglers = rng.uniform(0.03, 0.97, size=(8, 2))
    return np.clip(np.vstack([clustered, stragglers]), 0.0, 1.0)


def fig_nncv_explainer(outdir):
    rng = np.random.default_rng(7)
    sets = [
        ("regular", _make_regular(rng), "low NN-CV\n(even coverage)"),
        ("random", _make_random(rng), "medium NN-CV\n(iid uniform)"),
        ("clumpy", _make_clumpy(rng), "high NN-CV\n(clusters + gaps)"),
    ]
    # shared histogram x-range for comparability
    all_nn = [_nn_index_dist(p)[1] for _, p, _ in sets]
    hi = max(d.max() for d in all_nn)
    bins = np.linspace(0, hi, 22)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2),
                             gridspec_kw={"height_ratios": [2.4, 1]})
    for c, (name, pts, note) in enumerate(sets):
        j, nn = _nn_index_dist(pts)
        cv = float(nn.std() / nn.mean())
        top = axes[0][c]
        for i in range(len(pts)):
            top.plot([pts[i, 0], pts[j[i], 0]], [pts[i, 1], pts[j[i], 1]],
                     color=SEG, lw=1.0, zorder=1)
        top.scatter(pts[:, 0], pts[:, 1], s=26, color=DOT, zorder=2)
        top.set_xlim(-0.03, 1.03)
        top.set_ylim(-0.03, 1.03)
        top.set_aspect("equal")
        top.set_xticks([])
        top.set_yticks([])
        top.grid(False)
        top.set_title(f"{name}   —   NN-CV = {cv:.2f}", fontsize=11)
        top.text(0.035, 0.965, note, transform=top.transAxes, ha="left",
                 va="top", fontsize=8.5, color=INK2,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))

        bot = axes[1][c]
        bot.hist(nn, bins=bins, color=BLUE, alpha=0.85, edgecolor="white")
        bot.axvline(nn.mean(), color=INK, lw=1.4, ls="--")
        bot.set_xlim(0, hi)
        bot.set_xlabel("nearest-neighbour distance")
        if c == 0:
            bot.set_ylabel("count")
        style(bot)

    fig.suptitle("NN-CV = spread (coefficient of variation) of nearest-neighbour "
                 "distances:  0 = perfectly even, larger = clumpier",
                 fontsize=12)
    fig.text(0.5, 0.035,
             "Top row: each point is joined to its nearest neighbour.   "
             "Bottom row: histogram of those distances (dashed line = mean).",
             ha="center", fontsize=9, color=INK2)
    fig.text(0.5, 0.008,
             "Even spacing → tight histogram → low CV.     "
             "Clumpiness → wide, bimodal histogram → high CV.",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    p = os.path.join(outdir, "fig_nncv_explainer.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# TASK 6 — Gaussian samples -> uniform, old vs fixed  (ties metric to pictures)
# ---------------------------------------------------------------------------

def fig_gaussian_to_uniform(df, outdir, results_dir):
    def has(exp, L):
        return not df[(df.experiment == exp) & (df.L == L)
                      & (df.seed == 42)].empty
    L = 100 if has("baseline_L", 100) and has("strict_L", 100) else 40
    rows = [("baseline_L", "old default stop"),
            ("strict_L", "fixed stop (ftolRel=0)")]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.2))
    for r_i, (exp, label) in enumerate(rows):
        row = df[(df.experiment == exp) & (df.L == L) & (df.seed == 42)].iloc[0]
        x = load_samples(row, results_dir)
        u = special.ndtr(x)                     # sigma = 1 -> u = Phi(x)
        cv = nn_cv(u)

        axL = axes[r_i][0]
        _density(axL, faint=True)
        axL.scatter(x[:, 0], x[:, 1], s=26, c="#e8000b", edgecolors="white",
                    linewidths=0.5, zorder=3)
        axL.set_title(f"{label}\nsamples in x-space (over N(0, I))", fontsize=10)

        axR = axes[r_i][1]
        axR.scatter(u[:, 0], u[:, 1], s=26, c=BLUE, edgecolors="white",
                    linewidths=0.5, zorder=3)
        axR.set_xlim(0, 1)
        axR.set_ylim(0, 1)
        axR.set_aspect("equal")
        axR.set_xticks([0, 0.5, 1])
        axR.set_yticks([0, 0.5, 1])
        axR.set_xlabel("Φ(x₁)")
        axR.set_ylabel("Φ(x₂)")
        axR.grid(True)
        axR.set_title(f"uniformized  u = Φ(x)\nNN-CV(u) = {cv:.2f}", fontsize=10)
        style(axR)

    fig.suptitle("Uniformizing  u = Φ(x)  removes the intended centre-dense "
                 "gradient:\na good sample set then tiles the unit square "
                 f"evenly   (L = {L}, seed 42)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(outdir, "fig_gaussian_to_uniform.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# TASK 7 — optimizer stops early  (gradient-norm trajectory)
# ---------------------------------------------------------------------------

def fig_optimizer_stops_early(outdir, results_dir, d_it=32, gtol=1e-6):
    trace_path = os.path.join(results_dir, "trace_gradnorm.csv")
    if not os.path.exists(trace_path):
        print("  (skip fig_optimizer_stops_early: no trace_gradnorm.csv; run "
              "analysis/trace_gradnorm.py)")
        return
    tr = pd.read_csv(trace_path).sort_values("iter")
    it, g = tr["iter"].values, tr["gradNorm"].values
    f_it, f_g = int(it[-1]), float(g[-1])
    # place the default-stop marker exactly on the reconstructed trajectory
    d_g = float(g[np.argmin(np.abs(it - d_it))])

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 1.6e-2)

    # full trajectory = the path the fixed run traverses
    ax.plot(it, g, color=BLUE, lw=2.0, zorder=2,
            label=f"fixed stop: runs to {f_it} iters, reaches |∇f| ≈ gtol")
    # portion the old default run actually reaches
    mask = it <= d_it
    ax.plot(it[mask], g[mask], color=RED, lw=3.2, zorder=3,
            label=f"old default stop: halts at {d_it} iters")

    ax.axhline(gtol, color=INK, ls="--", lw=1.4, zorder=1)
    ax.text(195, gtol * 1.04, "gtol = 1e-6   (true convergence target)",
            ha="center", va="bottom", fontsize=9, color=INK)

    ax.scatter([d_it], [d_g], s=95, color=RED, zorder=4, edgecolor="white")
    ax.annotate(f"default stops at {d_it} iters,\n|∇f| ≈ {d_g:.0e}  "
                f"(≈{d_g / gtol:.0f}× above target)",
                xy=(d_it, d_g), xytext=(d_it + 34, d_g * 7),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.scatter([f_it], [f_g], s=95, color=BLUE, zorder=4, edgecolor="white")
    ax.annotate(f"fixed run reaches target\nat {f_it} iters",
                xy=(f_it, f_g), xytext=(f_it - 120, 2.3e-5),
                fontsize=9.5, color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

    ax.set_xlabel("BFGS iteration")
    ax.set_ylabel("gradient norm  |∇f|")
    ax.set_title("The default stop halts while the gradient is still ~300× "
                 "above target\n(2D standard normal, L = 40, seed 42)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    style(ax)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_optimizer_stops_early.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# TASK 8 — objective offset  (why a relative f-tolerance fails)
# ---------------------------------------------------------------------------

def fig_objective_offset(outdir):
    # measured at bMax=100, L=40, seed 42 (see analysis/LOG.md):
    offset = 4995.39           # |f| ~ 1/2 * bMax^2, independent of positions
    signal = 5.9e-3            # f(iter 1) - f(optimum): the position-dependent part
    threshold = 1e-10 * offset  # ftolRel * |f| ~ 5e-7: the absolute stop threshold

    labels = ["|f|  (objective magnitude)\n≈ ½·bMax²  — constant offset",
              "position-dependent signal\n(what actually gets optimized)",
              "ftolRel · |f|  stop threshold\n(1e-10 × ≈5000)"]
    vals = [offset, signal, threshold]
    colors = [INK2, BLUE, RED]

    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=colors, height=0.55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 3e4)
    ax.set_xlabel("magnitude (log scale)")
    for yi, v in zip(y, vals):
        ax.text(v * 1.6, yi, f"{v:.3g}", va="center", fontsize=10)
    ax.set_title("Why a relative f-tolerance fails here", fontsize=12)
    style(ax)
    ax.grid(True, axis="x")
    fig.text(0.5, -0.02,
             "The objective is dominated by a constant ≈ −½·bMax² ≈ −5000; the "
             "part that depends on where the points sit is only ~10⁻³.\n"
             "A relative tolerance measures progress against that ≈5000, so it "
             "calls the run 'done' once a step changes f by ≈5×10⁻⁷ — long "
             "before the configuration is actually optimized.",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = os.path.join(outdir, "fig_objective_offset.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.results, "results.csv"))

    fig_quality_vs_L(df, args.outdir, args.results)
    fig_quality_vs_bmax(df, args.outdir, args.results)
    fig_before_after(df, args.outdir, args.results)
    fig_nncv_explainer(args.outdir)
    fig_gaussian_to_uniform(df, args.outdir, args.results)
    fig_optimizer_stops_early(args.outdir, args.results)
    fig_objective_offset(args.outdir)


if __name__ == "__main__":
    main()
