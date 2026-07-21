# How to choose bMax — the full evidence behind the selection rule

Companion to `REPORT.md` Q3, with dedicated experiments and figures.
Everything here uses the **fixed stopping rule** (`ftolRel = 0`, the library
default on this branch), so bMax effects are measured cleanly, not through
the old premature-stop bug. All sweeps: 2D standard normal unless stated;
data from `run_sweeps.py --only bmax_fine,sigma_collapse,bmax_vs_L`
(228 runs, all converged); figures from `make_bmax_figures.py`.

**The rule**

> **bMax ≈ 10–15 · σ_max · √L, rounded up — in practice any integer in
> [50 σ_max, 100 σ_max] is fine for L ≲ 150.**
> Below ~5 σ the approximation collapses (failure mode 1); beyond ~300 σ
> quality slowly degrades through quadrature noise (failure mode 2), and
> from bMax/σ ≈ 10⁴ the library can abort the process outright.
> Quality is *flat* across the whole recommended window, so the choice is
> forgiving — getting the order of magnitude right is what matters.

---

## 1. What bMax actually is

The library minimizes a distance between the target Gaussian and the sample
set that works by **smoothing both with a Gaussian kernel of width b and
comparing the smoothed versions, for every b from 0 to bMax** (that is the
integral `calculateD2` evaluates; `calculateD3` is the closed form of the
sample–sample part). The kernel width b is a *magnifying-glass scale*:

- small b sees individual samples (that is what forces them apart evenly),
- b of a few σ sees the overall bell shape (that is what fixes the spread),
- b ≫ σ sees both distributions as the same featureless blob.

`figures/fig_bmax_role.png` shows this directly with the smoothed profiles
of the target vs a converged 40-sample set: at b = 0.3σ they differ by 18%
of peak, at b = 2σ by 0.11%, at b = 20σ by 0.006%. **Scales beyond a few σ
contribute almost nothing — bMax only needs to be "comfortably past the
informative scales".** The rest of this note quantifies "comfortably".

## 2. Failure mode 1: bMax too small prefers collapsed sets

Using an exact Python replica of the code's objective (`lcd_distance.py`,
matches the C++ `dist` output to relative 3·10⁻¹⁵ at bMax = 3, 50, 500 —
`python analysis/lcd_distance.py --samples <csv> --bMax 50 --dist <value>`),
I evaluated D(wrong set) − D(optimal set) as a function of the cutoff for
three controlled corruptions (`figures/fig_bmax_information.png`, left):

- a set **shrunk ×0.8**: for bMax ≲ 4σ the difference is *negative* — the
  objective genuinely prefers the collapsed set. The optimizer is not
  failing; it is faithfully minimizing a distance that cannot see the
  missing tails, because it never looks at scales that large.
- measured consequence (`figures/fig_bmax_failure_gallery.png`, top row,
  and the bmax_fine sweep): at bMax = 1σ the L=40 samples collapse to a
  ring capturing 28% of the target covariance (NN-CV(u) 0.54); at 3σ, 69%;
  at 10σ, 92%; at 50σ, 97.4% ≈ the plateau.

## 3. Why growing bMax stops helping: the 1/bMax² law

Two independent measurements give the same law:

1. **Objective side** (`fig_bmax_information.png`, right): for mean-free
   perturbations (inflation, clumping) the share of the discriminating
   penalty still missing at cutoff B falls as **1/B²** over two decades
   (the b-integrand's x-dependent part decays ~1/b³ once the log terms of
   D2 and D3 cancel). At bMax = 50σ ≈ 3.6% of the signal is missing; at
   100σ ≈ 0.8%.
   (Aside: a *mean shift* is the one perturbation whose penalty grows like
   ln bMax without saturating — the cancellation needs zero mean. That is
   presumably why the library runs `correctMean` after optimization.)
2. **Sample side** (`fig_bmax_convergence_law.png`, left): the converged
   sets' covariance-trace deficit relative to the bMax→∞ plateau follows
   **≈ 6·(σ/bMax)²** — the fitted constant is 5.6–6.7 across bMax = 10 to
   100 (L = 40, 4 seeds). A finite bMax biases the optimum, and the bias
   dies quadratically.

So there is no "optimal" bMax to hunt for — there is a *saturation
threshold*, above which everything is equally good until noise takes over.

## 4. Failure mode 2: bMax too large buys noise, then crashes

The D2 integral is computed by adaptive quadrature with relative tolerance
1e-10 (`gsl_quadrature_adaptive_gauss_kronrod.h`). The integral's *value*
grows like bMax²/2 (the constant offset described in REPORT.md), so the
**absolute noise in every objective evaluation grows as ≈ 1e-10 · bMax²**:
~10⁻⁶ at bMax = 100, ~2.5·10⁻⁵ at 500. Late-stage BFGS steps improve f by
less than that, so the line search hits "no progress" earlier and earlier:

- iterations until convergence at L = 40 (4-seed mean, bmax_fine sweep):
  423 at bMax = 100 → 283 at 200 → 221 at 500 → 145 at 1000.
- at L = 40 the lost iterations are cosmetic (quality metrics stay flat to
  bMax = 1000), but at **L = 200 — where late fine-tuning matters more —
  quality visibly degrades**: NN-CV(u) 0.145 at bMax = 100 → 0.198 at 500;
  covariance error 0.0062 at 300 → 0.0077 at 500
  (`fig_bmax_failure_gallery.png`, bottom row).
- **hard failure**: from bMax/σ ≈ 10⁴ the adaptive quadrature can fail
  outright, and with GSL's default error handler that **aborts the calling
  process**. It is configuration-dependent, i.e. a reliability lottery:
  σ = 0.1, bMax = 1000 aborted with seeds 2 and 3 but not seed 1 or 42
  (`GSL_RNG_SEED=2 build/plots/generate_samples.exe 2 20 out.csv 0.1 0.1
  1000` → crash); σ = 0.01, bMax = 1000 aborted with seed 42.

## 5. Only bMax/σ matters (so the rule is in units of σ)

`figures/fig_bmax_scale_invariance.png`: six sweeps with σ from 0.1 to 10,
plotted against raw bMax, look like six different problems. Plotted against
**bMax/σ they collapse onto a single curve** — the relative covariance
error agrees to the 4th decimal at every ratio (e.g. 6.45% at bMax/σ = 10
and 3.27% at 50, for *all* σ). This is expected from the math (b only ever
appears alongside σ and point coordinates) and is now verified to
measurement precision. Anisotropic targets are handled upstream by sampling
per-axis σ = √eigenvalues and rotating, so **σ_max, the largest per-axis
standard deviation, is the scale that must satisfy the rule** (with very
anisotropic targets the smaller axes then sit deep in their comfortable
range — harmless, since the flat window is wide).

## 6. Where the numbers in the rule come from

Putting §3 and §4 together:

- The bMax-induced covariance bias is ≈ 6·(σ/bMax)². The *intrinsic*
  covariance deficit of an optimal L-sample LCD set is ≈ 1/L (trace ratio
  0.96 at L=40, 0.99 at L=200 — a property of the optimum, not an error).
  Requiring the bMax bias to be ≤ 10% of the intrinsic deficit gives
  **bMax ≥ σ·√(60·L) ≈ 8·σ·√L**, i.e. 50σ at L=40, 110σ at L=200.
- Measured saturation knees (bMax where covariance error is within 10% of
  its plateau) match: 50σ at L=20 and 40, 100σ at L=100, 200σ at L=200 —
  consistent with **(10–15)·σ·√L** given the sweep's grid resolution
  (`fig_bmax_convergence_law.png`, right).
- The noise ceiling (§4) sits around 300–500σ for L ≤ 200. The window
  [10√L·σ, ~300σ] is wide open for every L we measured; it narrows as L
  grows, which is an open point beyond L ≈ 500 (see §8).
- Cost is *not* a reason to keep bMax small: runtime per iteration grows
  only mildly (0.5 ms → 1.1 ms from bMax = 10 to 100 at L = 40; adaptive
  quadrature is roughly logarithmic in bMax), and total runtime is
  dominated by the iteration count.

Practical summary (2D, verified for L = 20…200, σ = 0.1…10):

| bMax/σ | what happens |
|---|---|
| ≲ 4 | objective *prefers* collapsed sets — catastrophic |
| 5–30 | works, but measurable covariance bias (6σ²/bMax²) |
| **≈ 10–15·√L (≈ 50–100 for typical L)** | **saturated quality — recommended** |
| ~300–1000 | flat at small L; at L ≳ 200 late-stage quality erodes (quadrature noise) |
| ≳ 10⁴ | seed-dependent hard abort of the process |

The current default `bMax = 100` is right for σ = 1 and L ≲ 150. A σ-aware
default such as `ceil(100·σ_max)` (or the √L-aware version) remains the
recommended library change; the field is a `size_t`, so sub-integer bMax
for tiny σ is impossible — for σ = 0.01 the *smallest representable* bMax
is already 100σ, which luckily lands in the good window, but σ ≲ 0.001
would force bMax/σ ≥ 10⁵ into the crash zone. That is a genuine API
limitation worth raising with the maintainers.

## 7. Solid vs uncertain

Solid (each independently measured, and the mechanism reproduces in the
exact Python replica of the objective):
- scale invariance (collapse to 4 decimals across two orders of σ);
- the sign flip at bMax ≈ 4σ explaining the collapse failure mode;
- the 1/bMax² information tail and the 6·(σ/bMax)² bias law (slope −2 over
  two decades);
- the noise mechanism at large bMax (εrel = 1e-10 quadrature on a value
  growing as bMax²), its measured iteration collapse, and the quality
  damage at L = 200;
- the seed-dependent abort from bMax/σ ≈ 10⁴.

Uncertain:
- the constant "6" in the bias law and the knee positions are 2D
  measurements; the 1/bMax² *form* is dimension-independent, but the
  constants were not re-measured for N > 2;
- the upper (noise) ceiling was mapped coarsely; for L ≳ 500 the window
  between 10√L·σ and the noise ceiling may narrow enough to require
  tightening the quadrature tolerance instead of just picking bMax;
- knee estimates are quantized by the sweep grid (roughly a factor 1.5).

## 8. Reproduce

```
cmake --build build --target lcd_experiment generate_samples -j
python analysis/run_sweeps.py --outdir <results-dir> \
    --only bmax_fine,sigma_collapse,bmax_vs_L          # ~4 min, 228 runs
python analysis/make_bmax_figures.py --results <results-dir> \
    --outdir analysis/figures
# objective-replica self-check against any driver run:
build/plots/lcd_experiment.exe --L 40 --seed 42 --bMax 50 --out s.csv
python analysis/lcd_distance.py --samples s.csv --bMax 50 --dist <RESULT dist=...>
# crash reproduction (WARNING: aborts the process, by design of GSL's
# default error handler):
GSL_RNG_SEED=2 build/plots/generate_samples.exe 2 20 out.csv 0.1 0.1 1000
```
