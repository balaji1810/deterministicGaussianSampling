# How to choose bMax — the full evidence behind the selection rule

Companion to `REPORT.md` Q3, with dedicated experiments and figures.
Everything here uses the **fixed stopping rule** (`ftolRel = 0`, the library
default on this branch), so bMax effects are measured cleanly, not through
the old premature-stop bug. All sweeps: 2D standard normal unless stated;
data from `run_sweeps.py --only bmax_fine,sigma_collapse,bmax_vs_L`
(228 runs, all converged); figures from `make_bmax_figures.py`.

**The rule**

> **Use bMax ≳ 10–15 · σ_max · √L (any integer in [50 σ_max, 100 σ_max] is
> a good default for L ≲ 150).** There is a genuine *lower* bound — below
> ~5 σ the approximation collapses — but **no upper bound**: quality is
> flat or slowly improving all the way to the largest bMax tested
> (bMax/σ = 10⁵). Larger bMax only costs runtime.

> **Update — the upper failure mode has been removed, not worked around.**
> An earlier revision of this note recommended an upper limit of ~300 σ
> because quality degraded beyond it and the library could abort at
> bMax/σ ≈ 10⁴. That degradation was **not** mathematical: it came from an
> x-independent, ≈ bMax²/2-sized constant sitting inside the value handed
> to the optimizer, which made the quadrature's *absolute* noise grow as
> bMax². §7 derives and removes that constant; §8 measures the result. The
> upper limit and the crash are both gone, and §4 is kept only as the
> record of the original diagnosis.

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

## 4. Failure mode 2 (now FIXED — see §7): bMax too large bought noise

*This section records the original diagnosis, measured on the pre-fix
library. Everything in it was caused by the removable constant of §7; after
the fix none of these effects remain (§8). It is kept because the mechanism
is what motivated the fix.*

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
- There is **no upper ceiling any more** (§7–§8): after the offset fix,
  quality is flat from the knee out to bMax/σ = 10⁵, the largest tested.
- Cost is the only remaining reason not to set bMax huge: runtime grows
  roughly logarithmically with bMax (at L = 40, ~9 ms at bMax = 10 →
  ~100 ms at 1000, dominated by the iteration count, which is itself flat).

Practical summary (2D, verified for L = 20…200, σ = 0.1…10, post-fix):

| bMax/σ | what happens |
|---|---|
| ≲ 4 | objective *prefers* collapsed sets — catastrophic |
| 5–30 | works, but measurable covariance bias (6σ²/bMax²) |
| **≈ 10–15·√L (≈ 50–100 for typical L)** | **saturated quality — recommended default** |
| 100 – 10⁵ | still saturated; costs runtime, buys nothing (but harms nothing) |

The current default `bMax = 100` is right for σ = 1 and L ≲ 150. A σ-aware
default such as `ceil(100·σ_max)` remains the recommended library change.
Note the `size_t` type is no longer dangerous now that huge ratios are
safe, but it still prevents bMax < 1, so for σ ≳ 1 the *lower* bound is
what the caller must respect.

## 7. Removing the constraint instead of living with it

§4's ceiling was a numerical artifact, and it can be removed at the source.
Write the Gaussian–Dirac integrand as

    prefactor(b) · Σᵢ wᵢ exp(...)  =  prefactor(b)·Σᵢwᵢ  +  prefactor(b)·Σᵢwᵢ(exp(...) − 1)

The first piece does not contain the sample positions at all. Likewise every
pair in D3 contributes an identical ½·bMax²·wᵢwⱼ. Together they are exactly

    C(bMax) = −2·W·∫₀^bMax prefactor(b) db + ½·bMax²·W²,     W = Σᵢwᵢ

Verified numerically (`lcd_distance.py`): `f_full = f_reduced + C` to
machine precision (residual ≤ 1e-13 relative at bMax up to 10⁴), and
**f_reduced converges** as bMax grows — −0.345852 (bMax=50) → −0.346131
(1000) → −0.346132 (10⁴) — while `f_full` diverges as −bMax²/2 + ln bMax.
So *all* of the divergence is x-independent: the user-visible intuition
("larger bMax should never be worse") is mathematically correct, and only
the constant was breaking it.

Because the quadrature works to a **relative** tolerance (1e-10), carrying
C inside the optimizer's objective made the absolute noise per evaluation
≈ 1e-10·bMax²/2 — 5·10⁻⁷ at bMax = 100 but 5·10⁻³ at 10⁴ — against a signal
of size |f_reduced| ≈ 0.35 whose late-stage steps are ~1e-6. That is the
whole of failure mode 2, including the aborts (the quadrature simply could
not reach 1e-10 relative on a value that large).

The fix (on `dev`, ~15 lines, no API change):

1. `calculateP2` integrates `Σᵢwᵢ·expm1(...)` instead of `Σᵢwᵢ·exp(...)`.
   `expm1` is essential — the exponent → 0 as b grows, so `exp(...)−1`
   computed directly would lose every significant digit.
2. `calculateD3` omits the per-pair ½·bMax² constant.
3. A new `constantOffset()` adds C back **only in the public
   `modified_van_mises_distance_sq` reporting path**, so the reported
   distance is bit-for-bit comparable with previous versions.

The optimizer's objective differs from before by a constant only, so the
minimizer and the analytic gradient are mathematically unchanged.
Verified: reported distances agree with the pre-fix binary to ≤ 1.3e-11
relative at bMax = 10/100/1000, and the analytic gradient still matches
central finite differences to ~1e-7 (best over a step-size sweep; at
bMax = 1000 the *FD probe itself* is limited by the offset that remains in
the reported value — an independent illustration of the same disease).

## 8. Result: the upper limit is gone

`figures/fig_bmax_offset_fix.png`, before vs after (strict stop, 2 seeds):

| | L=40, bMax=100 | L=40, bMax=10⁴ | L=200, bMax=100 | L=200, bMax=10⁴ |
|---|---|---|---|---|
| iterations before | 396 | **5.5** | 183 | **2** |
| iterations after | 448 | **513** | 339 | **352** |
| NN-CV(u) before | 0.134 | **0.460** | 0.145 | **0.563** |
| NN-CV(u) after | 0.133 | **0.134** | 0.158 | **0.156** |
| cov. error before | 0.0337 | **0.306** | 0.0088 | **0.129** |
| cov. error after | 0.0336 | **0.0325** | 0.0089 | **0.0071** |

- Quality is now **flat across two extra decades** of bMax, at both L.
- At L = 200, larger bMax is now genuinely *better* (covariance error
  0.0088 → 0.0071), exactly as the 1/bMax² theory of §3 predicts and as it
  could never show before.
- Even at the *old* recommended bMax = 100 the fix helps: L = 200 runs 339
  iterations instead of 183, i.e. noise was already limiting there.
- **The aborts are gone.** All three previously-crashing configurations now
  run to completion with good samples: σ = 0.1, bMax = 1000, seeds 2 and 3;
  and σ = 0.01, bMax = 1000 (ratio 10⁵) → trace ratio 0.956 (intrinsic for
  L = 20 is ≈ 0.95), NN-CV(u) 0.081.
- Low-bMax behaviour is untouched (bMax = 10 gives covariance error 0.1126
  before and after), as it must be: §2's collapse is mathematical.

Side effect worth noting: with the offset gone, `f` is O(1), so a
*relative* f-tolerance is meaningful again. The `ftolRel = 0` default from
the earlier stopping fix is kept (it is what these numbers were measured
with), but `ftolRel = 1e-10` would now be a sane criterion rather than a
premature-stop bug.

## 9. Solid vs uncertain

Solid (each independently measured, and the mechanism reproduces in the
exact Python replica of the objective):
- scale invariance (collapse to 4 decimals across two orders of σ);
- the sign flip at bMax ≈ 4σ explaining the collapse failure mode;
- the 1/bMax² information tail and the 6·(σ/bMax)² bias law (slope −2 over
  two decades);
- the noise mechanism at large bMax (εrel = 1e-10 quadrature on a value
  growing as bMax²), its measured iteration collapse, and the quality
  damage at L = 200;
- the seed-dependent abort from bMax/σ ≈ 10⁴ (pre-fix);
- the exact decomposition `f_full = f_reduced + C(bMax)` and the
  convergence of `f_reduced` (machine precision, §7);
- that the fix restores flat quality to bMax/σ = 10⁵ and removes the
  aborts, with the reported distance and the gradient unchanged.

Uncertain:
- the constant "6" in the bias law and the knee positions are 2D
  measurements; the 1/bMax² *form* is dimension-independent, but the
  constants were not re-measured for N > 2;
- knee estimates are quantized by the sweep grid (roughly a factor 1.5);
- the gradient integrand (`calculateGradP2`) still grows like ln bMax and
  cancels against D3's log term. That cancellation is ~7 orders milder
  than the bMax² one and no ill effect is visible at bMax/σ = 10⁵, but at
  truly extreme ratios it would be the next thing to hit. The same
  splitting trick would apply.
- the fix was measured at L ∈ {40, 200} for the extreme sweep and L = 40
  for the fine sweep; other L values are covered only by the pre-existing
  (pre-fix) data.

## 10. Reproduce

```
cmake --build build --target lcd_experiment generate_samples -j
python analysis/run_sweeps.py --outdir analysis/results \
    --only bmax_fine,sigma_collapse,bmax_vs_L,bmax_extreme
python analysis/make_bmax_figures.py --results analysis/results \
    --outdir analysis/figures [--before <pre-fix-results-dir>]
# objective-replica self-check against any driver run:
build/plots/lcd_experiment.exe --L 40 --seed 42 --bMax 50 --out s.csv
python analysis/lcd_distance.py --samples s.csv --bMax 50 --dist <RESULT dist=...>
# formerly-crashing configuration, now fine:
GSL_RNG_SEED=2 build/plots/generate_samples.exe 2 20 out.csv 0.1 0.1 1000
```

To reproduce the before/after comparison, build the pre-fix binary by
reverting the `calculateP2` / `calculateD3` / `constantOffset` hunks
(commit touching `lib/gm_to_dirac/`), run `--only bmax_extreme` into a
separate directory, and pass it as `--before`.
