# Why the LCD standard-normal samples look worse at larger L — and how to fix it

Study of `gm_to_dirac_short` (2D standard normal, Gaussian-kernel LCD /
modified Cramér–von Mises distance, BFGS2). All claims below are grounded in
source code, the papers' math, or measurements produced by
`analysis/run_sweeps.py` + `analysis/metrics.py` (commands at the end).
Every number is a mean over 4 seeds unless stated otherwise.

**TL;DR** — The optimizer is stopped long before convergence by the default
relative f-tolerance (`ftolRel = 1e-10`), because the objective carries an
x-independent offset of magnitude ≈ bMax²/2 (≈ 5000 at the default
bMax = 100). The premature stop gets *worse* as L grows, which is exactly why
the L=40 panel in the docs figure looks non-homogeneous. Fixing the stopping
rule (one line) restores clean, monotone quality-vs-L behavior. bMax itself
is best chosen as ≈ 50–100 × σ_max; the default 100 is fine for σ = 1.

---

## Q1 — Why does visual smoothness fail to improve (or worsen) with L?

**Answer (solid): premature termination, not the LCD objective and not local
minima.** Mechanism, verified step by step:

1. The minimized objective is D2 + D3 (D1 is an x-independent constant that
   is omitted). D3 contains an exactly constant term
   ½·bMax²·(Σᵢwᵢ)² and D2 an asymptotically constant −bMax²·Σᵢwᵢ, so
   f ≈ −bMax²/2 + (x-dependent part of order 1). At bMax=100, |f| ≈ 4995
   while the entire x-dependent signal is O(1).
2. The stop rule fires when |Δf|/|f| ≤ 1e-10, i.e. at an *absolute*
   per-iteration progress of ≈ 5·10⁻⁷ — thousands of times coarser than the
   objective differences that separate a clumpy set from a homogeneous one
   (the total remaining improvement at the stop is ~4·10⁻⁵).
3. Each BFGS iteration moves f less when L is larger (each point carries
   weight 1/L), so the criterion fires *earlier* for larger L:
   measured iterations until stop fall from ~40 (L=10) to ~17 (L=200),
   for an 80–400-dimensional problem. With the stop disabled, the optimizer
   wants 150–600 iterations.

Measured consequences (bMax=100, defaults vs `ftolRel=0`; NN-CV(u) =
coefficient of variation of nearest-neighbour distances of the uniformized
samples, lower = more homogeneous):

| L | iterations (default→strict) | NN-CV(u) default | NN-CV(u) strict |
|---|---|---|---|
| 20 | 39 → 238 | 0.130 | 0.132 |
| 40 | 28 → 423 | 0.158 | 0.139 |
| 100 | 19 → 283 | **0.281** | 0.143 |
| 200 | 17 → 144 | **0.368** | 0.152 |

With defaults, the nearest-neighbour homogeneity degrades monotonically as L
grows (NN-CV(u) 0.13 → 0.37 from L=10 to L=200) — the "more samples look
worse" effect. With the strict stop it stays flat (~0.13–0.15) and the
covariance error keeps decreasing. See `figures/fig_quality_vs_L.png` and the
before/after scatter `figures/fig_scatter_before_after.png`.

**Ruled out (solid):**
- *Local minima / bad starts*: 20 random seeds at L=40 (strict) all reach the
  same objective within 2·10⁻⁶. Multi-start on the distance buys nothing.
- *Intrinsic LCD limitation*: after the fix, homogeneity (NN-CV(u)),
  discrepancy, and covariance error all improve monotonically with L
  (`fig_quality_vs_L`). The earlier plateau was the stopping bug, not a
  ceiling of LCD-via-optimization.
- *Awkward L values*: no special-L structure remains once converged.

**Nuance (solid but secondary):** the converged optima are a *near-degenerate
family*: across seeds the objective agrees to 7 digits while NN-CV(u) varies
0.10–0.16 (isotropic targets are optimal only up to rotation, plus genuinely
distinct near-tied configurations). The distance cannot discriminate these; if
one cares about the last ~10% of homogeneity, select best-of-K by NN-CV(u) or
discrepancy, not by distance.

## Q2 — Can it be fixed? Is something wrong in the code?

Yes. The objective, gradient, and integration are **correct** (I verified the
D3 closed form against Theorem 2 of the Dirac-reduction paper — the code even
keeps one extra Taylor term — and the D2 integrand against the Gaussian-kernel
LCD cross term with weight w(b)=b^(1−N); both drop the same global π^(N/2)).
What is wrong is the **default stopping configuration**, an optimizer-usage
bug, not a math bug:

- `ftolRel = 1e-10` is meaningless against an objective with a ~bMax²/2
  offset. Fix: disable it (rely on `gtol` and the line-search no-progress
  condition, which in practice terminate all runs well below
  `maxIterations`). Applied as a one-line default change in
  `lib/options/approximate_options.h` on branch `dev`.
- Cost of the fix: 3–17× more iterations; absolute runtimes stay small
  (L=40: 29 ms → 0.5 s; L=200: 0.13 s → 1.3 s on this machine).
  Callers who want the old speed can set `ftolRel = 1e-10` explicitly.

## Q3 — Optimal bMax / selection rule

> **Dedicated study:** `bMax_analysis.md` treats this question in depth
> (mechanism figures, the 1/bMax² law, scale-invariance collapse, failure
> galleries) and refines the rule to **bMax ≈ 10–15·σ_max·√L**, with
> 50–100·σ_max remaining a fine flat-rate for L ≲ 150.

**Rule (solid): bMax ≈ 50–100 × σ_max, as an integer (the field is
`size_t`).** Evidence:

- The x-dependent log(bMax) terms of D2 and D3 cancel asymptotically, so the
  *optimum location* stops depending on bMax once bMax ≫ point spread.
  Measured: at L=40 (strict), NN-CV(u) and covariance error saturate from
  bMax ≈ 20–50 and stay flat through 500; at L=100–200 the sweet spot is
  50–100.
- Scale invariance confirmed: relative covariance error matches across
  (σ, bMax) pairs at equal bMax/σ (σ=1,bMax=10 → 6.4% ≡ σ=10,bMax=100 →
  6.4%; σ=1,bMax=100 → 3.3% ≡ σ=10,bMax=1000 → 3.3%).
- Too small (bMax ≲ 10σ) is catastrophic: at bMax=σ the kernel window never
  sees the global structure (covariance error ≈ 100%, samples badly
  inhomogeneous).
- Too large is mildly harmful even with the strict stop (quadrature noise:
  covariance error and NN-CV(u) creep back up from bMax≈100 to 500) and
  *very* large ratios (bMax/σ ≈ 10⁴) make `gsl_integration_qag` fail — with
  GSL's default error handler this **aborts the process** (reproduced at
  σ=0.1, bMax=1000).
- With the *old* default stopping, bMax also controls how early ftolRel
  fires: quality at L=40 peaked at bMax=20–50 and degraded beyond. After the
  fix this coupling disappears — that was the main practical "bMax effect".
- The anisotropic case is handled upstream (samples generated for
  σ = √eigenvalues, then rotated), so σ_max is the largest target std dev;
  for strongly anisotropic diagonals σ_max is the safe choice.

Default bMax=100 is therefore right for σ=1; a σ-aware default like
`bMax = ceil(100·σ_max)` (clamped ≥ 10) would make the library robust for
σ ≠ 1. Not applied — needs a maintainer decision about API defaults.

## Q4 — Code modifications needed?

Applied on `dev` (all additive except one default):

1. **`lib/options/approximate_options.h`** — `ftolRel` default `1e-10 → 0`
   (with explanatory comment). This is the fix. One line, revertable,
   unit tests unaffected (they only test derivatives).
2. **`plots/lcd_experiment.cpp` (+ CMake target)** — experiment driver
   exposing seed/tolerances/bMax/warm-start; also calls
   `gsl_set_error_handler_off()` so quadrature failures return an error
   instead of aborting. Not shipped in release artifacts.
3. **`analysis/`** — metrics (L2 star discrepancy, moments, NN stats),
   sweep harness, figure generation, this report.

Recommended but *not* applied (maintainer decisions):
- σ-aware bMax default (above).
- A library-level policy for GSL integration errors (currently any hard
  quadrature failure aborts the caller's process).
- Optional exact-covariance post-step behind a flag: the LCD optimum has a
  covariance trace ratio ≈ 1 − 1/L (a deliberate shrinkage), so whitening to
  the exact target covariance would zero the moment error but perturb the
  homogeneity/discrepancy optimum — opt-in, use-case-driven.
- Bug (off my path, found while reading): the float wrapper of
  `approximate` shadows `wXDouble` in an inner scope
  (`gm_to_dirac_short.cpp:147`), so float-precision custom weights are
  silently ignored and leaked.

## Tradeoffs of each candidate fix

| Fix | Homogeneity gain (NN-CV(u)) | Moment gain | Cost |
|---|---|---|---|
| **Strict stop (applied)** | NN-CV(u) stops degrading with L: 0.28→0.14 at L=100, 0.37→0.15 at L=200 (0.16→0.14 at L=40) | covariance error similar, slightly better at large L; mean is exact either way (`correctMean`) | 3–17× iterations (≤1.3 s at L=200); no optimality property sacrificed |
| bMax = 50–100σ rule | prevents the two failure modes (tiny bMax: catastrophic; huge bMax: quadrature noise/crash) | same | none; larger bMax costs linearly in integration time |
| Multi-start (best of 20 by distance) | none (objective spread 2·10⁻⁶) | none | 20× compute — **not worth it** |
| Best-of-K selected *by NN-CV(u)* | up to ~35% NN-CV at L=40 | none | K× compute + metric evaluation; only worthwhile for precomputed libraries |

## Solid vs uncertain

Solid: everything marked (solid) above; the mechanism (offset ≈ bMax²/2 —
derived and matched numerically: measured |f| = 4995.39 at bMax=100); the
sweep trends; the D3↔reduction-paper equivalence; the covariance shrinkage
≈ 1/L.

Uncertain / open:
- ENOPROG ("line search makes no progress") is treated as success and is now
  the de-facto terminator; final gradient norms (10⁻⁶–10⁻⁵) say this is fine
  in practice, but a principled absolute-f or gradient-only criterion would
  be cleaner.
- NN-CV(u) is a local homogeneity measure; cross-seed differences of ≲5%
  partly reflect the rotational freedom of the isotropic optimum, so the
  exact "best-of-K by NN-CV(u)" gains wobble ±few %.
- The bMax upper comfort bound (quadrature noise onset) was only mapped
  coarsely (100 → 500 at L≤200, 2D); higher L / N not measured.
- All results are 2D; the mechanism is dimension-independent (the offset and
  stopping interact identically), but constants like "50–100σ" were not
  re-measured for N>2.

## Contradictions found (expected: README is outdated)

- README's `approximate(covDiag, L, N, bMax, x, ...)` signature is stale:
  bMax lives in `ApproximateOptions` since commit 1590b4e. `covDiag` holds
  per-axis **standard deviations**, not variances (evidenced by
  `gsl_ran_gaussian(r, covDiag[k])` and by squaring into `covDiagSqrd`).
- `calculateD1` integrates ∏ b/√(σ²+2b²); the math for ∫F² gives σ²+b².
  Harmless today (D1 never enters optimization or reported distances) but
  wrong if absolute distances are ever reported.
- Reported "distance" omits D1 and is offset by the bMax² constant — it can
  be negative and is only comparable between runs with identical bMax/σ/N.

## Reproduce

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHOWCASE=ON
cmake --build build --target lcd_experiment generate_samples -j
python analysis/run_sweeps.py --outdir analysis/results          # ~25 min
python analysis/make_figures.py --results analysis/results --outdir analysis/figures
python analysis/metrics.py                                    # self-tests
python plots/plot_samples.py                                  # docs figures
```

Single illustrative run (the smoking gun):

```
build/plots/lcd_experiment.exe --L 40 --seed 42 --ftolRel 1e-10 --verbose  # old default: stops at 32 iters, |grad| 3e-4
build/plots/lcd_experiment.exe --L 40 --seed 42 --verbose                  # fixed default: ~375 iters, |grad| ~1e-6
```
