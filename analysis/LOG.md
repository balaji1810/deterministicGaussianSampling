# Running log — LCD sampler quality study

Working branch: `dev`. All numbers reproducible via `analysis/run_sweeps.py`
(driver: `plots/lcd_experiment.cpp`, additive target, not shipped).

## Phase 0 — source facts (verified against code, not README)

- Entry point for standard-normal samples: `gm_to_dirac_short<double>::approximate`
  (`lib/gm_to_dirac/gm_to_dirac_short.cpp:178`). `covDiag` holds per-axis
  **standard deviations** (evidence: passed as sigma to `gsl_ran_gaussian`;
  squared into `covDiagSqrd` where the math needs sigma^2).
- Flow: random Gaussian init from GSL RNG (`GSL_RNG_SEED` env; no seed option
  in the API) -> BFGS2 (`gsl_multimin_fdfminimizer_vector_bfgs2`,
  initialStepSize=0.1, line-search tol=0.15) -> `correctMean` (subtract
  weighted mean). **No covariance correction post-step exists.**
- Objective = D2 + D3 only (`combined_distance_metric`); the constant D1 is
  computed but never added (fine for optimization; reported distances are
  offset and can be negative).
- D2 = -2*Integral_0^bMax P2(b) db via adaptive Gauss-Kronrod (epsrel 1e-10);
  P2 matches the Gaussian-kernel LCD cross term with kernel weight
  w(b) = b^(1-N), global factor pi^(N/2) dropped consistently in D2 and D3.
- D3 = closed form from Hanebeck "Optimal Reduction" Thm 2 (Gaussian-kernel
  Dirac-Dirac term, large-bMax asymptotic; keeps one more Taylor term than the
  paper: coefficient Gamma-1-2log2 instead of Gamma-2log2).
- Stopping: ftolRel=1e-10 (default), gtol=1e-6, xtol disabled, maxIter=10000;
  `GSL_ENOPROG` is treated as *success*. Checks run in order f, x, g.
- `bMax` is a `size_t` (integer!), default 100. `cB = log(4 bMax^2)` is
  computed but unused (`(void)cB` in calculateD3).

### Inconsistencies noticed (for the report)
- `calculateD1` integrand uses sigma^2 + 2b^2; the math for Int F^2 gives
  sigma^2 + b^2. Harmless (D1 is x-independent, never added), but wrong if
  anyone ever reports absolute distances.
- Float wrapper `approximate(gsl_vector_float*...)`: inner
  `gsl_vector* wXDouble = gsl_vector_alloc(L);` shadows the outer variable ->
  custom float weights are silently ignored + leaked. Not on our path
  (double, null weights) but a real bug.
- README API (`approximate(covDiag, L, N, bMax, x, ...)`) is outdated: bMax
  moved into `ApproximateOptions` (commit 1590b4e).

## Phase 0 — first measurements (L=40, sigma=I, bMax=100, seed 42)

- Default options: stops after **32 iterations** via ftolRel;
  gradient norm still 3.1e-4 (gtol target 1e-6). dist = -4995.39425005.
- ftolRel=0: runs to **375 iterations** (ENOPROG), gradient 1.5e-6,
  dist = -4995.39429287 (improvement 4.3e-5 on top of a ~5e3 offset).
- Mechanism hypothesis (analytic): f carries an x-independent offset
  ~ 0.5*bMax^2 - (D2 tail) of magnitude ~bMax^2/4 => ftolRel * |f| becomes an
  *absolute* f tolerance ~ 5e-7 at bMax=100. The x-dependent part of f is
  O(1). So the default run stops when per-iteration progress ~ 1e-7 of the
  *offset*, i.e. very early. Larger bMax => earlier stop. Testable:
  bMax sweep at default vs strict tolerances.
- Asymptotics: for bMax >> spread, the x-dependent log(bMax) terms cancel
  between D2 and D3, so the *optimum location* should be bMax-independent;
  bMax affects convergence/conditioning, not the target configuration.

## Phase 1 — metrics implemented (`analysis/metrics.py`)

- L2 star discrepancy (Warnock) of u = Phi(x) uniformized samples.
- Moments: weighted, divisor L (the Dirac mixture *is* the distribution).
- NN-distance CV in u-space (primary homogeneity measure); min pairwise
  distance in x-space.

## Phase 2 — sweeps (~600 runs; results.csv in scratch results dir)

- baseline_L vs strict_L (ftolRel=0): with defaults, iterations *fall* with L
  (40 -> 17) and NN-CV(u) degrades 0.13 -> 0.37; strict: monotone improvement
  through L=200. Premature stop confirmed as the Q1 mechanism.
- bMax sweep (L=40): strict-stop quality saturates for bMax >= ~20-50 and
  stays flat to 500; default-stop quality peaks at bMax=20-50 and degrades
  beyond (stopping-offset coupling). bMax <= 3 catastrophic at sigma=1.
- sigma scaling: relative cov error matches at equal bMax/sigma
  (scale invariance verified); bMax/sigma ~ 1e4 made gsl_integration_qag
  fail hard -> process abort (GSL default handler). Driver now calls
  gsl_set_error_handler_off().
- multistart (20 seeds, L=40 strict): objective identical to 2e-6; NN-CV
  varies 0.10-0.16 -> near-degenerate optimum family; multi-start by
  distance is worthless.

## Phase 3/4 — bMax saturation

- Saturation mini-sweep L=100/200: sweet spot bMax = 50-100; 200-500 mildly
  worse (quadrature noise). LCD covariance shrinkage ~ 1/L (cov trace ratio
  ~ 1 - 1/L) is part of the optimum, not an error.

## Phase 5 — deliverables

- One-line fix applied: ApproximateOptions.ftolRel default 1e-10 -> 0.
- Docs figures regenerated (L=40 panel now homogeneous).
- Report: analysis/REPORT.md; figures: analysis/figures/.
- Local build caveats (pre-existing, unrelated to changes): benchmark_target
  does not compile against msys2 google-benchmark; unit_test_target links
  against an MSVC gtest from miniconda and fails under MinGW. Both build in
  CI (vcpkg). Verified: approxLCD_static/shared, generate_samples,
  lcd_experiment, main all build locally; unit tests don't touch
  approximate()/options.

## Meeting figure set

make_figures.py produces a self-contained meeting set that shows homogeneity
via NN-CV(u), accuracy via covariance error, effort via iteration count.
Figures:
  fig_quality_vs_L          iterations / NN-CV(u) / cov-error vs L
  fig_quality_vs_bmax       NN-CV(u) / cov-error / iterations vs bMax
  fig_scatter_before_after  x-space scatters (NN-CV + iters annotations only)
  fig_nncv_explainer        regular/random/clumpy sets + NN segments + NN hist
  fig_gaussian_to_uniform   x-space vs u=Phi(x); old vs fixed stop at L=100
  fig_optimizer_stops_early gradient-norm trajectory; default vs fixed stop
  fig_objective_offset      |f|~5000 vs signal~6e-3 vs ftolRel threshold ~5e-7

## Gradient-norm trace (analysis/trace_gradnorm.py)

The trajectory of |grad| vs BFGS iteration is reconstructed WITHOUT touching
the shipped library: the optimizer path is deterministic given the seed, so
re-running with an increasing --maxIter cap and reading the reported lastGtol
recovers the exact per-iteration gradient norm. (Preferred over instrumenting
gsl_minimizer, which would mean plumbing a trace through the public
ApproximateOptions.) Writes results/trace_gradnorm.csv; consumed by
fig_optimizer_stops_early. Confirms: default (ftolRel=1e-10) halts at 32 iters
with |grad|~3e-4 (~300x above gtol=1e-6); fixed run reaches target at 375.

Reproduce the meeting set:
  cmake --build build --target lcd_experiment -j
  python analysis/trace_gradnorm.py --outdir <results-dir> --L 40 --seed 42
  python analysis/make_figures.py --results <results-dir> --outdir analysis/figures
