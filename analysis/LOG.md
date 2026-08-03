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

## bMax deep-dive (analysis/bMax_analysis.md)

Dedicated study of the bMax selection rule for the supervisor discussion.
New opt-in sweeps in run_sweeps.py (--only bmax_fine,sigma_collapse,
bmax_vs_L; 228 runs, ~4 min, all converged). New tools:
- lcd_distance.py: exact Python replica of the code's reported distance
  (-2*int P2 + D3), validated to rel. 3e-15 against the driver at
  bMax 3/50/500. Used to measure how much discriminating signal each
  kernel scale contributes without re-optimizing.
- make_bmax_figures.py: six figures (fig_bmax_role, _information,
  _quality, _scale_invariance, _convergence_law, _failure_gallery).

Headline findings (details + numbers in bMax_analysis.md):
- D(shrunk x0.8) - D(optimal) is NEGATIVE for bMax <~ 4*sigma: tiny bMax
  genuinely prefers collapsed sets (explains failure mode 1).
- Information beyond cutoff decays as 1/bMax^2 (two decades, slope -2);
  bias law: trace deficit ~ 6*(sigma/bMax)^2, c in [5.6, 6.7].
- Scale invariance verified to 4 decimals across sigma 0.1..10:
  only bMax/sigma matters.
- Noise ceiling: f-eval noise ~ 1e-10*bMax^2 (quadrature epsrel on a
  bMax^2/2-sized value) -> iterations fall 423->145 from bMax 100->1000
  at L=40; visible quality damage at L=200 for bMax >= 300-500.
- Seed-dependent process abort from bMax/sigma ~ 1e4 (sigma=0.1 bMax=1000
  crashes with GSL_RNG_SEED=2,3; not 1,42). size_t bMax means sigma <~
  0.001 CANNOT be given a safe bMax -> API limitation to raise.
- Rule refined: bMax ~ (10-15)*sigma*sqrt(L) (knees measured at 50/50/
  100/200 sigma for L=20/40/100/200); flat 50-100*sigma fine for L <~ 150.

## Removing the bMax ceiling (objective offset fix)

Follow-up to the bMax study: the *upper* bMax failure mode was numerical,
not mathematical, so it was removed at the source instead of being turned
into a recommended limit.

Derivation (verified in lcd_distance.py to machine precision):
  f_full(x, B) = f_reduced(x, B) + C(B),
  C(B) = -2*W*int_0^B prefactor(b) db + 0.5*B^2*W^2,   W = sum_i w_i
C is entirely x-independent and contains the whole divergence
(-B^2/2 + ln B); f_reduced CONVERGES: -0.345852 (B=50) -> -0.346131 (1000)
-> -0.346132 (1e4). So larger bMax is mathematically always fine.

Because the quadrature works to a RELATIVE tolerance (1e-10), carrying C
inside the optimizer's objective made the absolute per-evaluation noise
~1e-10*B^2/2 (5e-7 at B=100, 5e-3 at B=1e4) against an O(0.35) signal whose
late-stage steps are ~1e-6. That, and nothing else, produced the iteration
collapse, the quality loss at large bMax, and the qag aborts.

Fix (lib/gm_to_dirac/gm_to_dirac_short.{h,tpp}, .cpp; ~15 lines, no API
change): calculateP2 integrates sum_i w_i*expm1(...) instead of
sum_i w_i*exp(...) (expm1 is essential -- the exponent tends to 0, so
exp(...)-1 would lose all digits); calculateD3 drops the per-pair
0.5*bMax^2 term; new constantOffset() adds C back ONLY in the public
reporting path, so reported distances stay comparable across versions.

Verification:
- reported distance vs pre-fix binary: agrees to <=1.3e-11 rel at
  bMax=10/100/1000.
- analytic gradient vs central FD: ~1e-7 at the optimal step size (swept h;
  at bMax=1000 the FD probe itself is limited by the offset still present
  in the *reported* value -- an independent demo of the same disease).
- quality now FLAT to bMax=1e4 at L=40 and L=200 (see table in
  bMax_analysis.md section 8); at L=200 large bMax is now BETTER
  (cov err 0.0088 -> 0.0071).
- all three previously-aborting configs now succeed; sigma=0.01/bMax=1000
  (ratio 1e5) gives trace ratio 0.956, NN-CV(u) 0.081.
- low-bMax behaviour unchanged (cov err 0.1126 at bMax=10, before & after).
- build + docs pipeline (plots/plot_samples.py) verified.

New sweep: run_sweeps.py --only bmax_extreme (L in {40,200} x bMax in
{100,300,1000,3000,10000} x 2 seeds). New figure: fig_bmax_offset_fix.png
(make_bmax_figures.py --before <pre-fix-results-dir>).

## fig_samples_before_after.png

Direct visual of the offset fix: the actual point sets, 2x3, rows =
pre-fix / current dev, columns = bMax 100 / 1000 / 10000, all at N=2,
sigma=I, L=200, GSL_RNG_SEED=42, library defaults (ftolRel=0 in both).

The pre-fix binary is built from a throwaway git worktree of dev with only
lib/gm_to_dirac/gm_to_dirac_short.{h,tpp,cpp} checked out from main, so the
offset removal is the sole difference (dev itself is never touched, and
approximate_options.h keeps dev's ftolRel=0). Two notes for whoever repeats
this: git needs -c core.longpaths=true for this repo's deepest test paths,
and the worktree must live at a SHORT path (the build output paths blow
past MAX_PATH otherwise); C:/Users/balaj/AppData/Local/Temp/lcdpre worked.

Driver: run_sweeps.py --only samples_before_after, run once per binary into
separate --outdir trees; figure via make_bmax_figures.py
--samples-before <dir> --samples-after <dir>, which also prints the six
(NN-CV, iterations) pairs.

  row       bMax    NN-CV(u)   iters
  before     100      0.1409     171
  before    1000      0.2748      46
  before   10000      0.5644       2
  after      100      0.1590     395
  after     1000      0.1515     291
  after    10000      0.1613     355

At bMax=10000 the pre-fix run stops after 2 iterations, so its panel is
essentially the random initialisation; the post-fix panels are visually
identical lattices at all three bMax values.
