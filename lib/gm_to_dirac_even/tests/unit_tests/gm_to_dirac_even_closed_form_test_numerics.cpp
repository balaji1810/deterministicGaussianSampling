#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "gm_to_dirac_even_closed_form.h"
#include "gm_to_dirac_short_standard_normal_deviation.h"
#include "lcd_even_closed_form.h"

namespace {

std::vector<double> randomSamples(gsl_rng* rng, size_t L, size_t N) {
  std::vector<double> x(L * N);
  for (size_t i = 0; i < L * N; ++i) x[i] = gsl_ran_gaussian(rng, 1.00);
  return x;
}

// max |closed form - quadrature path| over the gradient, at one bMax
double gradientGap(size_t L, size_t N, double bMax, std::vector<double> x) {
  std::vector<double> closedForm(L * N, 0.00);
  std::vector<double> quadrature(L * N, 0.00);

  gm_to_dirac_even_closed_form<double> gm2dEven;
  gm2dEven.modified_van_mises_distance_sq_derivative(closedForm.data(), L, N,
                                                     bMax, x.data(), nullptr);

  gm_to_dirac_short_standard_normal_deviation<double> gm2dShort;
  gm2dShort.modified_van_mises_distance_sq_derivative(
      quadrature.data(), L, N, static_cast<size_t>(bMax), x.data(), nullptr);

  double worst = 0.00;
  for (size_t i = 0; i < L * N; ++i)
    worst = std::max(worst, std::abs(closedForm[i] - quadrature[i]));
  return worst;
}

}  // namespace

// The quadrature path computes the same objective, with two known deviations:
// it omits D1, and its D3 uses a large-bMax logarithmic approximation instead
// of the exact Ei form. Asserting the residual falls as 1/bMax^2 is a much
// stronger statement than any fixed epsilon: it only holds if the two really
// are the same object differing by that one approximation.
TEST(gm_to_dirac_even_closed_form_numerics_test, matches_quadrature_path) {
  gsl_rng_env_setup();
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 424242);

  for (size_t N : {2U, 4U}) {
    for (size_t L : {5U, 40U}) {
      const std::vector<double> x = randomSamples(rng, L, N);

      const double atTen = gradientGap(L, N, 10.00, x);
      const double atHundred = gradientGap(L, N, 100.00, x);

      ASSERT_GT(atHundred, 0.00) << "N=" << N << " L=" << L;
      const double ratio = atTen / atHundred;
      EXPECT_GT(ratio, 50.00) << "N=" << N << " L=" << L;
      EXPECT_LT(ratio, 200.00) << "N=" << N << " L=" << L;
    }
  }

  gsl_rng_free(rng);
}

// D1, D2 and D3 are each O(bMax^2) while the distance is O(1e-4), so building
// the objective at full magnitude leaves it with an absolute noise floor of
// bMax^2 * eps. The gradient stays clean (its terms are O(ln bMax^2)), so BFGS
// keeps a good direction, but the line search compares function values and
// stops being able to verify progress: the optimizer then bails on no-progress
// short of the gradient criterion and the achieved distance degrades with
// bMax. The objective therefore uses the reduced, O(1) forms.
//
// Before that fix, from this start: 203 / 69 / 36 iterations at
// bMax = 100 / 1000 / 10000, distances 8.21e-5 / 9.03e-5 / 1.12e-4, terminal
// |grad| up to 117x gtol.
TEST(gm_to_dirac_even_closed_form_numerics_test, optimum_is_invariant_in_bmax) {
  const size_t L = 100;
  const size_t N = 2;

  gsl_rng_env_setup();
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 20240815);
  const std::vector<double> start = randomSamples(rng, L, N);
  gsl_rng_free(rng);

  ApproximateOptions options;
  options.initialX = true;

  gm_to_dirac_even_closed_form<double> gm2dEven;
  std::vector<double> distances;
  std::vector<size_t> iterations;

  for (double bMax : {100.0, 1000.0, 10000.0}) {
    std::vector<double> x = start;
    GslminimizerResult result;
    ASSERT_TRUE(
        gm2dEven.approximate(L, N, bMax, x.data(), nullptr, &result, options));

    // must reach the gradient criterion, not bail on no-progress. This is the
    // norm at the minimizer's own iterate; the returned x differs by the
    // post-optimisation mean correction, which leaves |grad| a few times
    // larger while moving the distance by ~1e-10.
    EXPECT_LE(result.lastGtol, options.gtol) << "bMax=" << bMax;

    double d = 0.00;
    gm2dEven.modified_van_mises_distance_sq(&d, L, N, bMax, x.data(), nullptr);
    distances.push_back(d);
    iterations.push_back(result.iterations);
  }

  for (size_t i = 1; i < distances.size(); ++i)
    EXPECT_LT(std::abs(distances[i] / distances[0] - 1.00), 0.02)
        << distances[i] << " vs " << distances[0];

  for (size_t i = 0; i < iterations.size(); ++i)
    for (size_t j = 0; j < iterations.size(); ++j)
      EXPECT_LT(iterations[i], 3 * iterations[j])
          << iterations[i] << " vs " << iterations[j];
}

// lcd_small_c_threshold() hard-codes, per k, the c below which the closed
// forms cancel badly enough to need quadrature. Nothing else pins that table,
// so re-run the measurement behind it: accurate at the threshold, and NOT
// accurate two decades below it (otherwise the value is needlessly large).
//
// Two decades, not one: the table sits one to one-and-a-half decades above the
// measured failure boundary, deliberately. The fallback is exact, so
// overshooting only costs time, and it costs almost none - the fraction of
// samples with c = ||s||^2 below these thresholds is under 3e-4 for k <= 5.
TEST(gm_to_dirac_even_closed_form_numerics_test, small_c_thresholds_hold) {
  auto worstRelativeError = [](size_t k, double c) {
    double worst = 0.00;
    for (double bMax : {2.0, 100.0, 10000.0}) {
      const double reduced = lcd_delta_bkk_closed(k, bMax, c, 0, true) -
                             lcd_delta_bkk_zero_minus_leading(k, bMax);
      const double reducedRef = lcd_delta_bkk_quadrature(k, k, bMax, c, true);
      worst = std::max(worst,
                       std::abs(reduced - reducedRef) / std::abs(reducedRef));

      const double gradient = lcd_delta_bkk_closed(k, bMax, c, 1);
      const double gradientRef = lcd_delta_bkk_quadrature(k, k + 1, bMax, c);
      worst = std::max(
          worst, std::abs(gradient - gradientRef) / std::abs(gradientRef));
    }
    return worst;
  };

  for (size_t k = 1; k <= 6; ++k) {
    const double threshold = lcd_small_c_threshold(k);
    EXPECT_LT(worstRelativeError(k, threshold), 1e-10) << "k=" << k;
    EXPECT_GT(worstRelativeError(k, threshold / 100.0), 1e-10)
        << "k=" << k << ": threshold " << threshold << " can be tightened";
  }
}
