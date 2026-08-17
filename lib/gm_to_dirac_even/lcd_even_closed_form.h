#ifndef LCD_EVEN_CLOSED_FORM_H
#define LCD_EVEN_CLOSED_FORM_H

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_expint.h>

#include <cassert>
#include <cmath>
#include <cstddef>

#include "gsl_quadrature_adaptive_gauss_kronrod.h"

/**
 * @file lcd_even_closed_form.h
 * @brief quadrature-free closed forms for the LCD / modified Cramer-von-Mises
 * distance of an even-dimensional standard normal target
 *
 * The mCvM distance of a Dirac mixture with samples s_i and weights w_i
 * against N(0, I) decomposes into
 *
 *     D = D1 - 2 * D2 + D3
 *
 * where D1 is sample-independent, D2 is the attraction term (samples towards
 * the mean) and D3 is the repulsion term (samples away from each other).
 *
 * All quantities in this header are expressed in units of pi^(N/2), i.e. they
 * are pi^(-N/2) * D. This is the same normalisation the quadrature path in
 * lib/gm_to_dirac uses (GMToDiracBaseOptimizationParams::twoPiNHalf is
 * pow(2, N/2), the pi^(N/2) having been divided out of the whole objective),
 * so the two can be compared without rescaling.
 *
 * @note Preconditions for the D2 helpers (lcd_delta_bkk, lcd_delta_bkk1):
 * the target must be the STANDARD normal N(0, I) and N = 2k must be EVEN.
 * lcd_a_n and lcd_c_repulsion are valid for all N.
 */

/**
 * @brief argument below which Ei is truncated to zero
 *
 * gsl_sf_expint_Ei_e() reports GSL_EUNDRFLW for arguments below approximately
 * -701.83 (measured by bisection against GSL 2.8). That status is raised
 * through gsl_error(), so with GSL's DEFAULT error handler installed the call
 * aborts the process rather than returning - the status can therefore not be
 * relied upon and the argument has to be guarded before the call.
 *
 * The cutoff is placed at -600, where |Ei(x)| < 4.5e-264: far below the
 * round-off of every sum this value enters, and with about 100 of margin to
 * the measured underflow point.
 */
#define LCD_EI_CUTOFF (-600.00)

/**
 * @brief exponential integral Ei(x) for negative arguments, underflow-safe
 *
 * @param x argument, must be strictly negative
 * @return Ei(x), or exactly 0.0 for x <= LCD_EI_CUTOFF
 */
inline double lcd_ei_safe(double x) {
  // every call site evaluates Ei of a negative quantity; a non-negative
  // argument means the caller computed the wrong thing (Ei(0) = -inf).
  assert(x < 0.00);

  if (x <= LCD_EI_CUTOFF) return 0.00;

  gsl_sf_result result;
  const int status = gsl_sf_expint_Ei_e(x, &result);

  // only reachable if the caller installed a non-aborting GSL error handler;
  // Ei(-z) -> 0 for large z, so an underflow is a correct answer, not a fault.
  if (status == GSL_EUNDRFLW) return 0.00;
  assert(status == GSL_SUCCESS);

  return result.val;
}

/**
 * @brief D1, the sample-independent term of the mCvM distance
 *
 * Evaluates A_N(b) = integral_0^b x^(N+1) / (1 + x^2)^(N/2) dx via the
 * recursion
 *
 *     A_1(b) = 0.5 * (b * sqrt(1 + b^2) - asinh(b))
 *     A_2(b) = 0.5 * (b^2 - log(1 + b^2))
 *     A_n(b) = (n * A_{n-2}(b) - b^n / (1 + b^2)^((n-2)/2)) / (n - 2)
 *
 * applied iteratively. Valid for all N, odd and even.
 *
 * @param N dimension, must be >= 1
 * @param b upper integration bound, must be >= 0
 * @return A_N(b)
 */
inline double lcd_a_n(size_t N, double b) {
  assert(N >= 1);
  assert(b >= 0.00);

  const double bSqrd = b * b;

  // seed the recursion on the base case of matching parity
  size_t n = (N % 2 == 0) ? 2 : 1;
  double aN = (N % 2 == 0)
                  ? 0.50 * (bSqrd - std::log1p(bSqrd))
                  : 0.50 * (b * std::sqrt(1.00 + bSqrd) - std::asinh(b));

  for (n += 2; n <= N; n += 2) {
    const double dn = static_cast<double>(n);
    // b^n / (1+b^2)^((n-2)/2) written as b^2 * (b^2/(1+b^2))^((n-2)/2): the
    // base is < 1, so this cannot overflow the way the literal form does.
    const double tail =
        bSqrd * std::pow(bSqrd / (1.00 + bSqrd), (dn - 2.00) / 2.00);
    aN = (dn * aN - tail) / (dn - 2.00);
  }

  return aN;
}

/**
 * @brief C(b, c), the exact per-pair repulsion kernel of D3
 *
 * C(b, c) = integral_0^b x * exp(-c / (4 x^2)) dx
 *         = (b^2 / 2) * exp(-c / (4 b^2)) + (c / 8) * Ei(-c / (4 b^2))
 *
 * This is exact for any b > 0 - unlike the large-bMax logarithmic
 * approximation used by gm_to_dirac_short::calculateD3, it carries no
 * O(1/bMax^2) error.
 *
 * @param b upper integration bound, must be > 0
 * @param c squared distance between the two samples, must be >= 0
 * @return C(b, c); C(b, 0) = b^2 / 2
 */
inline double lcd_c_repulsion(double b, double c) {
  assert(b > 0.00);
  assert(c >= 0.00);

  const double bSqrd = b * b;
  if (c <= 0.00) return 0.50 * bSqrd;  // coincident samples

  const double z = -c / (4.00 * bSqrd);
  return 0.50 * bSqrd * std::exp(z) + 0.125 * c * lcd_ei_safe(z);
}

/**
 * @brief C(b, c) - b^2/2, the repulsion kernel with its leading term removed
 *
 * C(b, c) is O(bMax^2) for every pair, but the assembled D3 is O(1): the
 * bMax^2/2 cancels against D1 and D2 in the total. Summing the full kernel and
 * cancelling afterwards costs the objective all its significant digits once
 * bMax is large - the whole point of the reduced form is to never form that
 * cancellation. Using expm1 makes the subtraction exact rather than a
 * difference of two nearly equal numbers.
 *
 * Removed from D3 in total: (sum_i w_i)^2 * bMax^2 / 2.
 *
 * @param b upper integration bound, must be > 0
 * @param c squared distance between the two samples, must be >= 0
 * @return C(b, c) - b^2/2; exactly 0 at c = 0, so coincident samples
 * (including every i == j pair) now contribute nothing
 */
inline double lcd_c_repulsion_reduced(double b, double c) {
  assert(b > 0.00);
  assert(c >= 0.00);

  if (c <= 0.00) return 0.00;  // coincident samples

  const double bSqrd = b * b;
  const double z = -c / (4.00 * bSqrd);
  return 0.50 * bSqrd * std::expm1(z) + 0.125 * c * lcd_ei_safe(z);
}

/**
 * @brief smallest c for which the closed-form D2 helpers stay accurate
 *
 * Both dBkk and dBkk1 cancel catastrophically as c -> 0: the dB_{0,d} terms
 * carry 1/c^(d-j+1) poles that only cancel between the two halves of the
 * difference. dBkk1 is always the worse of the two, but dBkk is NOT immune -
 * at k = 4 it is already 30x off at c = 1e-6. Below the returned threshold
 * all of them are computed by quadrature instead.
 *
 * Calibrated by scanning c over a half-decade grid and comparing the closed
 * forms against lcd_delta_bkk_quadrature() for bMax in
 * {2, 5, 20, 100, 1e3, 1e4}, taking the worst over lcd_delta_bkk_reduced()
 * and dBkk1. The tabulated value is one decade above the largest c at which
 * either still exceeded 1e-10 relative error. Measured worst case:
 *
 *     k | failure boundary | threshold | err at threshold | err a decade below
 *     --+------------------+-----------+------------------+-------------------
 *     1 |            1e-05 |     1e-04 |          1.0e-11 |            2.3e-09
 *     2 |            1e-03 |     1e-02 |          2.0e-12 |            2.1e-10
 *     3 |            1e-02 |     1e-01 |          1.3e-12 |            5.8e-07
 *     4 |            3e-02 |     1e+00 |          6.5e-15 |            3.4e-11
 *     5 |            1e-01 |     1e+00 |          1.3e-13 |            6.8e-09
 *     6 |            3e-01 |     3e+00 |          7.3e-15 |            2.7e-06
 *
 * The table is driven by lcd_delta_bkk_reduced(), which is far more demanding
 * than the unreduced form it replaced in the objective: the unreduced value is
 * O(bMax^2) while the reduced one is O(c * ln U), so the same absolute
 * cancellation error is a vastly larger relative error. Only k = 1 actually
 * moved (1e-5 -> 1e-4); the old table sat exactly on the new boundary there.
 *
 * lcd_even_closed_form_small_c_test.cpp re-runs this measurement and asserts
 * the thresholds still hold.
 *
 * The fallback is exact, so overshooting only costs time. It costs very
 * little: with c = ||s||^2 chi-squared with N = 2k degrees of freedom, the
 * fraction of samples routed to quadrature is below 3e-4 for every k <= 5 and
 * about 5e-3 for k = 6.
 *
 * @param k half the dimension, N = 2k
 * @return threshold on c below which quadrature must be used
 */
inline double lcd_small_c_threshold(size_t k) {
  assert(k >= 1);

  switch (k) {
    case 1: return 1e-4;
    case 2: return 1e-2;
    case 3: return 1e-1;
    case 4: return 1.00;
    case 5: return 1.00;
    case 6: return 3.00;
    default: break;
  }

  // k >= 7 is outside the calibrated range. Deliberately conservative: this
  // sends essentially every sample down the (exact, but slower) quadrature
  // path rather than risk silently wrong values. Extend the table above after
  // measuring if that regime matters.
  return 1e2;
}

/**
 * @brief parameters of the raw b-integrand used by the small-c fallback
 */
struct LcdDeltaBIntegrandParams {
  size_t k;      ///< half the dimension
  size_t q;      ///< exponent of (1 + 2b^2); k for dBkk, k+1 for dBkk1
  double cValue; ///< squared norm of the sample
  bool reduced;  ///< true to use expm1, giving dBkk(c) - dBkk(0) directly
};

/**
 * @brief b^(2k+1) * (1 + 2b^2)^(-q) * exp(-c / (2 (1 + 2b^2)))
 *
 * With reduced set, exp is replaced by expm1, so the integral evaluates
 * dBkk(c) - dBkk(0) directly and never forms the O(bMax^2) intermediate -
 * the same trick gm_to_dirac_short::calculateP2 uses.
 *
 * @param b integration variable
 * @param params LcdDeltaBIntegrandParams
 * @return value of the integrand at b
 */
inline double lcd_delta_b_integrand(double b, void* params) {
  const LcdDeltaBIntegrandParams* p =
      static_cast<const LcdDeltaBIntegrandParams*>(params);

  const double u = 1.00 + 2.00 * b * b;
  // b^(2k+1)/u^k factored as b * (b^2/u)^k: b^2/u < 0.5, so no overflow for
  // any k, unlike evaluating the two powers separately.
  double value = b * std::pow(b * b / u, static_cast<double>(p->k));
  for (size_t i = p->k; i < p->q; ++i) value /= u;

  const double exponent = -p->cValue / (2.00 * u);
  return value * (p->reduced ? std::expm1(exponent) : std::exp(exponent));
}

/**
 * @brief dBkk / dBkk1 by adaptive quadrature of the raw b-integrand
 *
 * Fallback for small c, where the closed forms lose accuracy to cancellation.
 * This path is entered per sample and only for samples very close to the
 * mean, so its cost is irrelevant.
 *
 * @param k half the dimension, N = 2k
 * @param q exponent of (1 + 2b^2); k for dBkk, k+1 for dBkk1
 * @param bMax upper integration bound, must be > 0
 * @param c squared norm of the sample, must be >= 0
 * @param reduced true to integrate the expm1 form, returning
 * dBkk(c) - dBkk(0); only meaningful for q == k
 * @return the integral over [0, bMax]
 */
inline double lcd_delta_bkk_zero(size_t k, double bMax);

inline double lcd_delta_bkk_quadrature(size_t k, size_t q, double bMax,
                                       double c, bool reduced = false) {
  assert(bMax > 0.00);
  assert(c >= 0.00);
  assert(!reduced || q == k);  // no reduction is defined for dBkk1

  // one shared helper: construction allocates a workspace per OpenMP thread
  // and integrate() is const and dispatches on omp_get_thread_num(), so this
  // is safe to share exactly as GMToDiracBaseOptimizationParams::gaussKronrod
  // is shared across the existing parallel gradient loop.
  static const GslQuadratureAdaptiveGaussKronrod quadrature;

  // The unreduced q == k integrand grows like b/2^k, so its integral is
  // O(bMax^2) while the integrand spans many orders of magnitude across
  // [0, bMax]; GSL_EROUND out of gsl_integration_qag is reachable that way
  // (measured at bMax = 1e5, k >= 5), and with GSL's default error handler
  // that aborts. Route it through the reduced integrand, which decays, and
  // add the exact constant back: dBkk(c) = [dBkk(c) - dBkk(0)] + dBkk(0).
  // Mathematically identical, better conditioned, and abort-free.
  if (!reduced && q == k)
    return lcd_delta_bkk_quadrature(k, k, bMax, c, true) +
           lcd_delta_bkk_zero(k, bMax);

  LcdDeltaBIntegrandParams params =
      LcdDeltaBIntegrandParams{k, q, c, reduced};
  double result = 0.00;
  double abserr = 0.00;
  quadrature.integrate(lcd_delta_b_integrand, &params, 0.00, bMax, &result,
                       &abserr);

  return result;
}

/**
 * @brief dB_{0,d}(c) = B_{0,d}(bMax, c) - B_{0,d}(0, c)
 *
 * The two halves of the difference each diverge as c -> 0; only the
 * difference is finite, so every term is formed as a difference directly.
 *
 * Only the d == 0 branch carries an O(u) term, and it is the only place the
 * reduced form differs: passing eULeading = expm1(-c/(2u)) instead of
 * exp(-c/(2u)) subtracts u/4 exactly, leaving an O(c) quantity. The d >= 2
 * branch still needs the full eU either way, which is why the two are
 * separate parameters.
 *
 * @param d index of the base case
 * @param u 1 + 2 * bMax^2
 * @param eULeading exp(-c / (2u)), or expm1(-c / (2u)) for the reduced form
 * @param eU exp(-c / (2u))
 * @param e0 exp(-c / 2)
 * @param dEi Ei(-c / (2u)) - Ei(-c / 2)
 * @param c squared norm of the sample, must be > 0
 * @return dB_{0,d}(c), less u/4 for d == 0 if eULeading is the expm1 form
 */
inline double lcd_delta_b0(size_t d, double u, double eULeading, double eU,
                           double e0, double dEi, double c) {
  assert(c > 0.00);

  if (d == 0) return 0.25 * u * eULeading - 0.25 * e0 + 0.125 * c * dEi;
  if (d == 1) return -0.25 * dEi;

  // sum_{j=2..d} coef(d,j) * (eU / (c^(d-j+1) u^(j-2)) - e0 / c^(d-j+1)),
  // coef(d,j) = (d-2)! * 2^(d-j-1) / (j-2)!
  //
  // Walking j downwards from d turns the factorials into a running product:
  // coef(d,j-1) = coef(d,j) * (j-2) * 2, starting from coef(d,d) = 1/2.
  double coefficient = 0.50;
  double cPower = c;      // c^(d-j+1), j = d
  double uPower = 1.00;   // u^(j-2),   j = d ... built downwards below
  for (size_t j = d; j > 2; --j) uPower *= u;

  double sum = 0.00;
  for (size_t j = d;; --j) {
    sum += coefficient * (eU / (cPower * uPower) - e0 / cPower);
    if (j == 2) break;
    coefficient *= static_cast<double>(j - 2) * 2.00;
    cPower *= c;
    uPower /= u;
  }

  return sum;
}

/**
 * @brief dBkk(k, c) = B_{k,k}(bMax, c) - B_{k,k}(0, c), the D2 term
 *
 * D2 = pow(2, k) * sum_i w_i * lcd_delta_bkk(k, bMax, c_i), with
 * c_i = ||s_i||^2.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @param c squared norm of the sample, must be >= 0
 * @return dBkk(k, c)
 * @note standard-normal target and even N only
 */
inline double lcd_delta_bkk(size_t k, double bMax, double c);

/**
 * @brief dBkk1(k, c) = B_{k,k+1}(bMax, c) - B_{k,k+1}(0, c), the D2 gradient
 *
 * The attraction part of the gradient is
 * pow(2, k+1) * w_q * s_q * lcd_delta_bkk1(k, bMax, c_q).
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @param c squared norm of the sample, must be >= 0
 * @return dBkk1(k, c)
 * @note standard-normal target and even N only
 */
inline double lcd_delta_bkk1(size_t k, double bMax, double c);

/**
 * @brief closed form of dBkk / dBkk1, without the small-c guard
 *
 * Exposed separately so the threshold calibration test can measure exactly
 * where this loses accuracy. Production callers want lcd_delta_bkk() /
 * lcd_delta_bkk1(), which apply the guard.
 *
 * @param k half the dimension, N = 2k
 * @param bMax upper integration bound
 * @param c squared norm of the sample, must be > 0
 * @param shift 0 for dBkk (uses dB_{0,j}), 1 for dBkk1 (uses dB_{0,j+1})
 * @return pow(2,-k) * sum_{j=0..k} (-1)^j * binom(k,j) * dB_{0,j+shift}(c)
 */
inline double lcd_delta_bkk_closed(size_t k, double bMax, double c,
                                   size_t shift, bool reduced = false) {
  assert(k >= 1);
  assert(bMax > 0.00);
  assert(c > 0.00);
  assert(!reduced || shift == 0);  // no reduction is defined for dBkk1

  const double u = 1.00 + 2.00 * bMax * bMax;
  const double exponent = -c / (2.00 * u);
  const double eU = std::exp(exponent);
  const double eULeading = reduced ? std::expm1(exponent) : eU;
  const double e0 = std::exp(-c / 2.00);
  const double dEi = lcd_ei_safe(exponent) - lcd_ei_safe(-c / 2.00);

  double sum = 0.00;
  double binomial = 1.00;  // binom(k, j)
  for (size_t j = 0; j <= k; ++j) {
    const double term =
        lcd_delta_b0(j + shift, u, eULeading, eU, e0, dEi, c);
    sum += (j % 2 == 0) ? binomial * term : -binomial * term;
    binomial *= static_cast<double>(k - j) / static_cast<double>(j + 1);
  }

  return std::ldexp(sum, -static_cast<int>(k));
}

/**
 * @brief dBkk(k, 0) = integral_0^bMax b^(2k+1) / (1 + 2b^2)^k db
 *
 * The c = 0 value of the attraction term, in elementary closed form. With
 * u = 1 + 2b^2 the substitution gives
 *
 *   Z = 2^-(k+2) * integral_1^U (1 - 1/u)^k du
 *     = 2^-(k+2) * [ (U-1) - k*ln(U)
 *                    + sum_{j=2..k} (-1)^j C(k,j) (U^(1-j) - 1)/(1-j) ]
 *
 * This is the constant lcd_delta_bkk_reduced() removes. It is O(bMax^2) by
 * construction - that is the whole point - so it belongs in the reported
 * offset, never inside the optimizer's objective.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @return dBkk(k, 0)
 */
inline double lcd_delta_bkk_zero(size_t k, double bMax) {
  assert(k >= 1);
  assert(bMax > 0.00);

  const double u = 1.00 + 2.00 * bMax * bMax;

  double sum = (u - 1.00) - static_cast<double>(k) * std::log(u);

  double binomial = static_cast<double>(k);  // binom(k, 1), stepped up below
  for (size_t j = 2; j <= k; ++j) {
    binomial *= static_cast<double>(k - j + 1) / static_cast<double>(j);
    const double dj = static_cast<double>(j);
    const double term =
        binomial * (std::pow(u, 1.00 - dj) - 1.00) / (1.00 - dj);
    sum += (j % 2 == 0) ? term : -term;
  }

  return std::ldexp(sum, -static_cast<int>(k) - 2);
}

/**
 * @brief lcd_delta_bkk_zero(k, bMax) - pow(2, -k) * U / 4
 *
 * The same expression with (U - 1) replaced by -1, i.e. with the O(bMax^2)
 * leading term removed. The result is O(ln U), so subtracting it from the
 * expm1 form of the closed dBkk introduces no cancellation.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @return dBkk(k, 0) - pow(2, -k) * (1 + 2 bMax^2) / 4
 */
inline double lcd_delta_bkk_zero_minus_leading(size_t k, double bMax) {
  assert(k >= 1);
  assert(bMax > 0.00);

  const double u = 1.00 + 2.00 * bMax * bMax;

  double sum = -1.00 - static_cast<double>(k) * std::log(u);

  double binomial = static_cast<double>(k);
  for (size_t j = 2; j <= k; ++j) {
    binomial *= static_cast<double>(k - j + 1) / static_cast<double>(j);
    const double dj = static_cast<double>(j);
    const double term =
        binomial * (std::pow(u, 1.00 - dj) - 1.00) / (1.00 - dj);
    sum += (j % 2 == 0) ? term : -term;
  }

  return std::ldexp(sum, -static_cast<int>(k) - 2);
}

/**
 * @brief dBkk(k, c) - dBkk(k, 0), the cancellation-free attraction term
 *
 * Equal to integral_0^bMax b^(2k+1)/(1+2b^2)^k * expm1(-c/(2(1+2b^2))) db.
 *
 * This is what the optimizer's D2 must use. lcd_delta_bkk() is O(bMax^2) for
 * every sample while the assembled objective is O(1), so summing the
 * unreduced form and cancelling against D1 and D3 afterwards destroys every
 * significant digit of the objective once bMax is large: the absolute noise
 * floor is bMax^2 * eps, which at bMax = 1e4 is ~2e-8 against an optimum of
 * ~1e-4. The reduced form is O(c * ln U) and never builds that intermediate.
 *
 * The constant removed per sample is lcd_delta_bkk_zero(k, bMax); D2 as a
 * whole loses pow(2, k) * (sum_i w_i) * lcd_delta_bkk_zero(k, bMax).
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @param c squared norm of the sample, must be >= 0
 * @return dBkk(k, c) - dBkk(k, 0)
 * @note standard-normal target and even N only
 */
inline double lcd_delta_bkk_reduced(size_t k, double bMax, double c) {
  assert(k >= 1);
  assert(bMax > 0.00);
  assert(c >= 0.00);

  if (c <= 0.00) return 0.00;  // dBkk(0) - dBkk(0)

  // below the calibrated threshold the closed form cancels; the quadrature
  // fallback integrates the expm1 integrand and is reduced by construction
  // if (c < lcd_small_c_threshold(k))
  //   return lcd_delta_bkk_quadrature(k, k, bMax, c, true);

  // the expm1 form has already dropped pow(2,-k) * U/4; take off what is left
  // of dBkk(k, 0), which is only O(ln U)
  return lcd_delta_bkk_closed(k, bMax, c, 0, true) -
         lcd_delta_bkk_zero_minus_leading(k, bMax);
}

/**
 * @brief shared body of lcd_delta_bkk / lcd_delta_bkk1
 *
 * @param k half the dimension, N = 2k
 * @param bMax upper integration bound
 * @param c squared norm of the sample
 * @param shift 0 for dBkk (uses dB_{0,j}), 1 for dBkk1 (uses dB_{0,j+1})
 * @return dBkk (shift 0) or dBkk1 (shift 1)
 */
inline double lcd_delta_bkk_impl(size_t k, double bMax, double c,
                                 size_t shift) {
  assert(k >= 1);
  assert(bMax > 0.00);
  assert(c >= 0.00);

  // c <= 0 (a sample exactly at the mean) has no closed form at all: every
  // dB_{0,d>=2} term is a 0/0. Route it, and everything below the calibrated
  // threshold, straight to quadrature.
  // if (c <= 0.00 || c < lcd_small_c_threshold(k))
  //   return lcd_delta_bkk_quadrature(k, k + shift, bMax, c);

  return lcd_delta_bkk_closed(k, bMax, c, shift);
}

inline double lcd_delta_bkk(size_t k, double bMax, double c) {
  return lcd_delta_bkk_impl(k, bMax, c, 0);
}

inline double lcd_delta_bkk1(size_t k, double bMax, double c) {
  return lcd_delta_bkk_impl(k, bMax, c, 1);
}

/**
 * @brief integrand of the non-isotropic D1, mirroring calculateP1
 */
struct LcdD1IntegrandParams {
  const double* covDiagStdDev;  ///< per-dimension standard deviations
  size_t N;                     ///< dimension
};

/**
 * @brief b^(N+1) * prod_k (sigma_k^2 + b^2)^(-1/2)
 *
 * @param b integration variable
 * @param params LcdD1IntegrandParams
 * @return value of the integrand at b
 */
inline double lcd_d1_integrand(double b, void* params) {
  const LcdD1IntegrandParams* p =
      static_cast<const LcdD1IntegrandParams*>(params);

  double value = b;
  for (size_t k = 0; k < p->N; ++k) {
    const double sigma = p->covDiagStdDev[k];
    value *= b / std::sqrt(sigma * sigma + b * b);
  }

  return value;
}

/**
 * @brief the D1 constant that gm_to_dirac_short omits from its reported
 * distance
 *
 * Add this to gm_to_dirac_short::modified_van_mises_distance_sq(...) to
 * obtain the true mCvM distance (in the library's pi^(N/2) units). The
 * quadrature path never computes it: its member called D1 is the x = 0
 * baseline of the D2 integrand that calculateP2 subtracts via expm1, which is
 * a different integral (it carries 2*b^2, not b^2).
 *
 * Closed form for isotropic covariance, numerical fallback otherwise.
 *
 * @param covDiagStdDev per-dimension standard deviations, length N
 * @param N dimension, must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @return integral_0^bMax b^(N+1) * prod_k (sigma_k^2 + b^2)^(-1/2) db
 */
inline double lcd_d1_offset(const double* covDiagStdDev, size_t N,
                            double bMax) {
  assert(covDiagStdDev != nullptr);
  assert(N >= 1);
  assert(bMax > 0.00);

  const double sigma = covDiagStdDev[0];
  bool isotropic = sigma > 0.00;
  for (size_t k = 1; k < N && isotropic; ++k)
    isotropic = (covDiagStdDev[k] == sigma);

  // isotropic: b = sigma * t reduces the integral to sigma^2 * A_N(bMax/sigma)
  if (isotropic) return sigma * sigma * lcd_a_n(N, bMax / sigma);

  // no closed form for unequal sigma_k: integrate once, as calculateD1 does
  static const GslQuadratureAdaptiveGaussKronrod quadrature;

  LcdD1IntegrandParams params = LcdD1IntegrandParams{covDiagStdDev, N};
  double result = 0.00;
  double abserr = 0.00;
  quadrature.integrate(lcd_d1_integrand, &params, 0.00, bMax, &result, &abserr);

  return result;
}

/**
 * @brief the constant that turns the reduced objective back into the true
 * distance
 *
 * The optimizer minimises a reduced objective in which every O(bMax^2) piece
 * has been removed analytically:
 *
 *   f_reduced = -2 * pow(2,k) * sum_i w_i * lcd_delta_bkk_reduced(k, bMax, c_i)
 *             + sum_ij w_i w_j * lcd_c_repulsion_reduced(bMax, T_ij)
 *
 * and the true distance is D = f_reduced + K, with
 *
 *   K = A_N(bMax)
 *     - pow(2, k+1) * W * lcd_delta_bkk_zero(k, bMax)
 *     + W^2 * bMax^2 / 2,          W = sum_i w_i
 *
 * collecting the D1 term, the per-sample attraction constant and the
 * per-pair repulsion constant respectively.
 *
 * @param N dimension, must be even and >= 2
 * @param bMax upper integration bound, must be > 0
 * @param weightSum sum of the Dirac weights, normally 1
 * @return K
 *
 * @note For N > 2 this is itself a difference of O(bMax^2) terms, so the
 * REPORTED distance carries an absolute offset error of order bMax^2 * eps
 * (about 1.3e-8 at bMax = 1e4). That is a property of the constant alone: it
 * adds no x-dependent noise, cancels exactly in every line-search comparison,
 * and therefore does not affect optimisation at all. N = 2 uses an exact
 * cancellation-free rearrangement and has no such error.
 */
inline double lcd_reported_offset(size_t N, double bMax, double weightSum) {
  assert(N >= 2);
  assert(N % 2 == 0);
  assert(bMax > 0.00);

  const size_t k = N / 2;
  const double bMaxSqrd = bMax * bMax;

  if (N == 2) {
    // K = W^2 bMax^2/2 + bMax^2/2 - W bMax^2 - ln(1+bMax^2)/2
    //     + W ln(1+2bMax^2)/2
    //
    // Two rearrangements make this exact rather than merely correct:
    //  - the three bMax^2 terms collapse to (1-W)^2 bMax^2 / 2, which
    //    vanishes identically at W = 1 instead of cancelling numerically;
    //  - W ln(1+2b^2)/2 - ln(1+b^2)/2 is split as
    //    W/2 * ln((1+2b^2)/(1+b^2)) + (W-1)/2 * ln(1+b^2), whose first factor
    //    is log1p(b^2/(1+b^2)). Written as a difference of logs it would lose
    //    two digits by bMax = 1e8, both logs being ~37 with a difference of
    //    ~0.35; written this way it is good to the last bit, and the second
    //    term vanishes identically at W = 1.
    const double weightDefect = 1.00 - weightSum;
    return 0.50 * bMaxSqrd * weightDefect * weightDefect +
           0.50 * weightSum * std::log1p(bMaxSqrd / (1.00 + bMaxSqrd)) -
           0.50 * weightDefect * std::log1p(bMaxSqrd);
  }

  return lcd_a_n(N, bMax) -
         std::ldexp(weightSum * lcd_delta_bkk_zero(k, bMax),
                    static_cast<int>(k) + 1) +
         0.50 * weightSum * weightSum * bMaxSqrd;
}

#endif  // LCD_EVEN_CLOSED_FORM_H
