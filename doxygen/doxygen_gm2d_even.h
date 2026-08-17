/**
 * \page page_gm_even gm_to_dirac_even_closed_form
 *
 * Gaussian-to-Dirac approximation, quadrature-free (even dimension).
 *
 * \section gm_even_overview Overview
 *
 * gm_to_dirac_even_closed_form<T> approximates a standard normal
 * distribution by a Dirac mixture with L components in N dimensions,
 * exactly as gm_to_dirac_short_standard_normal_deviation does, but with
 * every integral over the kernel-width variable b evaluated in closed
 * form.
 *
 * The quadrature path issues L × N adaptive Gauss-Kronrod integrations
 * per gradient evaluation. This path issues none.
 *
 * It is a single-threaded implementation.
 *
 *
 * \section gm_even_preconditions Preconditions
 *
 * - N must be EVEN and >= 2. Odd N fails loudly (assert in Debug,
 *   approximate() returns false in Release); there is no silent fallback.
 * - The target must be the standard normal N(0, I), or an isotropic
 *   N(0, sigma² I) through the approximate_isotropic() overloads.
 *   General diagonal covariance with unequal sigma_k is NOT supported:
 *   the scaling reduction that makes the isotropic case work does not
 *   exist there.
 * - bMax may be ANY value > 0. Unlike the quadrature path, whose D3 uses
 *   a large-bMax logarithmic approximation, D3 here is the exact Ei form,
 *   so no "for large bMax" assumption is made anywhere.
 *
 *
 * \section gm_even_differences Differences from the quadrature path
 *
 * Two behavioural differences are worth knowing when comparing results
 * against gm_to_dirac_short_standard_normal_deviation:
 *
 * - The reported distance is the TRUE modified Cramér-von-Mises distance:
 *   it includes the sample-independent D1 term, which the quadrature path
 *   omits. To compare, add lcd_d1_offset() (equivalently lcd_a_n(N, bMax)
 *   for the standard normal) to the quadrature path's output.
 * - D3 is exact rather than logarithmically approximated. The residual
 *   between the two implementations is therefore O(1/bMax²): measured
 *   3.5e-3 at bMax = 10 falling to 3.6e-5 at bMax = 100.
 *
 *
 * \section gm_even_interface Interface
 *
 * Inherits:
 *
 * - gm_to_dirac_approx_standard_normal_distribution_i<T>
 *
 * Template parameter:
 *
 * - T ∈ {float, double}; computation is always performed in double and
 *   cast at the interface boundary.
 *
 * Provides overloads of:
 *
 * - approximate(...)
 * - modified_van_mises_distance_sq(...)
 * - modified_van_mises_distance_sq_derivative(...)
 * - approximate_isotropic(...)
 * - modified_van_mises_distance_sq_isotropic(...)
 *
 *
 * \section gm_even_parameters Parameters
 *
 * Common parameters:
 *
 * - L     → number of Dirac components
 * - N     → dimension, must be even
 * - bMax  → integration bound, any value > 0
 * - x     → initial guess and output locations (L × N)
 * - wX    → weights of the Dirac mixture (optional)
 *
 * If wX is nullptr:
 *
 * - Uniform weights are assumed
 *
 * Two families of overloads exist. Those taking a `double bMax` are the
 * native interface. Those taking a `size_t bMax` match
 * gm_to_dirac_short_standard_normal_deviation exactly, so the two classes
 * can be swapped without touching call sites; approximate() in that family
 * reads bMax from ApproximateOptions::bMax.
 *
 * \note The two families have the same arity, so an untyped integer
 * literal is ambiguous between them. Pass bMax as an explicit double
 * (10.0) or through a typed variable.
 *
 *
 * \section gm_even_input Supported Input Formats
 *
 * Three overload families are available:
 *
 * - Raw pointer interface (T*)
 * - GSL vector interface (gsl_vector / gsl_vector_float)
 * - GSL matrix interface (gsl_matrix / gsl_matrix_float)
 *
 * Memory layout:
 *
 * - x represents L × N Dirac locations, row major
 *
 *
 * \section example_gm_even_raw Example (Raw Pointer)
 *
 * \code
 * gm_to_dirac_even_closed_form<double> approx;
 *
 * bool ok = approx.approximate(
 *     L,
 *     N,        // must be even
 *     100.0,    // bMax, any value > 0
 *     x,        // initial guess / output (L × N)
 *     wX,       // weights (optional)
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section example_gm_even_distance Example (true distance)
 *
 * \code
 * gm_to_dirac_even_closed_form<double> approx;
 *
 * double distance = 0.0;
 * approx.modified_van_mises_distance_sq(&distance, L, N, 100.0, x, wX);
 * // distance includes D1; the quadrature path's value would be short by
 * // lcd_a_n(N, 100.0)
 * \endcode
 *
 *
 * \section example_gm_even_isotropic Example (isotropic sigma² I)
 *
 * \code
 * gm_to_dirac_even_closed_form<double> approx;
 *
 * const double sigma = 2.0;
 * const double bMax  = 100.0;
 *
 * // solves the standard-normal problem at bMax / sigma and scales by sigma
 * bool ok = approx.approximate_isotropic(
 *     L, N, sigma, bMax, x, wX, &result, options);
 * \endcode
 *
 *
 * \section gm_even_math Closed forms
 *
 * The distance decomposes as D = D1 - 2·D2 + D3, all expressed in units
 * of pi^(N/2) — the same normalisation the quadrature path uses, so the
 * two are directly comparable. The building blocks live in
 * lcd_even_closed_form.h:
 *
 * - lcd_a_n()          → D1, valid for all N (odd and even)
 * - lcd_delta_bkk()    → the per-sample D2 term
 * - lcd_delta_bkk1()   → the per-sample D2 gradient term
 * - lcd_c_repulsion()  → the exact per-pair D3 kernel
 * - lcd_ei_safe()      → underflow-guarded Ei
 * - lcd_d1_offset()    → the D1 constant, usable with the quadrature path
 *
 * Both D2 helpers cancel catastrophically for samples very close to the
 * mean. Below a calibrated per-k threshold (lcd_small_c_threshold()) they
 * fall back to adaptive quadrature of the raw b-integrand, which is exact
 * and, being reached only by samples near the origin, costs almost
 * nothing.
 *
 *
 * \section gm_even_conditioning Conditioning: the reduced objective
 *
 * D1, D2 and D3 are each O(bMax^2) while their combination D is O(1e-4):
 * at bMax = 1000, N = 2 the three terms are about +5.0e5, -1.0e6 and
 * +5.0e5. Assembling the objective that way gives it an absolute noise
 * floor of bMax^2 * eps — 6e-11 at bMax = 1000, 2e-8 at bMax = 1e4 —
 * against an optimum that itself shrinks with L. The gradient is immune
 * (its terms are only O(ln bMax^2)), so BFGS keeps a good descent
 * direction, but the Wolfe line search compares FUNCTION values and stops
 * being able to verify progress; the optimizer then bails on no-progress
 * far short of the gradient criterion, and sample quality degrades as
 * bMax and L grow.
 *
 * The objective therefore never forms those terms. Each carries its
 * leading part analytically instead:
 *
 * - lcd_c_repulsion_reduced()  → C(b,c) - b^2/2, via expm1
 * - lcd_delta_bkk_reduced()    → dBkk(c) - dBkk(0), via expm1
 * - lcd_delta_bkk_zero()       → the removed attraction constant, dBkk(0)
 * - lcd_reported_offset()      → K, the sum of everything removed
 *
 * so the minimizer sees an O(1) objective and the true distance is
 * recovered as f_reduced + K. This mirrors what the quadrature path
 * already does by other means: gm_to_dirac_short::calculateP2 uses expm1
 * on the D2 integrand and calculateD3 keeps the 4*bMax^2 term out in
 * constantOffset.
 *
 * The reported distance is unchanged in value. K is itself a difference of
 * O(bMax^2) terms for N > 2, so it carries an absolute error of order
 * bMax^2 * eps; that is a property of a constant, adds no x-dependent
 * noise, and does not affect optimisation. N = 2 uses an exact
 * cancellation-free rearrangement and has no such error.
 *
 *
 * \section gm_even_notes Notes
 *
 * - Single-threaded implementation
 * - Uses analytical gradients with no quadrature
 * - The objective is assembled from reduced, O(1) terms and additionally
 *   carries the constant K by default
 *   (gm_to_dirac_even_closed_form::includeD1InObjective), so that f is the
 *   true distance and ApproximateOptions::ftolRel stays meaningful
 * - Interface-compatible with
 *   gm_to_dirac_short_standard_normal_deviation
 *
 */
