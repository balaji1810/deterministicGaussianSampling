#ifndef GM_TO_DIRAC_EVEN_CLOSED_FORM_TPP
#define GM_TO_DIRAC_EVEN_CLOSED_FORM_TPP

/**
 * @brief check the preconditions of the even-N closed-form path
 *
 * @param L number of Dirac components
 * @param N dimension
 * @param bMax upper bound of the b-integral
 * @return true if the closed form applies
 */
template <typename T>
inline bool gm_to_dirac_even_closed_form<T>::preconditionsHold(size_t L,
                                                               size_t N,
                                                               double bMax) {
  // odd N has no closed form here; fail loudly rather than silently falling
  // back to the quadrature path
  assert(N % 2 == 0);
  assert(N >= 2);
  assert(bMax > 0.00);
  assert(L >= 1);

  return N >= 2 && N % 2 == 0 && bMax > 0.00 && L >= 1;
}

/**
 * @brief attraction term: contributes -2 * D2 to f and its gradient to grad
 *
 * f += -2 * pow(2,k) * sum_i w_i * dBkkReduced(k, bMax, c_i)
 *
 * grad += pow(2,k+1) * w_q * s_q * dBkk1(k, bMax, c_q)
 *
 * f uses the REDUCED attraction term: the unreduced dBkk is O(bMax^2) per
 * sample while the assembled objective is O(1), so accumulating it and
 * cancelling against D1 and D3 afterwards leaves the objective with an
 * absolute noise floor of bMax^2 * eps. The removed constant lives in
 * GMToDiracEvenOptimizationParams::reportedOffset. The gradient needs no such
 * treatment: its two terms are O(ln bMax^2) and cancel to O(1), so it is
 * well conditioned as written.
 *
 * The gradient already carries the -2 of the -2*D2 term, matching the sign
 * convention of gm_to_dirac_short::modified_van_mises_distance_sq_derivative.
 *
 * @param x sample locations (L * N)
 * @param params optimization parameters, caches must be up to date
 * @param f objective accumulator, may be nullptr
 * @param grad gradient accumulator, may be nullptr
 */
template <typename T>
inline void gm_to_dirac_even_closed_form<T>::calculateD2(
    const gsl_vector* x, GMToDiracEvenOptimizationParams* params, double* f,
    gsl_vector* grad) {
  const size_t L = params->L;
  const size_t N = params->N;
  const size_t k = params->k;
  const double bMax = params->bMax;
  const gsl_vector* wX = params->wX;

  if (f) {
    double sum = 0.00;
    for (size_t i = 0; i < L; ++i)
      sum += wX->data[i] * lcd_delta_bkk_reduced(k, bMax, params->cSqrdNorm[i]);

    *f += -2.00 * std::ldexp(sum, static_cast<int>(k));
  }

  if (grad) {
    for (size_t q = 0; q < L; ++q) {
      const double factor =
          std::ldexp(wX->data[q] * lcd_delta_bkk1(k, bMax, params->cSqrdNorm[q]),
                     static_cast<int>(k) + 1);
      for (size_t d = 0; d < N; ++d)
        grad->data[q * N + d] += factor * x->data[q * N + d];
    }
  }
}

/**
 * @brief repulsion term: contributes D3 to f and its gradient to grad
 *
 * f    += sum_{i,j} w_i * w_j * (C(bMax, T_ij) - bMax^2/2)
 *
 * grad += 0.5 * w_q * sum_{i != q} w_i * (s_q - s_i) * Ei(-T_qi/(4 bMax^2))
 *
 * f uses the REDUCED kernel, which has the per-pair bMax^2/2 removed
 * analytically; the total removed, (sum_i w_i)^2 * bMax^2 / 2, lives in
 * GMToDiracEvenOptimizationParams::reportedOffset. Coincident samples - which
 * includes every i == j pair - therefore contribute exactly nothing now,
 * where the unreduced kernel gave them bMax^2/2 each.
 *
 * They contribute nothing to the gradient either, exactly as
 * gm_to_dirac_short::calculateD3 skips them.
 *
 * @param x sample locations (L * N)
 * @param params optimization parameters, caches must be up to date
 * @param f objective accumulator, may be nullptr
 * @param grad gradient accumulator, may be nullptr
 */
template <typename T>
inline void gm_to_dirac_even_closed_form<T>::calculateD3(
    const gsl_vector* x, GMToDiracEvenOptimizationParams* params, double* f,
    gsl_vector* grad) {
  const size_t L = params->L;
  const size_t N = params->N;
  const double bMax = params->bMax;
  const gsl_vector* wX = params->wX;

  const double fourBMaxSqrd = 4.00 * bMax * bMax;
  double d3 = 0.00;

  for (size_t i = 0; i < L; ++i) {
    const double wXi = wX->data[i];

    for (size_t j = 0; j < L; ++j) {
      const double wXiwXj = wXi * wX->data[j];
      const double localDistSq = params->distanceSq(i, j);

      // coincident samples, including every i == j pair: the reduced kernel
      // is exactly 0 there and the gradient contribution vanishes
      if (localDistSq <= 0.00) continue;

      // f and grad share the same Ei, as the log is shared in
      // gm_to_dirac_short::calculateD3; this is the whole cost of D3
      const double z = -localDistSq / fourBMaxSqrd;
      const double ei = lcd_ei_safe(z);

      // matches lcd_c_repulsion_reduced(bMax, localDistSq), reusing ei
      if (f)
        d3 += wXiwXj * (0.50 * bMax * bMax * std::expm1(z) +
                        0.125 * localDistSq * ei);

      if (!grad) continue;

      const double constFactor = 0.50 * wXiwXj * ei;
      for (size_t d = 0; d < N; ++d)
        grad->data[i * N + d] +=
            constFactor * (x->data[i * N + d] - x->data[j * N + d]);
    }
  }

  if (f) *f += d3;
}

/**
 * @brief remove the weighted mean from the sample set, in place
 *
 * @param x sample locations (L * N)
 * @param wX weights
 * @param L number of Dirac components
 * @param N dimension
 */
template <typename T>
inline void gm_to_dirac_even_closed_form<T>::correctMean(gsl_vector* x,
                                                         const gsl_vector* wX,
                                                         size_t L, size_t N) {
  std::vector<double> mean(N, 0.00);
  for (size_t i = 0; i < L; ++i) {
    const double wXi = wX->data[i];
    for (size_t d = 0; d < N; ++d) mean[d] += wXi * x->data[i * N + d];
  }
  for (size_t i = 0; i < L; ++i) {
    for (size_t d = 0; d < N; ++d) x->data[i * N + d] -= mean[d];
  }
}

#endif  // GM_TO_DIRAC_EVEN_CLOSED_FORM_TPP
