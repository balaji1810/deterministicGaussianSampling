#include <gsl/gsl_vector.h>
#include <gtest/gtest.h>

#include <vector>

#include "gm_to_dirac_even_closed_form.h"
#include "gm_to_dirac_even_test_case_params.h"
#include "gradient_van_mises_distance.h"
#include "gsl_utils_allocation.h"
#include "gtest_compare_vec.h"

class gm_to_dirac_even_closed_form_test
    : public ::testing::TestWithParam<GmToDiracEvenTestCaseParams> {
 protected:
  struct DistanceParams {
    size_t L;
    size_t N;
    double bMax;
  };

  // adapts the public API to the signature gradient_van_mises_distance wants
  static double distance(const gsl_vector* x, void* params) {
    DistanceParams* p = static_cast<DistanceParams*>(params);
    std::vector<double> xCopy(x->data, x->data + p->L * p->N);
    double d = 0.00;
    gm_to_dirac_even_closed_form<double> gm2dEven;
    gm2dEven.modified_van_mises_distance_sq(&d, p->L, p->N, p->bMax,
                                            xCopy.data(), nullptr);
    return d;
  }

  void SetUp() override {
    GmToDiracEvenTestCaseParams p = GetParam();
    x = create_gsl_vector(p.x);
    expectedGrad = create_gsl_vector(p.grad);
    numericalGrad = gsl_vector_alloc(p.L * p.N);
    analyticalGrad = gsl_vector_alloc(p.L * p.N);
    if (x == nullptr || expectedGrad == nullptr || numericalGrad == nullptr ||
        analyticalGrad == nullptr) {
      GTEST_SKIP() << "Failed to allocate memory for vectors.";
    }
  }

  void TearDown() override {
    gsl_vector_free(x);
    gsl_vector_free(expectedGrad);
    gsl_vector_free(numericalGrad);
    gsl_vector_free(analyticalGrad);
  }

  gsl_vector* x;
  gsl_vector* expectedGrad;
  gsl_vector* numericalGrad;
  gsl_vector* analyticalGrad;

  const double eps = 1e-5;

 private:
  static gradient_van_mises_distance gradVanMisesDistance;

 protected:
  static void gsl_numerical_gradient(const gsl_vector* x,
                                     DistanceParams* params,
                                     gsl_vector* grad) {
    gradVanMisesDistance.multivariativeGradient(x, grad, distance, params);
  }
};

gradient_van_mises_distance
    gm_to_dirac_even_closed_form_test::gradVanMisesDistance;

TEST_P(gm_to_dirac_even_closed_form_test, distance_matches_reference) {
  GmToDiracEvenTestCaseParams p = GetParam();

  double d = 0.00;
  auto gm2dEven = gm_to_dirac_even_closed_form<double>();
  gm2dEven.modified_van_mises_distance_sq(&d, p.L, p.N, p.bMax, x->data,
                                          nullptr);

  ASSERT_NEAR(d, p.distance, 1e-10 * std::abs(p.distance));
}

TEST_P(gm_to_dirac_even_closed_form_test, gradient_matches_reference) {
  GmToDiracEvenTestCaseParams p = GetParam();

  auto gm2dEven = gm_to_dirac_even_closed_form<double>();
  gm2dEven.modified_van_mises_distance_sq_derivative(
      analyticalGrad->data, p.L, p.N, p.bMax, x->data, nullptr);

  ASSERT_TRUE(assert_gsl_vectors_close(expectedGrad, analyticalGrad, 1e-12));
}

TEST_P(gm_to_dirac_even_closed_form_test, gradient_matches_numerical) {
  GmToDiracEvenTestCaseParams p = GetParam();
  DistanceParams params = DistanceParams{p.L, p.N, p.bMax};

  gsl_numerical_gradient(x, &params, numericalGrad);

  auto gm2dEven = gm_to_dirac_even_closed_form<double>();
  gm2dEven.modified_van_mises_distance_sq_derivative(
      analyticalGrad->data, p.L, p.N, p.bMax, x->data, nullptr);

  ASSERT_TRUE(assert_gsl_vectors_close(analyticalGrad, numericalGrad, eps));
}

INSTANTIATE_TEST_SUITE_P(
    ClosedFormEvenParameterizedTest, gm_to_dirac_even_closed_form_test,
    ::testing::Values(
        GmToDiracEvenTestCaseParams{
            {0.5, -0.25, -1.0, 0.75, 0.125, 1.5, -0.375, -0.5},  // x (4x2)
            {-0.1375254944028399, 0.18941441189379193,           // grad
             -0.034822045236016896, 0.16656976823938163,
             -0.12159464735365064, 0.1251272196018498,
             -0.07491585186960052, 0.21452678821259696},
            0.22350401527273078,  // distance
            4,                    // L
            2,                    // N
            10.0                  // bMax
        },
        GmToDiracEvenTestCaseParams{
            {0.5, -0.25, 1.0, 0.0, -1.0, 0.75, -0.5, 0.25,       // x (3x4)
             0.25, 0.5, -0.125, -1.5},
            {-0.043912182070676264, 0.14677541197596747,         // grad
             0.017458740048938548, -0.18780233519392056,
             0.002788668123894278, 0.10987636921718977,
             0.06527075628141624, -0.18400997083230003,
             -0.04838234184853639, 0.1014681556621104,
             0.06913065142156252, -0.06735249372624652},
            0.38081273335856913,  // distance
            3,                    // L
            4,                    // N
            5.0                   // bMax
        },
        // one sample 1e-7 from the origin: exercises the small-c quadrature
        // fallback of lcd_delta_bkk1. grad[0] deviates from what the pure
        // closed form produces -- at c = 1e-14 that form is ~1e-3 off, and
        // this component is a near-total cancellation between the attraction
        // and repulsion terms, so the error shows in full. The value below is
        // a 60-digit reference of the raw b-integral.
        GmToDiracEvenTestCaseParams{
            {1e-7, 0.0, 1.0, 0.5, -1.0, -0.5},                   // x (3x2)
            {3.181333290937428e-08, 8.861154476846878e-09,       // grad
             -0.04132055937492485, -0.020660294115873667,
             0.04132059936625909, 0.02066028525471919},
            0.11327028835358277,  // distance
            3,                    // L
            2,                    // N
            10.0                  // bMax
        },
        GmToDiracEvenTestCaseParams{
            {0.5, -0.25, 1.0, 0.0, 0.75, -0.5,                   // x (2x6)
             -1.0, 0.75, -0.5, 0.25, -0.25, 1.0},
            {-0.06445196661810901, 0.06537354485422255,          // grad
             0.06998143603479018, 0.033147561545168035,
             0.06905985779867665, 0.06445196661810901,
             -0.05464780969886335, 0.057559638046731504,
             0.07211877978607242, 0.03023573319729985,
             0.06920695143820425, 0.05464780969886335},
            0.5464383762451819,  // distance
            2,                   // L
            6,                   // N
            3.0                  // bMax
        }));
