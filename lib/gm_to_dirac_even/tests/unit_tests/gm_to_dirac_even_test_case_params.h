#ifndef GM_TO_DIRAC_EVEN_TEST_CASE_PARAMS_H
#define GM_TO_DIRAC_EVEN_TEST_CASE_PARAMS_H

#include <vector>

struct GmToDiracEvenTestCaseParams {
  std::vector<double> x;         // L x N sample locations
  std::vector<double> grad;      // L x N expected gradient
  double distance;               // expected mCvM distance
  size_t L;
  size_t N;
  double bMax;
};

#endif  // GM_TO_DIRAC_EVEN_TEST_CASE_PARAMS_H
