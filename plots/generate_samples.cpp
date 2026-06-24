// Reproducibility: the underlying library seeds GSL via gsl_rng_env_setup(),
// which reads the GSL_RNG_SEED environment variable. Set GSL_RNG_SEED to a
// fixed value (e.g. 42) before running for identical output across runs and
// machines. If the variable is unset we default it to 42 so that manual runs
// are reproducible too.

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gm_to_dirac_short.h"

int main(int argc, char** argv) {
  // Default GSL_RNG_SEED to 42 if unset, for reproducible runs
  if (std::getenv("GSL_RNG_SEED") == nullptr) {
    #ifdef _WIN32
      _putenv_s("GSL_RNG_SEED", "42");
    #else
      setenv("GSL_RNG_SEED", "42", 1);
    #endif
  }

  // Minimum is: prog + N + L + out_csv + (at least one) sigma.
  if (argc < 5) {
    std::cerr << "usage: " << argv[0]
              << " <N> <L> <out_csv> <sigma_1> ... <sigma_N> [bMax]\n";
    return 1;
  }

  const size_t N = static_cast<size_t>(std::stoi(argv[1]));
  const size_t L = static_cast<size_t>(std::stoi(argv[2]));
  const char* outPath = argv[3];

  // argv: [0]=prog [1]=N [2]=L [3]=out_csv [4..4+N-1]=sigmas [optional]=bMax
  const int sigmaStart = 4;
  const int argcNoBMax = sigmaStart + static_cast<int>(N);
  if (N == 0 || L == 0 || (argc != argcNoBMax && argc != argcNoBMax + 1)) {
    std::cerr << "error: expected N=" << N << " sigma value(s) (got "
              << (argc - sigmaStart) << "); N and L must be > 0\n";
    return 1;
  }

  // Per-axis standard deviations (sigma).
  std::vector<double> covDiag(N);
  for (size_t k = 0; k < N; ++k) {
    covDiag[k] = std::strtod(argv[sigmaStart + static_cast<int>(k)], nullptr);
  }

  ApproximateOptions options;
  if (argc == argcNoBMax + 1) {
    options.bMax =
        static_cast<size_t>(std::stoi(argv[argcNoBMax]));
  }

  std::vector<double> x(L * N, 0.0);
  GslminimizerResult result;
  gm_to_dirac_short<double> approx;

  const bool ok = approx.approximate(covDiag.data(), L, N, x.data(), nullptr, &result, options);
  if (!ok) {
    std::cerr << "error: LCD approximation did not converge for N=" << N
              << ", L=" << L << "\n";
    return 1;
  }

  std::ofstream out(outPath);
  if (!out) {
    std::cerr << "error: cannot open '" << outPath << "' for writing\n";
    return 1;
  }
  out << std::setprecision(17);
  for (size_t i = 0; i < L; ++i) {
    for (size_t k = 0; k < N; ++k) {
      if (k != 0) out << ',';
      out << x[i * N + k];
    }
    out << '\n';
  }
  return 0;
}
