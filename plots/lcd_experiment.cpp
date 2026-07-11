// Experiment driver for characterizing the LCD Gaussian sampler.
//
// Unlike generate_samples (kept minimal for the docs pipeline), this tool
// exposes the optimizer knobs (seed, tolerances, bMax, warm starts) and
// reports convergence diagnostics so that sweeps over L / bMax / seeds can
// be scripted. It only links against the library; no library code changes.
//
// Output: one "key=value" diagnostics line on stdout (prefixed RESULT),
// plus the sample CSV written to --out.

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gsl/gsl_errno.h>

#include "gm_to_dirac_short.h"

namespace {

void usage(const char* prog) {
  std::cerr
      << "usage: " << prog << " [options]\n"
      << "  --N <dim>            dimension (default 2)\n"
      << "  --L <count>          number of samples (required)\n"
      << "  --sigma <s1,s2,...>  per-axis std devs (default all 1)\n"
      << "  --bMax <int>         kernel-width upper limit (default 100)\n"
      << "  --seed <int>         GSL_RNG_SEED for the random init (default 42)\n"
      << "  --ftolRel <v>        relative f tolerance (default 1e-10)\n"
      << "  --ftolAbs <v>        absolute f tolerance (default 0)\n"
      << "  --gtol <v>           gradient-norm tolerance (default 1e-6)\n"
      << "  --maxIter <n>        max iterations (default 10000)\n"
      << "  --init <csv>         warm-start point set (skips random init)\n"
      << "  --out <csv>          output CSV path (default samples.csv)\n"
      << "  --evalBMax <int>     also evaluate distance at this bMax (0=off)\n"
      << "  --verbose            verbose optimizer output\n";
}

bool parseSigmaList(const std::string& s, std::vector<double>* out) {
  out->clear();
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) return false;
    out->push_back(std::strtod(tok.c_str(), nullptr));
  }
  return !out->empty();
}

bool readCsv(const char* path, size_t L, size_t N, std::vector<double>* x) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  size_t i = 0;
  while (std::getline(in, line) && i < L) {
    std::stringstream ss(line);
    std::string tok;
    size_t k = 0;
    while (std::getline(ss, tok, ',') && k < N) {
      (*x)[i * N + k] = std::strtod(tok.c_str(), nullptr);
      ++k;
    }
    if (k != N) return false;
    ++i;
  }
  return i == L;
}

}  // namespace

int main(int argc, char** argv) {
  size_t N = 2;
  size_t L = 0;
  std::vector<double> sigma;
  size_t bMax = 100;
  long seed = 42;
  size_t evalBMax = 0;
  const char* outPath = "samples.csv";
  const char* initPath = nullptr;
  ApproximateOptions options;

  for (int a = 1; a < argc; ++a) {
    auto need = [&](const char* flag) -> const char* {
      if (a + 1 >= argc) {
        std::cerr << "error: missing value for " << flag << "\n";
        std::exit(1);
      }
      return argv[++a];
    };
    if (!std::strcmp(argv[a], "--N")) {
      N = static_cast<size_t>(std::stoi(need("--N")));
    } else if (!std::strcmp(argv[a], "--L")) {
      L = static_cast<size_t>(std::stoi(need("--L")));
    } else if (!std::strcmp(argv[a], "--sigma")) {
      if (!parseSigmaList(need("--sigma"), &sigma)) {
        std::cerr << "error: bad --sigma list\n";
        return 1;
      }
    } else if (!std::strcmp(argv[a], "--bMax")) {
      bMax = static_cast<size_t>(std::stoi(need("--bMax")));
    } else if (!std::strcmp(argv[a], "--seed")) {
      seed = std::stol(need("--seed"));
    } else if (!std::strcmp(argv[a], "--ftolRel")) {
      options.ftolRel = std::strtod(need("--ftolRel"), nullptr);
    } else if (!std::strcmp(argv[a], "--ftolAbs")) {
      options.ftolAbs = std::strtod(need("--ftolAbs"), nullptr);
    } else if (!std::strcmp(argv[a], "--gtol")) {
      options.gtol = std::strtod(need("--gtol"), nullptr);
    } else if (!std::strcmp(argv[a], "--maxIter")) {
      options.maxIterations = static_cast<size_t>(std::stol(need("--maxIter")));
    } else if (!std::strcmp(argv[a], "--init")) {
      initPath = need("--init");
    } else if (!std::strcmp(argv[a], "--out")) {
      outPath = need("--out");
    } else if (!std::strcmp(argv[a], "--evalBMax")) {
      evalBMax = static_cast<size_t>(std::stoi(need("--evalBMax")));
    } else if (!std::strcmp(argv[a], "--verbose")) {
      options.verbose = true;
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  if (L == 0) {
    usage(argv[0]);
    return 1;
  }
  if (sigma.empty()) sigma.assign(N, 1.0);
  if (sigma.size() != N) {
    std::cerr << "error: --sigma needs " << N << " values\n";
    return 1;
  }

  // The library seeds its RNG from GSL_RNG_SEED via gsl_rng_env_setup().
  {
    std::string seedStr = std::to_string(seed);
#ifdef _WIN32
    _putenv_s("GSL_RNG_SEED", seedStr.c_str());
#else
    setenv("GSL_RNG_SEED", seedStr.c_str(), 1);
#endif
  }

  options.bMax = bMax;

  std::vector<double> x(L * N, 0.0);
  if (initPath) {
    if (!readCsv(initPath, L, N, &x)) {
      std::cerr << "error: could not read " << L << "x" << N
                << " values from '" << initPath << "'\n";
      return 1;
    }
    options.initialX = true;
  }

  // GSL's default error handler aborts the process on e.g. quadrature
  // round-off failures (seen for bMax >> sigma); report status instead.
  gsl_set_error_handler_off();

  gm_to_dirac_short<double> approx;
  GslminimizerResult result;
  const bool ok = approx.approximate(sigma.data(), L, N, x.data(), nullptr,
                                     &result, options);

  double dist = 0.0;
  approx.modified_van_mises_distance_sq(sigma.data(), &dist, L, N, bMax,
                                        x.data(), nullptr);
  double distEval = 0.0;
  if (evalBMax > 0) {
    approx.modified_van_mises_distance_sq(sigma.data(), &distEval, L, N,
                                          evalBMax, x.data(), nullptr);
  }

  std::ofstream out(outPath);
  if (!out) {
    std::cerr << "error: cannot open '" << outPath << "'\n";
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

  std::cout << std::setprecision(17) << "RESULT ok=" << (ok ? 1 : 0)
            << " L=" << L << " N=" << N << " bMax=" << bMax
            << " seed=" << seed << " iterations=" << result.iterations
            << " elapsedMicro=" << result.elapsedTimeMicro
            << " dist=" << dist << " distEval=" << distEval
            << " lastFtolAbs=" << result.lastFtolAbs
            << " lastFtolRel=" << result.lastFtolRel
            << " lastGtol=" << result.lastGtol << "\n";
  return 0;
}
