#ifndef APPROXIMATE_OPTIONS_H
#define APPROXIMATE_OPTIONS_H

#include <cstddef>

struct ApproximateOptions {
  double xtolAbs = 0;            // Absolute tolerance for x
  double xtolRel = 0;            // Relative tolerance for x
  double ftolAbs = 0;            // Absolute tolerance for f
  // Relative tolerance for f. Disabled (0) by default: the objective carries
  // an x-independent offset of magnitude ~bMax^2/2, so any relative-f
  // criterion stops the optimizer long before the point configuration has
  // converged (the samples then look inhomogeneous, increasingly so for
  // larger L or bMax). Convergence is governed by gtol / line-search
  // no-progress instead. Set e.g. 1e-10 to trade quality for speed.
  double ftolRel = 0;
  double gtol = 1e-6;            // Tolerance for gradient-norm
  bool initialX = false;         // True if x should be used as initial guess
  size_t maxIterations = 10000;  // Maximum number of iterations
  bool verbose = false;          // True if verbose output is needed
  size_t bMax = 100;
};

#endif // APPROXIMATE_OPTIONS_H