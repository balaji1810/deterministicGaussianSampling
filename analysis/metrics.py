"""Quality metrics for deterministic Gaussian sample sets (2D and general D).

Metrics implemented:

* L2 star discrepancy (Warnock's formula) of the probability-integral
  transform u_i = Phi(x_i / sigma) of the samples -- a standard
  quasi-Monte-Carlo uniformity measure, evaluated with p=2 exactly in O(L^2).

* Moments: weighted mean and covariance of the Dirac mixture.  The mixture
  with weights w_i = 1/L *is* the approximating distribution, so its
  covariance is sum_i w_i (x_i - m)(x_i - m)^T -- divisor L, not L-1
  (no iid unbiasedness correction applies).

* Local homogeneity: coefficient of variation of nearest-neighbour
  distances in u-space (uniformized samples; a perfectly homogeneous set
  has identical NN distances), and the minimum pairwise distance in
  x-space (packing radius).
"""

import numpy as np
from scipy import special


# ---------------------------------------------------------------------------
# L2 star discrepancy of the uniformized samples (Warnock's formula)
# ---------------------------------------------------------------------------

def to_uniform(x, sigma):
    """Probability integral transform per axis: u = Phi(x/sigma) in (0,1)."""
    x = np.asarray(x, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (x.shape[1],))
    return special.ndtr(x / sigma[None, :])


def l2_star_discrepancy(u):
    """Exact L2 star discrepancy (p=2) via Warnock's formula."""
    u = np.asarray(u, dtype=float)
    L, D = u.shape
    t1 = 3.0 ** (-D)
    t2 = np.sum(np.prod(1.0 - u ** 2, axis=1)) * 2.0 ** (1 - D) / L
    m = np.maximum(u[:, None, :], u[None, :, :])
    t3 = np.sum(np.prod(1.0 - m, axis=2)) / L ** 2
    return np.sqrt(max(t1 - t2 + t3, 0.0))


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def moments(x, weights=None):
    """Weighted mean and covariance of the Dirac mixture (divisor L)."""
    x = np.asarray(x, dtype=float)
    L = x.shape[0]
    w = np.full(L, 1.0 / L) if weights is None else np.asarray(weights)
    mean = w @ x
    xc = x - mean
    cov = (w[:, None] * xc).T @ xc
    return mean, cov


def moment_errors(x, sigma, weights=None):
    """Mean norm and Frobenius covariance error vs target diag(sigma^2)."""
    D = np.asarray(x).shape[1]
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (D,))
    mean, cov = moments(x, weights)
    target = np.diag(sigma ** 2)
    return {
        "mean_norm": float(np.linalg.norm(mean)),
        "cov_err_fro": float(np.linalg.norm(cov - target, "fro")),
        "cov_trace_ratio": float(np.trace(cov) / np.trace(target)),
        "cov": cov,
    }


# ---------------------------------------------------------------------------
# Local homogeneity
# ---------------------------------------------------------------------------

def nn_stats(x):
    """Nearest-neighbour distance statistics of a point set."""
    x = np.asarray(x, dtype=float)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return {
        "nn_mean": float(nn.mean()),
        "nn_cv": float(nn.std() / nn.mean()),
        "min_dist": float(nn.min()),
    }


def all_metrics(x, sigma, weights=None):
    """Bundle of every metric used in the study, as a flat dict."""
    x = np.asarray(x, dtype=float)
    out = {}
    u = to_uniform(x, sigma)
    out["l2_star_disc_u"] = float(l2_star_discrepancy(u))
    me = moment_errors(x, sigma, weights)
    out["mean_norm"] = me["mean_norm"]
    out["cov_err_fro"] = me["cov_err_fro"]
    out["cov_trace_ratio"] = me["cov_trace_ratio"]
    nu = nn_stats(u)
    out["nn_cv_u"] = nu["nn_cv"]
    nx = nn_stats(x)
    out["min_dist_x"] = nx["min_dist"]
    return out


if __name__ == "__main__":
    # sanity check: a jittered grid is more uniform (lower discrepancy and
    # lower NN-CV) than iid uniform points.
    rng = np.random.default_rng(0)
    n = 8
    g = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(g, g)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)
    rand = rng.uniform(0.0, 1.0, size=(n * n, 2))
    print(f"grid : discrepancy={l2_star_discrepancy(grid):.4f}  "
          f"NN-CV={nn_stats(grid)['nn_cv']:.3f}")
    print(f"rand : discrepancy={l2_star_discrepancy(rand):.4f}  "
          f"NN-CV={nn_stats(rand)['nn_cv']:.3f}")
    assert l2_star_discrepancy(grid) < l2_star_discrepancy(rand)
    assert nn_stats(grid)["nn_cv"] < nn_stats(rand)["nn_cv"]
    print("ok: jittered grid is more uniform than iid points")
