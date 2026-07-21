"""Python replica of the library's modified Cramer-von Mises distance.

Replicates exactly what gm_to_dirac_short<double> computes as the reported
distance (combined_distance_metric):  D(X, B) = -2 * int_0^B P2(b) db + D3(B)
with the same formulas and constants as gm_to_dirac_short.tpp (the constant
D1 term is excluded there too).  Used by make_bmax_figures.py to ask, for a
FIXED pair of point sets, how much the distance at cutoff B distinguishes
them -- i.e. how much optimization-relevant information each kernel scale b
contributes.  `validate()` checks the replica against the C++ driver output.

Also provides the 1D LCD profiles (Gaussian vs Dirac mixture) used in the
"what does the kernel scale b see" figure.
"""

import numpy as np
from scipy import integrate

EULER_GAMMA = 0.5772156649015  # value used in gm_to_dirac_short.tpp


def p2(b, x, sigma, w=None):
    """Integrand of the Gaussian-Dirac cross term (calculateP2)."""
    x = np.asarray(x, dtype=float)
    L, N = x.shape
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (N,))
    w = np.full(L, 1.0 / L) if w is None else np.asarray(w)
    two_b_sqrd = 2.0 * b * b
    denom = sigma ** 2 + two_b_sqrd
    prefactor = b * np.prod(np.sqrt(two_b_sqrd / denom))
    inner = np.sum(x ** 2 / denom[None, :], axis=1)
    return prefactor * np.sum(w * np.exp(-0.5 * inner))


def d2(x, sigma, bmax, w=None):
    """-2 * int_0^bmax P2(b) db  (calculateD2, f part), via adaptive quad."""
    val, _ = integrate.quad(p2, 0.0, float(bmax), args=(x, sigma, w),
                            limit=400, epsabs=0.0, epsrel=1e-12)
    return -2.0 * val


def d3(x, bmax, w=None):
    """Dirac-Dirac closed form (calculateD3, f part), code's constants."""
    x = np.asarray(x, dtype=float)
    L = x.shape[0]
    w = np.full(L, 1.0 / L) if w is None else np.asarray(w)
    bmax = float(bmax)
    bmax_sqrd = bmax * bmax
    log_approx = EULER_GAMMA - 1.0 - 2.0 * np.log(2.0)

    diff = x[:, None, :] - x[None, :, :]
    dsq = np.sum(diff ** 2, axis=2)
    ww = np.outer(w, w)

    coincident = dsq <= 0.0                       # includes the diagonal
    total = np.sum(ww[coincident]) * (bmax_sqrd / 2.0)

    m = ~coincident
    dm = dsq[m]
    total += np.sum(0.125 * ww[m] *
                    (4.0 * bmax_sqrd
                     + (log_approx - 2.0 * np.log(bmax)) * dm
                     + dm * np.log(dm)))
    return float(total)


def distance(x, sigma, bmax, w=None):
    """The library's reported distance: D2 + D3 (D1 omitted, as in code)."""
    return d2(x, sigma, bmax, w) + d3(x, bmax, w)


# ---------------------------------------------------------------------------
# 1D LCD profiles for the illustration figure (standard textbook convention:
# kernel K(x - m, b) = prod_k exp(-(x_k - m_k)^2 / (2 b^2)) )
# ---------------------------------------------------------------------------

def lcd_gaussian(m_grid, b, sigma):
    """LCD of N(0, diag(sigma^2)) along the first axis (other coords 0)."""
    sigma = np.asarray(sigma, dtype=float)
    amp = np.prod(b / np.sqrt(sigma ** 2 + b ** 2))
    return amp * np.exp(-m_grid ** 2 / (2.0 * (sigma[0] ** 2 + b ** 2)))


def lcd_dirac(m_grid, b, x, w=None):
    """LCD of the Dirac mixture along the first axis (other coords 0)."""
    x = np.asarray(x, dtype=float)
    L, N = x.shape
    w = np.full(L, 1.0 / L) if w is None else np.asarray(w)
    m = np.zeros((len(m_grid), N))
    m[:, 0] = m_grid
    e = np.exp(-np.sum((m[:, None, :] - x[None, :, :]) ** 2, axis=2)
               / (2.0 * b * b))
    return e @ w


def validate(samples_csv, sigma, bmax, expected_dist, tol=1e-6):
    """Compare this replica against a `dist` value reported by the driver."""
    x = np.loadtxt(samples_csv, delimiter=",")
    got = distance(x, sigma, bmax)
    rel = abs(got - expected_dist) / max(1.0, abs(expected_dist))
    assert rel < tol, (got, expected_dist, rel)
    return got, expected_dist, rel


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="validate against driver output")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sigma", type=float, nargs="+", default=[1.0, 1.0])
    ap.add_argument("--bMax", type=float, required=True)
    ap.add_argument("--dist", type=float, required=True,
                    help="dist= value from the driver's RESULT line")
    a = ap.parse_args()
    got, exp, rel = validate(a.samples, a.sigma, a.bMax, a.dist)
    print(f"python={got:.12f}  c++={exp:.12f}  rel.err={rel:.2e}  OK")
