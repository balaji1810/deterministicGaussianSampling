"""
Reference implementation of the closed forms in Gilitschenski, Chapter 4
("Approximation of Gaussian Densities"), verified against direct numerical
quadrature of the b-integral.

Conventions (all matched to the thesis):
  - target density: n-dim STANDARD normal N(0, I),  n = 2k
  - Dirac mixture:  m samples s_i with weights w_i, sum w_i = 1
  - weighting fn:   w(b) = b^{-n+1} on [0, bMax]
  - D = int w(b) (I1(b) - 2 I2(b) + I3(b)) db
  - everything below is reported in units of pi^{n/2}  (i.e. pi^{-k} D),
    which is exactly the normalisation the C++ library uses
    (twoPiNHalf = 2^{N/2}, i.e. pi^{N/2} factored out).

  c_i  = ||s_i||^2
  T_ij = ||s_i - s_j||^2
"""
import numpy as np
from scipy.special import expi, comb
from scipy.integrate import quad
from math import factorial, log, exp, sqrt, pi, gamma as _g

GAMMA_E = 0.5772156649015328606

# ----------------------------------------------------------------------------
# 1. Integrands  (w(b) * I_j),  divided by pi^{n/2}
# ----------------------------------------------------------------------------

def wI1_num(b, n):
    """w(b)*I1(b)/pi^{n/2} = b^{n+1} / (1+b^2)^{n/2}"""
    return b**(n + 1) / (1.0 + b * b)**(n / 2.0)

def wI2_num(b, n, S, w):
    """w(b)*I2(b)/pi^{n/2} = 2^{n/2} b^{n+1} (1+2b^2)^{-n/2} sum_i w_i exp(-c_i/(2(1+2b^2)))"""
    u = 1.0 + 2.0 * b * b
    c = np.sum(S * S, axis=1)
    return 2.0**(n / 2.0) * b**(n + 1) / u**(n / 2.0) * np.sum(w * np.exp(-0.5 * c / u))

def wI3_num(b, n, S, w):
    """w(b)*I3(b)/pi^{n/2} = b * sum_ij w_i w_j exp(-T_ij/(4 b^2))"""
    T = _T(S)
    return b * np.sum(np.outer(w, w) * np.exp(-T / (4.0 * b * b)))

def _T(S):
    d = S[:, None, :] - S[None, :, :]
    return np.sum(d * d, axis=2)

# ----------------------------------------------------------------------------
# 2. Closed forms
# ----------------------------------------------------------------------------

def A(n, b):
    """Theorem 4.6: int_0^b x^{n+1}/(1+x^2)^{n/2} dx, valid for ALL n>=1."""
    if n == 1:
        return 0.5 * (b * sqrt(1 + b * b) - np.arcsinh(b))
    if n == 2:
        return 0.5 * (b * b - log(1 + b * b))
    # recursion A_n = 1/(n-2) * ( n A_{n-2} - b^n/(1+b^2)^{(n-2)/2} )
    return (n * A(n - 2, b) - b**n / (1 + b * b)**((n - 2) / 2.0)) / (n - 2)

def B0(d, b, c):
    """Theorem 4.7 base cases B_{0,d}(b,c)."""
    u = 1.0 + 2.0 * b * b            # = (2+4b^2)/2
    e = exp(-c / (2.0 * u))
    if d == 0:
        return u / 4.0 * e + c / 8.0 * expi(-c / (2.0 * u))
    if d == 1:
        return -0.25 * expi(-c / (2.0 * u))
    s = 0.0
    for j in range(2, d + 1):
        s += factorial(d - 2) * 2.0**(d - j - 1) / (
            factorial(j - 2) * c**(d - j + 1) * u**(j - 2))
    return e * s

def Bkk(k, b, c):
    """B_{k,k}(b,c) = 2^{-k} sum_{j=0}^{k} (-1)^j C(k,j) B_{0,j}(b,c)"""
    return sum((-1)**j * comb(k, j, exact=True) * B0(j, b, c)
               for j in range(k + 1)) / 2.0**k

def Bkk1(k, b, c):
    """B_{k,k+1}(b,c) = 2^{-k} sum_{j=0}^{k} (-1)^j C(k,j) B_{0,j+1}(b,c)"""
    return sum((-1)**j * comb(k, j, exact=True) * B0(j + 1, b, c)
               for j in range(k + 1)) / 2.0**k

def C(b, c):
    """Eq. (4.6): int_0^b x exp(-c/(4x^2)) dx, EXACT for any b>0."""
    if c <= 0.0:
        return 0.5 * b * b
    z = -c / (4.0 * b * b)
    return 0.5 * b * b * exp(z) + c / 8.0 * expi(z)

# ------- assembled distance, finite bMax, closed form (pi^{-k} D) ------------

def D_closed(n, bmax, S, w):
    k = n // 2
    c = np.sum(S * S, axis=1)
    T = _T(S)
    d1 = A(n, bmax)
    d2 = 2.0**k * sum(wi * (Bkk(k, bmax, ci) - Bkk(k, 0.0, ci))
                      for wi, ci in zip(w, c))
    d3 = sum(w[i] * w[j] * C(bmax, T[i, j])
             for i in range(len(w)) for j in range(len(w)))
    return d1 - 2.0 * d2 + d3, d1, d2, d3

def D_numeric(n, bmax, S, w):
    d1 = quad(wI1_num, 0, bmax, args=(n,), limit=400)[0]
    d2 = quad(wI2_num, 0, bmax, args=(n, S, w), limit=400)[0]
    d3 = quad(wI3_num, 0, bmax, args=(n, S, w), limit=400)[0]
    return d1 - 2.0 * d2 + d3, d1, d2, d3

# ------- bMax -> infinity form (Theorem 4.10), equal means only -------------
# NOTE: the coefficient on B_{k,k}(0,c_i) is 2^{k+1}, NOT 2 as printed in the
# thesis.  The printed version factors -2 out of a bracket that still carried a
# 2^k, which silently drops 2^k from a POSITION-DEPENDENT term (so it moves the
# optimum, it is not a harmless constant).  Verified in verify_all().

def Sk(k, c):
    return sum((-1)**j * comb(k, j, exact=True) * factorial(j - 2) * 2.0**(j - 3) / c**(j - 1)
               for j in range(2, k + 1))

def D_infinite(n, S, w):
    """pi^{-k} D for bMax -> inf.  VALID ONLY IF sum_i w_i s_i = 0."""
    k = n // 2
    c = np.sum(S * S, axis=1)
    T = _T(S)
    L = len(w)
    const = k * sum(1.0 / (2.0 * i) for i in range(2, k + 1)) - 0.5 - k * GAMMA_E / 2.0
    attract = (-2.0 * sum(w[i] * ((c[i] + 2 * k) / 8.0 * log(c[i] / 4.0) + Sk(k, c[i]))
                          for i in range(L))
               + 2.0**(k + 1) * sum(w[i] * Bkk(k, 0.0, c[i]) for i in range(L)))
    rep = sum(w[i] * w[j] * T[i, j] / 8.0 * log(abs(T[i, j] / 4.0))
              for i in range(L) for j in range(L) if T[i, j] > 0)
    return const + attract + rep

def D_infinite_n2(S, w):
    """Vectorised n=2 (k=1) special case -- the one worth porting first."""
    c = np.sum(S * S, axis=1)
    T = _T(S)
    B0 = 0.125 * np.exp(-c / 2) + (c + 2) / 16 * expi(-c / 2)      # B_{1,1}(0,c)
    att = -2 * np.sum(w * ((c + 2) / 8 * np.log(c / 4))) + 4 * np.sum(w * B0)
    M = np.outer(w, w) * T / 8 * np.log(np.abs(np.where(T > 0, T, 1)) / 4)
    return att + M.sum() - 0.5 - GAMMA_E / 2

def grad_infinite_n2(S, w):
    c = np.sum(S * S, axis=1)
    T = _T(S)
    e = np.exp(-c / 2); Ei = expi(-c / 2)
    dphi = -0.25 * np.log(c / 4) - (c + 2) / (4 * c) - 0.25 * e + 0.25 * Ei + (c + 2) * e / (4 * c)
    g = 2 * S * (w * dphi)[:, None]
    logT = np.log(np.abs(np.where(T > 0, T, 1)) / 4) + 1.0
    np.fill_diagonal(logT, 0.0)
    diff = S[:, None, :] - S[None, :, :]
    g += 0.5 * w[:, None] * np.einsum('j,ijk->ik', w, diff * logT[:, :, None])
    return g

# ----------------------------------------------------------------------------
# 3. Gradients
# ----------------------------------------------------------------------------

def grad_closed(n, bmax, S, w):
    """d(pi^{-k} D)/d s_q, finite bMax, fully closed form (even n)."""
    k = n // 2
    L, _ = S.shape
    c = np.sum(S * S, axis=1)
    T = _T(S)
    g = np.zeros_like(S)
    # attraction: -2 * dD2, with  int w G1 = -(2)^k w_q s_q (B_{k,k+1}(bmax,c)-B_{k,k+1}(0,c))
    for q in range(L):
        dB = Bkk1(k, bmax, c[q]) - Bkk1(k, 0.0, c[q])
        g[q] += 2.0**(k + 1) * w[q] * S[q] * dB
    # repulsion: dD3, exact Ei form
    for q in range(L):
        acc = np.zeros(n)
        for i in range(L):
            if i == q:
                continue
            acc += w[i] * (S[q] - S[i]) * expi(-T[q, i] / (4.0 * bmax * bmax))
        g[q] += 0.5 * w[q] * acc
    return g

def grad_fd(fun, S, h=1e-6):
    g = np.zeros_like(S)
    for i in range(S.shape[0]):
        for k in range(S.shape[1]):
            Sp = S.copy(); Sp[i, k] += h
            Sm = S.copy(); Sm[i, k] -= h
            g[i, k] = (fun(Sp) - fun(Sm)) / (2 * h)
    return g

# ----------------------------------------------------------------------------
# 4. What the C++ library currently does for D3 (large-bMax log approximation)
# ----------------------------------------------------------------------------

def D3_library_logapprox(bmax, S, w):
    """(1/8) sum_ij w_i w_j (4 bmax^2 - Cb T + T log T),  Cb = log(4 bmax^2) - gamma"""
    T = _T(S)
    Cb = log(4.0 * bmax * bmax) - GAMMA_E
    tot = 0.0
    L = len(w)
    for i in range(L):
        for j in range(L):
            t = T[i, j]
            tot += w[i] * w[j] * (4.0 * bmax * bmax - Cb * t + (t * log(t) if t > 0 else 0.0))
    return tot / 8.0

def D3_exact_Ei(bmax, S, w):
    T = _T(S)
    L = len(w)
    return sum(w[i] * w[j] * C(bmax, T[i, j]) for i in range(L) for j in range(L))

# ----------------------------------------------------------------------------
# 5. Verification suite  --  run:  python3 lcd_closed_form.py
# ----------------------------------------------------------------------------

def verify_all():
    rng = np.random.default_rng(7)
    ok = True
    print("[1] closed forms vs direct quadrature")
    for n in (2, 4, 6):
        for bmax in (1.0, 5.0, 100.0):
            L = 7; S = rng.normal(size=(L, n)); w = np.ones(L) / L
            a = D_closed(n, bmax, S, w); b = D_numeric(n, bmax, S, w)
            e = max(abs(a[i] - b[i]) / max(abs(b[i]), 1e-12) for i in (1, 2, 3))
            ok &= e < 1e-8   # limited by the reference quadrature, not the closed form
            print(f"    n={n} bMax={bmax:6.1f}  max rel err {e:.2e}")

    print("[2] analytic gradient vs finite differences")
    for n in (2, 4):
        for bmax in (5.0, 100.0):
            L = 6; S = rng.normal(size=(L, n)); w = np.ones(L) / L
            e = np.max(np.abs(grad_closed(n, bmax, S, w)
                              - grad_fd(lambda Z: D_closed(n, bmax, Z, w)[0], S)))
            ok &= e < 1e-5
            print(f"    n={n} bMax={bmax:6.1f}  max abs err {e:.2e}")

    print("[3] bMax->inf form vs lim_{bMax->inf} of the finite form (zero-mean sets)")
    for n in (2, 4, 6):
        L = 8; S = rng.normal(size=(L, n)); w = np.ones(L) / L; S = S - w @ S
        lim_ = D_closed(n, 3e4, S, w)[0]; inf_ = D_infinite(n, S, w)
        ok &= abs(lim_ - inf_) < 1e-5
        print(f"    n={n}  lim={lim_: .10f}  D_inf={inf_: .10f}  diff {abs(lim_-inf_):.2e}")

    print("[4] non-zero-mean -> true distance diverges like ||mu||^2/2 * ln(bMax)")
    n = 2; L = 8
    S = rng.normal(size=(L, n)); w = np.ones(L) / L
    S = S - w @ S + np.array([0.30, 0.0]); mu = w @ S
    d = [D_closed(n, b, S, w)[0] for b in (1e3, 1e4, 1e5)]
    slope = (d[2] - d[0]) / (2 * log(10))
    ok &= abs(slope - mu @ mu / 2) < 1e-4
    print(f"    measured slope {slope:.6f}   predicted ||mu||^2/2 = {mu@mu/2:.6f}")

    print("[5] library D3 (log form, incl. the -1) vs exact Ei form: error is O(1/bMax^2)")
    L = 40; S = rng.normal(size=(L, 2)); w = np.ones(L) / L
    for bmax in (5.0, 10.0, 100.0):
        lo = log(bmax); la = GAMMA_E - 1.0 - 2.0 * log(2.0); T = _T(S)
        lib = sum(0.125 * w[i] * w[j] * ((la - 2 * lo) * T[i, j] + T[i, j] * log(T[i, j]))
                  for i in range(L) for j in range(L) if i != j) + 0.5 * bmax**2
        print(f"    bMax={bmax:6.1f}  err vs exact Ei = {lib - D3_exact_Ei(bmax,S,w):+.3e}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    verify_all()