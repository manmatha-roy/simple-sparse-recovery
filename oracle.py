"""
Shared foundation for the exact Fourier-sparse recovery routines.

Contains everything BOTH algorithms depend on, so neither algorithm file imports
the other and the two implementations stay genuinely independent:

  - fwht                    : fast Walsh-Hadamard transform (E_x[.] normalization)
  - make_sparse_spectrum    : random k-Fourier-sparse INPUT (support + coeffs)
  - random_full_rank_basis  : random d-dim subspace H <= F_2^n
  - enumerate_subspace      : list all 2^d elements of H
  - Oracle                  : query-counting function oracle in two modes
                                'preprocess' -- build the full 2^n table once
                                'dynamic'    -- evaluate f on the fly, O(k)/query

Fourier convention (+-1):  f(x) = sum_alpha fhat(alpha) chi_alpha(x),
                           chi_alpha(x) = (-1)^{<alpha,x>},  fhat(alpha)=E_x[f chi].
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard transform
# ---------------------------------------------------------------------------

def fwht(a):
    """FWHT with the E_x[.] normalization (divide by N). Length of a must be 2^m."""
    a = a.astype(np.float64).copy()
    h, N = 1, len(a)
    while h < N:
        for i in range(0, N, h * 2):
            x = a[i:i + h].copy()
            y = a[i + h:i + 2 * h].copy()
            a[i:i + h] = x + y
            a[i + h:i + 2 * h] = x - y
        h *= 2
    return a / N


# ---------------------------------------------------------------------------
# Instance generation: random k-Fourier-sparse input
# ---------------------------------------------------------------------------

def make_sparse_spectrum(n, k, rng, coeff_scale=1.0):
    """Random k-Fourier-sparse input.

    Support: k DISTINCT frequencies drawn uniformly from F_2^n (no replacement).
    Coeffs : each ~ Uniform[-coeff_scale, coeff_scale], independent; any draw
             within 1e-6 of 0 is nudged away so the frequency is truly nonzero.

    Returns (support ndarray, coeffs ndarray, true_spec dict {alpha_int: coeff}).
    No 2^n object is built here -- only the k-sparse description.
    """
    N = 1 << n
    support = rng.choice(N, size=k, replace=False).astype(np.int64)
    coeffs = rng.uniform(-coeff_scale, coeff_scale, size=k)
    coeffs[np.abs(coeffs) < 1e-6] += 0.5
    true_spec = {int(a): float(c) for a, c in zip(support, coeffs)}
    return support, coeffs, true_spec


# ---------------------------------------------------------------------------
# Subspace helpers over F_2^n
# ---------------------------------------------------------------------------

def random_full_rank_basis(n, d, rng):
    """d integers spanning a random d-dimensional subspace H <= F_2^n."""
    basis = []
    while len(basis) < d:
        v = int(rng.integers(1, 1 << n))
        red = v
        for b in basis:
            red = min(red, red ^ b)
        if red != 0:
            basis.append(v)
    return basis


def enumerate_subspace(basis, d):
    """All 2^d elements of the subspace spanned by `basis` (list of d ints)."""
    elems = np.zeros(1 << d, dtype=np.int64)
    for j in range(d):
        stride, block = 1 << j, 1 << (j + 1)
        idx = (np.arange(1 << d) % block) >= stride
        elems[idx] ^= basis[j]
    return elems


def parity(masks, alpha):
    """popcount(masks & alpha) mod 2, vectorized over the array `masks`."""
    v = masks & alpha
    par = np.zeros_like(v)
    while v.any():
        par ^= (v & 1)
        v >>= 1
    return par


# ---------------------------------------------------------------------------
# Oracle: two modes behind one interface
# ---------------------------------------------------------------------------

class Oracle:
    """Query-counting oracle over a k-sparse function, plus peeled residual terms.

    mode = 'preprocess' : build the full 2^n table once; queries are O(1) lookups.
    mode = 'dynamic'    : evaluate f(x) = sum_alpha c_alpha (-1)^{<alpha,x>} per x,
                          O(k) per query point; no 2^n storage (scales to large n).

    Both modes expose the same method the algorithms use:
        restricted_spectrum(a, H_elems) -> spectrum of the residual on coset a+H.

    Timing:  self.count       counts oracle queries (points evaluated),
             self.oracle_time  accumulates ONLY function-evaluation time, so the
                               driver can separate it from pure algorithm time.
                               (The FWHT is algorithmic work and is NOT counted
                               as oracle time.)
    """

    def __init__(self, n, support, coeffs, mode):
        self.n = n
        self.support = np.asarray(support, dtype=np.int64)
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.mode = mode
        self.count = 0
        self.oracle_time = 0.0
        self.peeled = []                      # list of (alpha_int, coeff)

        if mode == "preprocess":
            N = 1 << n
            fhat = np.zeros(N)
            fhat[self.support] = self.coeffs
            self.table = fwht(fhat) * N        # f on all 2^n inputs
        elif mode == "dynamic":
            self.table = None
        else:
            raise ValueError("mode must be 'preprocess' or 'dynamic'")

    def _f(self, masks):
        """Base function values at the given subset masks."""
        if self.mode == "preprocess":
            return self.table[masks].astype(np.float64)
        vals = np.zeros(len(masks), dtype=np.float64)
        for alpha, c in zip(self.support.tolist(), self.coeffs.tolist()):
            vals += c * np.where(parity(masks, alpha) == 0, 1.0, -1.0)
        return vals

    def restricted_spectrum(self, a, H_elems):
        """Spectrum of the residual restricted to coset a + H (length 2^dim H)."""
        masks = (a ^ H_elems).astype(np.int64)
        self.count += masks.size
        _t = time.perf_counter()
        vals = self._f(masks)
        for alpha, coeff in self.peeled:
            vals -= coeff * np.where(parity(masks, alpha) == 0, 1.0, -1.0)
        self.oracle_time += time.perf_counter() - _t
        return fwht(vals)
