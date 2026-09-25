"""
Shared foundation for the exact Fourier-sparse recovery routines.

Contains everything BOTH algorithms depend on, so neither algorithm file imports
the other and the two implementations stay independent:

  - fwht                    : fast Walsh-Hadamard transform (E_x[.] normalization)
  - make_sparse_spectrum    : random k-Fourier-sparse INPUT (support + coeffs)
  - random_full_rank_basis  : random d-dim subspace H <= F_2^n
  - enumerate_subspace      : list all 2^d elements of H
  - parity                  : vectorized <mask, alpha> mod 2
  - bucket_index            : the bucket of a frequency alpha under H
  - Oracle                  : query-counting function oracle in two modes
                                'preprocess' -- build the full 2^n table once
                                'dynamic'    -- evaluate f on the fly, O(k)/query

Fourier convention (+-1):  f(x) = sum_alpha fhat(alpha) chi_alpha(x),
                           chi_alpha(x) = (-1)^{<alpha,x>},  fhat(alpha)=E_x[f chi].

Bucket convention.  enumerate_subspace(basis, d)[t] = XOR of basis[j] over the
set bits j of t.  With this ordering, fwht of the values of f on the coset c+H
returns, at index s, the bucket value

        sum_{alpha : bucket_index(alpha) = s}  fhat(alpha) chi_alpha(c),

where bucket_index(alpha) = sum_j <alpha, basis[j]> 2^j.  This is the
restriction formula of Section 3 of the paper, with buckets indexed through the
basis of H (so the complement W never needs to be computed).
"""

import time

import numpy as np


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard transform
# ---------------------------------------------------------------------------

def fwht(a):
    """FWHT with the E_x[.] normalization (divide by N). Length of a must be 2^m."""
    a = np.asarray(a, dtype=np.float64).copy()
    h, N = 1, len(a)
    while h < N:
        a = a.reshape(-1, 2 * h)
        x = a[:, :h].copy()
        y = a[:, h:].copy()
        a[:, :h] = x + y
        a[:, h:] = x - y
        a = a.reshape(-1)
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
    """d integers spanning a uniformly random d-dimensional subspace H <= F_2^n.

    Draws random nonzero vectors and keeps each one that is independent of the
    ones kept so far (Gaussian elimination on the fly). The span is uniform over
    all d-dimensional subspaces.
    """
    basis, reduced = [], []
    while len(basis) < d:
        v = int(rng.integers(1, 1 << n))
        red = v
        for b in reduced:
            red = min(red, red ^ b)
        if red != 0:
            basis.append(v)
            reduced.append(red)
            reduced.sort(reverse=True)
    return basis


def enumerate_subspace(basis, d):
    """All 2^d elements of the subspace spanned by `basis` (list of d ints).

    Element t is the XOR of basis[j] over the set bits j of t.
    """
    elems = np.zeros(1 << d, dtype=np.int64)
    t = np.arange(1 << d)
    for j in range(d):
        elems[(t >> j) & 1 == 1] ^= basis[j]
    return elems


def parity(masks, alpha):
    """popcount(masks & alpha) mod 2, vectorized over the array `masks`."""
    v = np.asarray(masks, dtype=np.int64) & alpha
    return (np.bitwise_count(v) & 1).astype(np.int64)


def bucket_index(alphas, basis):
    """Bucket of each frequency: sum_j <alpha, basis[j]> 2^j (vectorized)."""
    alphas = np.asarray(alphas, dtype=np.int64)
    idx = np.zeros(alphas.shape, dtype=np.int64)
    for j, h in enumerate(basis):
        idx |= (np.bitwise_count(alphas & h) & 1).astype(np.int64) << j
    return idx


# ---------------------------------------------------------------------------
# Oracle: two modes behind one interface
# ---------------------------------------------------------------------------

class Oracle:
    """Query-counting oracle for a k-sparse function f.

    mode = 'preprocess' : build the full 2^n table once; queries are O(1) lookups.
    mode = 'dynamic'    : evaluate f(x) = sum_alpha c_alpha (-1)^{<alpha,x>} per x,
                          O(k) per query point; no 2^n storage (scales to large n).

    Both modes expose the one method the algorithms use:
        restricted_spectrum(c, H_elems) -> bucket values of f on the coset c + H.

    The oracle knows nothing about residuals: the algorithms subtract their own
    dictionary in the bucket domain, as in the paper ("evaluated in software").

    Accounting:  self.count        number of query points evaluated,
                 self.oracle_time  time spent ONLY on function evaluation, so the
                                   driver can separate it from algorithm time.
                                   The FWHT and all residual bookkeeping are
                                   algorithmic work and are NOT counted here.
    """

    def __init__(self, n, support, coeffs, mode):
        self.n = n
        self.support = np.asarray(support, dtype=np.int64)
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.mode = mode
        self.count = 0
        self.oracle_time = 0.0

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
        """Function values at the given points (as integer bit masks)."""
        if self.mode == "preprocess":
            return self.table[masks].astype(np.float64)
        vals = np.zeros(len(masks), dtype=np.float64)
        for alpha, c in zip(self.support.tolist(), self.coeffs.tolist()):
            vals += c * (1.0 - 2.0 * parity(masks, alpha))
        return vals

    def query(self, masks):
        """Evaluate f at the given points, counting queries and oracle time."""
        masks = np.asarray(masks, dtype=np.int64)
        self.count += masks.size
        t0 = time.perf_counter()
        vals = self._f(masks)
        self.oracle_time += time.perf_counter() - t0
        return vals

    def restricted_spectrum(self, c, H_elems):
        """Bucket values of f on the coset c + H (length 2^dim H)."""
        return fwht(self.query(c ^ H_elems))
