"""
Algorithm 1 -- warm-up hash-and-decode for exact Fourier-sparse recovery.

Restrict f to a random subspace H of dimension d = ceil(2 log2(4k)), so large
that (birthday bound, Lemma 1) the map alpha -> bucket(alpha) is injective on the
support, except with probability 1/16. Every active frequency then lands alone in
its bucket, and it is decoded coordinate-wise from signs:

    fhat_{e_i+H}(gamma) * fhat_{H}(gamma) = (-1)^{alpha_i} fhat(alpha)^2,

which is negative exactly when alpha_i = 1.

Queries: (n+1) * 2^d <= 32 k^2 (n+1).   Time: O(n k^2 log k).

dim_slack adds to d. Each extra unit DOUBLES the queries and halves the
collision probability. dim_slack = 0 is the paper's setting.
"""

import numpy as np

from oracle import random_full_rank_basis, enumerate_subspace


def algorithm1(oracle, n, k, rng, dim_slack=0, tol=1e-9):
    d = int(np.ceil(2 * np.log2(4 * max(k, 1)))) + dim_slack
    d = max(1, min(d, n))
    H = random_full_rank_basis(n, d, rng)
    He = enumerate_subspace(H, d)

    # cosets 0 + H, e_1 + H, ..., e_n + H
    base = oracle.restricted_spectrum(0, He)
    occ = np.nonzero(np.abs(base) > tol)[0]

    alphas = np.zeros(occ.size, dtype=np.int64)
    for i in range(n):
        shifted = oracle.restricted_spectrum(1 << i, He)
        alphas |= ((shifted[occ] * base[occ]) < 0).astype(np.int64) << i

    return {int(a): float(base[t]) for a, t in zip(alphas.tolist(), occ.tolist())}
