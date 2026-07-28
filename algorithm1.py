"""
Algorithm 1 -- warm-up hash-and-decode for exact Fourier-sparse recovery.

Restrict f to a random subspace H of dimension d = 2*ceil(log2 k) + dim_slack, so
large that (birthday bound) the map alpha -> alpha|_H is injective on the support:
every active frequency lands in its own bucket. Then decode each isolated frequency
coordinate-wise from phase ratios:

        fhat_{e_i}(gamma) / fhat_0(gamma) = (-1)^{alpha_i}.

Queries / time:  O~(n k^2).

dim_slack sits in the EXPONENT of the subspace size (2^d), so each extra unit
DOUBLES the query count. The birthday bound needs collision prob k^2 / 2^d < const:
  slack = 2  -> ~1/4 collision prob  -> success comfortably above 2/3
  slack = 5+ -> high-probability recovery
Raise it for a stronger guarantee, at 2x queries per unit.
"""

import numpy as np

from oracle import random_full_rank_basis, enumerate_subspace


def algorithm1(oracle, n, k, rng, dim_slack=2):
    d = 2 * max(1, int(np.ceil(np.log2(max(k, 2))))) + dim_slack
    d = min(d, n)
    H = random_full_rank_basis(n, d, rng)
    He = enumerate_subspace(H, d)

    # cosets 0, e_1, ..., e_n
    base = oracle.restricted_spectrum(0, He)
    coord = [oracle.restricted_spectrum(1 << i, He) for i in range(n)]

    occ = np.nonzero(np.abs(base) >= 1e-9)[0]
    recovered = {}
    for t in occ:
        b0 = base[t]
        alpha = 0
        for i in range(n):
            if coord[i][t] / b0 < 0:
                alpha |= (1 << i)
        recovered[alpha] = float(b0)
    return recovered
