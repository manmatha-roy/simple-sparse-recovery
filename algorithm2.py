"""
Algorithm 2 -- peeling with BLR singleton test for exact Fourier-sparse recovery.

Each round restrict f to a SMALL random subspace (~iso_const * d_est buckets), so
collisions occur but a constant fraction of frequencies are isolated. Use a
Blum-Luby-Rubinfeld linearity test (shared across all buckets) to keep singleton
buckets, decode them coordinate-wise, subtract the recovered characters from the
residual, and repeat. Sparsity falls geometrically -> O(log k) rounds.

Queries / time:  O~(n k).

iso_const controls buckets per round: dimH = ceil(log2(iso_const * d_est)). The
paper uses 100 (isolation failure <= 1/100). For a >=2/3-success demo a much
smaller constant suffices and cuts queries by ~log2(100/iso_const) in the exponent:
  iso_const = 10  -> success ~ 85%+ across a range of n, k (default)
  iso_const = 100 -> the paper's high-probability constant
Note: very small constants (e.g. 6) can dip below 2/3 at larger n, since success
requires every peeling round to isolate -- failures compound across rounds.
"""

import numpy as np

from oracle import random_full_rank_basis, enumerate_subspace


def algorithm2(oracle, n, k, rng, max_rounds=None, iso_const=10):
    recovered = {}
    d_est = k
    max_rounds = max_rounds or (4 * int(np.ceil(np.log2(max(k, 2)))) + 30)

    for _ in range(max_rounds):
        if len(recovered) >= k:
            break
        dimH = max(1, min(n, int(np.ceil(
            np.log2(max(iso_const * max(d_est, 1), 2))))))
        H = random_full_rank_basis(n, dimH, rng)
        He = enumerate_subspace(H, dimH)

        base = oracle.restricted_spectrum(0, He)
        occ = np.nonzero(np.abs(base) >= 1e-9)[0]
        if occ.size == 0:
            d_est = max(1, k - len(recovered)); continue

        # BLR singleton filter -- all buckets share the same anchor cosets.
        keep = np.ones(occ.size, dtype=bool)
        for _ in range(3):
            a = int(rng.integers(0, 1 << n))
            b = int(rng.integers(0, 1 << n))
            sa = oracle.restricted_spectrum(a, He)[occ]
            sb = oracle.restricted_spectrum(b, He)[occ]
            sab = oracle.restricted_spectrum(a ^ b, He)[occ]
            den = base[occ]
            keep &= np.isclose((sa / den) * (sb / den), sab / den, atol=1e-6)
        singles = occ[keep]
        if singles.size == 0:
            d_est = max(1, k - len(recovered)); continue

        # Decode surviving singletons -- shared coordinate cosets a, a+e_i.
        a = int(rng.integers(0, 1 << n))
        base_a = oracle.restricted_spectrum(a, He)[singles]
        good = np.abs(base_a) >= 1e-12
        singles = singles[good]; base_a = base_a[good]
        alphas = np.zeros(singles.size, dtype=np.int64)
        for i in range(n):
            ratio = oracle.restricted_spectrum(a ^ (1 << i), He)[singles] / base_a
            alphas |= (ratio < 0).astype(np.int64) << i

        for alpha, g in zip(alphas.tolist(), singles.tolist()):
            if alpha not in recovered:
                coeff = float(base[g])
                recovered[alpha] = coeff
                oracle.peeled.append((alpha, coeff))
        d_est = max(1, k - len(recovered))

    return recovered
