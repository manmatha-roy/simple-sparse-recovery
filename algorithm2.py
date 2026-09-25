"""
Algorithm 2 -- the main algorithm of the paper (exact Fourier-sparse recovery).

Round r = 0, 1, ..., R-1 with R = floor(log_4 k) + 1:

  1. d_r = k / 2^r.  Sample a uniformly random subspace H with
     dim H = ceil(log2(iso_const * d_r))   (paper: iso_const = 100).
  2. Compute the bucket values of the RESIDUAL f - sum_alpha D(alpha) chi_alpha on
     the n+1 cosets 0+H, e_1+H, ..., e_n+H. The oracle only returns bucket values
     of f; the dictionary is subtracted here, in the bucket domain.
  3. Decode EVERY bucket gamma with a nonzero value on 0+H. No singleton test:
         alpha_i = 1[ fhat_{e_i+H}(gamma) * fhat_{H}(gamma) < 0 ],   v = fhat_H(gamma).
  4. ADD every output to the dictionary, D(alpha) += v, deleting alpha if it
     becomes 0. A wrong output from a bucket with a collision is corrected in a
     later round (Lemma 3 of the paper).

The query points depend only on the random subspaces, so the algorithm is
non-adaptive.

Queries: (n+1) * sum_r |H_r| <= 400 k (n+1) = O(nk).   Time: O(nk log k).
Success probability >= 2/3 with iso_const = 100 (the analysis gives > 0.88).

iso_const trades queries for success probability. The analysis bounds the
failure probability by about 12 / iso_const, so iso_const = 100 is the paper's
guarantee. Smaller values (e.g. 10-20) use fewer queries and often still succeed
in practice, but lose the guarantee. `rounds` overrides R, which can help when
iso_const is small.
"""

import numpy as np

from oracle import random_full_rank_basis, enumerate_subspace, bucket_index


def num_rounds(k):
    """R = floor(log_4 k) + 1, computed exactly: the least R with 4^R > k."""
    R = 1
    while 4 ** R <= k:
        R += 1
    return R


def _residual_spectra(oracle, n, H, He, D, stats=None):
    """Bucket values of the residual on the cosets 0+H, e_1+H, ..., e_n+H.

    Returns (base, shifted) where base = residual spectrum on 0 + H and
    shifted[i] = residual spectrum on e_i + H.
    """
    base = oracle.restricted_spectrum(0, He)
    shifted = [oracle.restricted_spectrum(1 << i, He) for i in range(n)]

    if D:
        alphas = np.fromiter(D.keys(), dtype=np.int64, count=len(D))
        vals = np.fromiter(D.values(), dtype=np.float64, count=len(D))
        buckets = bucket_index(alphas, H)
        # chi_alpha(0) = 1 and chi_alpha(e_i) = (-1)^{alpha_i}
        np.subtract.at(base, buckets, vals)
        for i in range(n):
            sign = 1.0 - 2.0 * ((alphas >> i) & 1)
            np.subtract.at(shifted[i], buckets, vals * sign)
    if stats is not None:
        s, dimH = len(He), len(H)
        stats["wht_ops"] += (n + 1) * s * dimH              # s log2 s per transform
        stats["residual_ops"] += len(D) * (dimH + n + 1)     # bucket + n+1 updates
        stats["max_dict"] = max(stats["max_dict"], len(D))
    return base, shifted


def new_stats():
    """Operation counters for the running-time analysis (see algorithm2)."""
    return {"wht_ops": 0, "residual_ops": 0, "decode_ops": 0,
            "dict_updates": 0, "max_dict": 0}


def algorithm2(oracle, n, k, rng, iso_const=100, rounds=None, tol=1e-9,
               stats=None):
    """Run Algorithm 2. If `stats` is a dict from new_stats(), count operations:

      wht_ops       additions in the n+1 Walsh-Hadamard transforms per round
                    (|H| log2 |H| each)
      residual_ops  bucket computations and subtractions for the dictionary
                    (|D| (dim H + n + 1) per round)
      decode_ops    bucket scans and sign tests (|H| + n * #decoded per round)
      dict_updates  dictionary insertions / updates / deletions
      max_dict      largest dictionary size seen at the start of a round
    """
    D = {}                                  # dictionary of recovered coefficients
    R = num_rounds(k) if rounds is None else rounds

    for r in range(R):
        d_r = k / 2 ** r
        dimH = int(np.ceil(np.log2(max(iso_const * d_r, 2))))
        dimH = max(1, min(n, dimH))
        H = random_full_rank_basis(n, dimH, rng)
        He = enumerate_subspace(H, dimH)

        base, shifted = _residual_spectra(oracle, n, H, He, D, stats)

        # Decode every bucket with a nonzero value on 0 + H.
        occ = np.nonzero(np.abs(base) > tol)[0]
        if stats is not None:
            stats["decode_ops"] += len(He) + n * occ.size
            stats["dict_updates"] += occ.size
        if occ.size == 0:
            continue
        alphas = np.zeros(occ.size, dtype=np.int64)
        for i in range(n):
            alphas |= ((shifted[i][occ] * base[occ]) < 0).astype(np.int64) << i

        # Additive dictionary update; delete keys that return to zero.
        for alpha, v in zip(alphas.tolist(), base[occ].tolist()):
            new = D.get(alpha, 0.0) + v
            if abs(new) <= tol:
                D.pop(alpha, None)
            else:
                D[alpha] = new

    return D
