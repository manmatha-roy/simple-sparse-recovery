"""
Self-checks for the two algorithms.  Run:  python3 tests.py

  1. fwht / bucket convention: bucket values of a single character on a coset
     equal the restriction formula of Section 3.
  2. Exact recovery on random instances, both oracle modes, at the empirical
     rate the paper guarantees (>= 2/3 per run).
  3. Query budgets match the paper: Algorithm 1 uses (n+1) 2^d queries with
     d = ceil(2 log2(4k)); Algorithm 2 uses at most 400 k (n+1) queries.
  4. Structured supports that force collisions and cancellations (a whole
     subspace with equal coefficients) are still recovered.
  5. Algorithm 2 decodes from the fixed shifts 0, e_1, ..., e_n only, so its
     query points depend only on the random subspaces (non-adaptivity).
"""

import numpy as np

from oracle import (Oracle, make_sparse_spectrum, fwht, random_full_rank_basis,
                    enumerate_subspace, bucket_index, parity)
from algorithm1 import algorithm1
from algorithm2 import algorithm2


def exact(rec, true_spec, tol=1e-6):
    return set(rec) == set(true_spec) and all(
        abs(rec[a] - true_spec[a]) <= tol for a in true_spec)


def test_bucket_convention(rng):
    n, d = 12, 5
    for _ in range(20):
        H = random_full_rank_basis(n, d, rng)
        He = enumerate_subspace(H, d)
        assert len(set(He.tolist())) == 1 << d, "basis is not independent"
        alpha = int(rng.integers(1, 1 << n))
        c = int(rng.integers(0, 1 << n))
        vals = 1.0 - 2.0 * parity(c ^ He, alpha)          # chi_alpha on c + H
        spec = fwht(vals)
        s = int(bucket_index(np.array([alpha]), H)[0])
        expected = np.zeros(1 << d)
        expected[s] = 1.0 - 2.0 * parity(np.array([c]), alpha)[0]
        assert np.allclose(spec, expected), "restriction formula mismatch"
    print("ok  bucket convention matches the restriction formula")


def test_random_instances(rng):
    for mode, n in (("preprocess", 14), ("dynamic", 20)):
        for k in (1, 3, 8, 20, 50):
            trials, ok1, ok2 = 30, 0, 0
            for _ in range(trials):
                sup, co, ts = make_sparse_spectrum(n, k, rng)
                ok1 += exact(algorithm1(Oracle(n, sup, co, mode), n, k, rng), ts)
                ok2 += exact(algorithm2(Oracle(n, sup, co, mode), n, k, rng), ts)
            assert ok1 / trials >= 2 / 3, (mode, n, k, "A1", ok1)
            assert ok2 / trials >= 2 / 3, (mode, n, k, "A2", ok2)
            print(f"ok  {mode:10s} n={n:2d} k={k:3d}  "
                  f"A1 {ok1}/{trials}  A2 {ok2}/{trials}")


def test_query_budgets(rng):
    n = 30
    for k in (1, 5, 16, 40, 100):
        sup, co, _ = make_sparse_spectrum(n, k, rng)
        o1 = Oracle(n, sup, co, "dynamic")
        algorithm1(o1, n, k, rng)
        d = min(n, int(np.ceil(2 * np.log2(4 * k))))
        assert o1.count == (n + 1) * (1 << d), ("A1 queries", k, o1.count)
        o2 = Oracle(n, sup, co, "dynamic")
        algorithm2(o2, n, k, rng)
        assert o2.count <= 400 * k * (n + 1), ("A2 queries", k, o2.count)
        print(f"ok  n={n} k={k:3d}  A1 queries {o1.count:>9d}   "
              f"A2 queries {o2.count:>7d} <= 400k(n+1) = {400 * k * (n + 1)}")


def test_structured_support(rng):
    # Support = a whole random subspace of dimension m, all coefficients equal.
    # Many buckets hold several coefficients whose values can cancel.
    n, m = 16, 4
    k = 1 << m
    trials, ok = 30, 0
    for _ in range(trials):
        B = random_full_rank_basis(n, m, rng)
        sup = enumerate_subspace(B, m)
        co = np.ones(k)
        ts = {int(a): 1.0 for a in sup}
        ok += exact(algorithm2(Oracle(n, sup, co, "dynamic"), n, k, rng), ts)
    assert ok / trials >= 2 / 3, ("structured", ok)
    print(f"ok  structured support (subspace, equal coeffs) A2 {ok}/{trials}")


def test_nonadaptive_query_points(rng):
    # Record the query points of two runs with the same seed on DIFFERENT
    # functions: they must coincide, since queries depend only on randomness.
    n, k = 14, 10
    points = []
    for fseed in (1, 2):
        frng = np.random.default_rng(fseed)
        sup, co, _ = make_sparse_spectrum(n, k, frng)
        o = Oracle(n, sup, co, "dynamic")
        log = []
        orig = o.query
        o.query = lambda masks, orig=orig, log=log: (log.append(np.array(masks)),
                                                    orig(masks))[1]
        algorithm2(o, n, k, np.random.default_rng(123))
        points.append(np.concatenate(log))
    assert np.array_equal(points[0], points[1]), "query points depend on f"
    print("ok  Algorithm 2 is non-adaptive (query points independent of f)")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    test_bucket_convention(rng)
    test_random_instances(rng)
    test_query_budgets(rng)
    test_structured_support(rng)
    test_nonadaptive_query_points(rng)
    print("\nall tests passed")
