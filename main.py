"""
Driver for exact Fourier-sparse recovery: sweep k, run both routines, report
per-routine timing, query count, and success rate.

Interactive:      python3 main.py
Non-interactive:  python3 main.py --mode dynamic --n 16 --k-start 4 --k-end 10 \
                                  --k-step 2 --trials 10 --seed 0

Interactive prompts (press Enter to accept the shown default):
  - oracle mode : preprocess (build 2^n table once) or dynamic (evaluate on the fly)
  - n           : ambient dimension                      [default 16]
  - k start     : first sparsity in the sweep            [default 4]
  - k end       : last sparsity in the sweep, INCLUSIVE  [default 10]
  - k step      : increment                              [default 2]
  - trials      : trials per (n, k) config               [default 10]

For each k the driver runs Algorithm 1 and Algorithm 2 on the SAME random
k-sparse instances and reports, per routine:
    time (s)  -- average recovery time (function-evaluation time excluded)
    queries   -- average number of oracle queries
    success   -- fraction of trials with exact recovery

The report is also saved to  testresults/<mode>_n<n>_k<range>_<timestamp>.txt.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

from oracle import make_sparse_spectrum, Oracle
from algorithm1 import algorithm1
from algorithm2 import algorithm2
from sysinfo import system_info


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def exact(rec, true_spec, tol=1e-6):
    return set(rec) == set(true_spec) and all(
        abs(rec[a] - true_spec[a]) <= tol for a in true_spec)


# ---------------------------------------------------------------------------
# One (n, k) config: run both routines over `trials` shared instances
# ---------------------------------------------------------------------------

def _run_one(alg, n, k, support, coeffs, mode, rng, true_spec):
    o = Oracle(n, support, coeffs, mode)
    t0 = time.perf_counter()
    rec = alg(o, n, k, rng)
    elapsed = (time.perf_counter() - t0) - o.oracle_time
    return elapsed, o.count, exact(rec, true_spec)


def run_config(n, k, trials, mode, rng):
    stats = {name: [0.0, 0, 0] for name in ("a1", "a2")}   # time, queries, ok

    for _ in range(trials):
        support, coeffs, ts = make_sparse_spectrum(n, k, rng)
        for name, alg in (("a1", algorithm1), ("a2", algorithm2)):
            t, q, ok = _run_one(alg, n, k, support, coeffs, mode, rng, ts)
            stats[name][0] += t
            stats[name][1] += q
            stats[name][2] += ok

    T = trials
    row = {"n": n, "k": k, "trials": T}
    for name, (t, q, ok) in stats.items():
        row[f"{name}_time"] = t / T
        row[f"{name}_queries"] = q / T
        row[f"{name}_success"] = ok / T
    return row


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(mode, n, k_start, k_end, k_step, trials, seed=0, verbose=True):
    """Run the sweep. k_end is INCLUSIVE. Returns (rows, report_text)."""
    rng = np.random.default_rng(seed)
    ks = list(range(k_start, k_end + 1, k_step))

    lines = []
    lines.append(f"Oracle mode: {mode.upper()}   n={n}   "
                 f"k = {k_start}..{k_end} step {k_step}   "
                 f"{trials} trials/config   seed={seed}")
    lines.append("")
    lines.append(f"{'k':>4} | {'A1 time(s)':>10} {'A1 queries':>11} {'A1 succ':>8} "
                 f"| {'A2 time(s)':>10} {'A2 queries':>11} {'A2 succ':>8}")
    lines.append("-" * 76)

    rows = []
    for k in ks:
        if k > (1 << n):
            lines.append(f"{k:>4} | skipped: k > 2^n")
            continue
        r = run_config(n, k, trials, mode, rng)
        rows.append(r)
        lines.append(
            f"{k:>4} | {r['a1_time']:>10.4f} {r['a1_queries']:>11.0f} "
            f"{r['a1_success']:>7.0%} "
            f"| {r['a2_time']:>10.4f} {r['a2_queries']:>11.0f} "
            f"{r['a2_success']:>7.0%}")

    report = "\n".join(lines)
    if verbose:
        print("\n" + report)
    return rows, report


# ---------------------------------------------------------------------------
# Input: command-line flags, or interactive prompts if no flags are given
# ---------------------------------------------------------------------------

def _ask_mode():
    print("Oracle mode:")
    print("  [1] preprocess  -- build full 2^n truth table once (small n, fast queries)")
    print("  [2] dynamic     -- answer queries on the fly from the k-sparse spectrum")
    choice = input("Choose 1 or 2 [2]: ").strip()
    return "preprocess" if choice == "1" else "dynamic"


def _ask_int(label, default):
    raw = input(f"{label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  (not an integer; using default {default})")
        return default


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--mode", choices=["preprocess", "dynamic"], default="dynamic")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--k-start", type=int, default=4)
    p.add_argument("--k-end", type=int, default=10)
    p.add_argument("--k-step", type=int, default=2)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        a = _parse_args()
        mode, n, k_start, k_end, k_step, trials, seed = (
            a.mode, a.n, a.k_start, a.k_end, a.k_step, a.trials, a.seed)
    else:
        mode = _ask_mode()
        n = _ask_int("n", 16)
        k_start = _ask_int("k start", 4)
        k_end = _ask_int("k end (inclusive)", 10)
        k_step = _ask_int("k step", 2)
        trials = _ask_int("trials", 10)
        seed = 0

    if not 1 <= n <= 62:
        sys.exit("n must be between 1 and 62 (frequencies are stored as int64).")
    if mode == "preprocess" and n > 24:
        sys.exit("preprocess mode builds a 2^n table; use dynamic mode for n > 24.")

    # Capture system configuration BEFORE the run, so timings are self-describing.
    sysconf = system_info()
    header = "System configuration (at run start)\n" + "-" * 36 + "\n" + sysconf
    print("\n" + header)

    rows, report = sweep(mode, n, k_start, k_end, k_step, trials, seed=seed)

    out_dir = "testresults"
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{mode}_n{n}_k{k_start}-{k_end}_{stamp}.txt")
    with open(out_path, "w") as fh:
        fh.write(header + "\n\n" + report + "\n")
    print(f"\nSaved results to {out_path}")
