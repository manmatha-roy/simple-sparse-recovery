"""
Driver for exact Fourier-sparse recovery: sweep k, run both routines, report
per-routine TIMING and success rate.

Run:  python3 main.py

Prompts (press Enter to accept the shown default):
  - oracle mode : preprocess (build 2^n table once) or dynamic (evaluate on the fly)
  - n           : ambient dimension                     [default 16]
  - k start     : first sparsity in the sweep           [default 4]
  - k end       : last sparsity in the sweep, INCLUSIVE  [default 10]
  - k step      : increment                              [default 2]
  - trials      : trials per (n, k) config               [default 10]

For each k the driver runs Algorithm 1 and Algorithm 2 on the SAME random
k-sparse instances and reports, per routine:
    time (s)  -- average recovery time (query-evaluation time excluded)
    success   -- fraction of trials with exact recovery

The final table is also saved to  testresults/<timestamped>.txt.
"""

import os
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

def run_config(n, k, trials, mode, rng):
    a1_time = a2_time = 0.0
    a1ok = a2ok = 0

    for _ in range(trials):
        support, coeffs, ts = make_sparse_spectrum(n, k, rng)

        o1 = Oracle(n, support, coeffs, mode)
        t0 = time.perf_counter()
        r1 = algorithm1(o1, n, k, rng)
        a1_time += (time.perf_counter() - t0) - o1.oracle_time
        a1ok += exact(r1, ts)

        o2 = Oracle(n, support, coeffs, mode)
        t0 = time.perf_counter()
        r2 = algorithm2(o2, n, k, rng)
        a2_time += (time.perf_counter() - t0) - o2.oracle_time
        a2ok += exact(r2, ts)

    T = trials
    return {
        "n": n, "k": k, "trials": T,
        "a1_time": a1_time / T, "a1_success": a1ok / T,
        "a2_time": a2_time / T, "a2_success": a2ok / T,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(mode, n, k_start, k_end, k_step, trials, seed=0, verbose=True):
    """Run the sweep. k_end is INCLUSIVE. Returns (rows, report_text)."""
    rng = np.random.default_rng(seed)
    ks = list(range(k_start, k_end + 1, k_step))

    lines = []
    lines.append(f"Oracle mode: {mode.upper()}   n={n}   "
                 f"k = {k_start}..{k_end} step {k_step}   {trials} trials/config")
    lines.append("")
    lines.append(f"{'k':>3} | {'A1 time(s)':>11} {'A1 succ':>8} "
                 f"| {'A2 time(s)':>11} {'A2 succ':>8}")
    lines.append("-" * 50)

    rows = []
    for k in ks:
        if k > (1 << n):
            lines.append(f"{k:>3} | skipped: k > 2^n")
            continue
        if k * k >= (1 << n):
            lines.append(f"{k:>3} | (warning: k^2 >= 2^n, past useful sparse regime)")
        r = run_config(n, k, trials, mode, rng)
        rows.append(r)
        lines.append(f"{k:>3} | {r['a1_time']:>11.4f} {r['a1_success']:>7.0%} "
                     f"| {r['a2_time']:>11.4f} {r['a2_success']:>7.0%}")

    report = "\n".join(lines)
    if verbose:
        print("\n" + report)
    return rows, report


# ---------------------------------------------------------------------------
# Prompts
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


if __name__ == "__main__":
    mode = _ask_mode()
    n = _ask_int("n", 16)
    k_start = _ask_int("k start", 4)
    k_end = _ask_int("k end (inclusive)", 10)
    k_step = _ask_int("k step", 2)
    trials = _ask_int("trials", 10)

    # Capture system configuration BEFORE the run, so timings are self-describing.
    sysconf = system_info()
    header = "System configuration (at run start)\n" + "-" * 36 + "\n" + sysconf
    print("\n" + header)

    rows, report = sweep(mode, n, k_start, k_end, k_step, trials)

    # Save system config + results table to testresults/<timestamp>.txt
    out_dir = "testresults"
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{mode}_n{n}_k{k_start}-{k_end}_{stamp}.txt")
    with open(out_path, "w") as fh:
        fh.write(header + "\n\n" + report + "\n")
    print(f"\nSaved results to {out_path}")
