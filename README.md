# Fourier-Sparse Recovery

Exact recovery of Fourier-sparse functions over the Boolean cube via random subspace restriction.

Given oracle access to a function `f : F_2^n -> R` with at most `k` nonzero Walsh–Hadamard coefficients, these routines recover all of them exactly. Two routines are provided:

- **Algorithm 1** (warm-up): restrict to a subspace large enough to isolate every frequency, decode coordinate-wise. `Õ(n k²)` queries.
- **Algorithm 2** (main): restrict to a small subspace each round, keep singleton buckets via a BLR test, peel, repeat. `Õ(n k)` queries.

Both handle general real-valued functions (no bounded-degree or Boolean-range assumption).

## Files

```
oracle.py       shared: oracle (both modes), instance generation, FWHT, helpers
algorithm1.py   Algorithm 1
algorithm2.py   Algorithm 2
sysinfo.py      captures system configuration at run time
main.py         driver: sweep k, time both routines, save results
```

## Install

```bash
pip install -r requirements.txt
```

`numpy` is required; `psutil` is optional (gives clock speed and memory in the
system-config report — the code runs without it).

## Run

From the directory containing all files:

```bash
python3 main.py
```

Prompts (press Enter for the default):

- **oracle mode** — `preprocess` (build the `2^n` table once; small `n`) or `dynamic` (evaluate on the fly; scales to large `n`)
- **n** — dimension (default `16`)
- **k start / k end / k step** — sparsity sweep, `k end` inclusive (default `4..10` step `2`)
- **trials** — trials per config (default `10`)

## Output

Before running, the driver records the **system configuration** — CPU, cores,
clock speed, current CPU load, and total / in-use memory at run start — so the
timing numbers are self-describing. For each `k` it then reports, per routine:

- **time (s)** — average recovery time (function-evaluation time excluded)
- **success** — fraction of trials with exact recovery

The full report (system config + table) is printed and saved to
`testresults/<mode>_n<n>_k<range>_<timestamp>.txt`.

Example:

```
System configuration (at run start)
------------------------------------
platform    : Linux-6.18.5-x86_64-with-glibc2.39
processor   : x86_64
python      : 3.12.3
cpu cores   : 1 logical
clock speed : 2800 MHz current
current load: 0% CPU
memory total: 3.9 GB
memory used : 0.2 GB (6% in use) at run start

Oracle mode: PREPROCESS   n=16   k = 4..10 step 2   10 trials/config

  k |  A1 time(s)  A1 succ |  A2 time(s)  A2 succ
--------------------------------------------------
  4 |      0.0029    100% |      0.0049    100%
  6 |      0.0108     90% |      0.0055     80%
  8 |      0.0111    100% |      0.0097     90%
 10 |      0.0436    100% |      0.0104     70%
```

## Notes

- Exact, noiseless, exactly-`k`-sparse recovery.
- `preprocess` mode builds a `2^n` table, so it is memory-bound (roughly `n <= 22`); use `dynamic` for larger `n`.
- `algorithm1(dim_slack=...)` and `algorithm2(iso_const=...)` trade success probability against query/time cost. Defaults target moderate (≥ 2/3) success; raise them for high-probability recovery.
