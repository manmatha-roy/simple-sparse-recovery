# Exact Fourier-Sparse Recovery Made Simple

Reference implementation for the paper *Exact Fourier-Sparse Recovery Made Simple*.

Given oracle access to a function `f : F_2^n -> R` with at most `k` nonzero
Walsh–Hadamard coefficients, these routines recover all of them exactly. Two
routines are provided, matching the two algorithms in the paper:

- **Algorithm 1** (warm-up, Theorem 2): restrict `f` to one random subspace of
  dimension `ceil(2 log2(4k))`, large enough that every frequency lands alone in
  its bucket (except with probability 1/16), then read each frequency off bit by
  bit. `(n+1) · 2^d ≤ 32 k² (n+1)` queries.
- **Algorithm 2** (main, Theorem 1): `floor(log4 k) + 1` rounds of iterative
  peeling. Round `r` restricts the residual to a random subspace with about
  `100 k / 2^r` buckets and decodes **every** nonempty bucket. There is no
  singleton test: each output is *added* to a dictionary, and a wrong output
  from a bucket with a collision is corrected in a later round.
  At most `400 k (n+1) = O(nk)` queries and `O(nk log k)` time,
  success probability at least 2/3.

In both algorithms, bit `i` of a frequency is decoded from the sign of
`fhat_{e_i+H}(γ) · fhat_{H}(γ)`: no linear algebra, no majority vote.
Both algorithms are **non-adaptive**: the query points depend only on the random
subspaces, never on the function values. Both handle general real-valued
functions (no bounded-degree or Boolean-range assumption).

## Files

```
oracle.py       shared: oracle (two modes), instance generation, FWHT,
                subspace helpers, bucket indexing
algorithm1.py   Algorithm 1 (warm-up)
algorithm2.py   Algorithm 2 (main algorithm)
tests.py        self-checks: bucket convention, exact recovery, query budgets,
                structured supports, non-adaptivity
sysinfo.py      captures the system configuration at run time
main.py         driver: sweep k, time both routines, save results
```

## Install

```bash
pip install -r requirements.txt
```

`numpy >= 2.0` is required (for `np.bitwise_count`). `psutil` is optional; it
adds clock speed and memory to the system-config report.

## Run

Self-checks (about a minute):

```bash
python3 tests.py
```

Experiments, interactively (press Enter for each default):

```bash
python3 main.py
```

or non-interactively, for reproducible runs:

```bash
python3 main.py --mode dynamic --n 16 --k-start 4 --k-end 10 --k-step 2 --trials 10 --seed 0
```

Options:

- **mode** — `preprocess` (build the `2^n` table once; `n ≤ 24`) or `dynamic`
  (evaluate `f` on the fly; any `n ≤ 62`)
- **n** — dimension (default `16`)
- **k start / k end / k step** — sparsity sweep, `k end` inclusive (default `4..10` step `2`)
- **trials** — trials per configuration (default `10`)
- **seed** — random seed (default `0`; command-line mode only)

## Output

Before running, the driver records the **system configuration** (CPU, cores,
clock speed, current CPU load, and total and in-use memory), so the timing numbers
are self-describing. For each `k` it then reports, per routine:

- **time (s)** — average recovery time, excluding function-evaluation time
- **queries** — average number of oracle queries
- **success** — fraction of trials with exact recovery

The full report is printed and saved to
`testresults/<mode>_n<n>_k<range>_<timestamp>.txt`.

Example (`--mode dynamic --n 16 --seed 0`):

```
Oracle mode: DYNAMIC   n=16   k = 4..10 step 2   10 trials/config   seed=0

   k | A1 time(s)  A1 queries  A1 succ | A2 time(s)  A2 queries  A2 succ
----------------------------------------------------------------------------
   4 |     0.0009        4352    100% |     0.0022       13056    100%
   6 |     0.0014       17408    100% |     0.0026       26112    100%
   8 |     0.0014       17408     90% |     0.0036       26112    100%
  10 |     0.0021       34816    100% |     0.0030       26112    100%
```

At small `k`, Algorithm 1 can use fewer queries than Algorithm 2, because its
bound `32 k² (n+1)` has a smaller constant than `400 k (n+1)`. Algorithm 2 wins
once `k` is larger than roughly 12; for example, at `n = 30, k = 100` Algorithm 1
uses about 8.1 million queries and Algorithm 2 about 0.95 million.

## Parameters

- `algorithm2(iso_const=100)` sets the number of buckets per round to about
  `iso_const · k / 2^r`. `100` is the paper's constant, which guarantees success
  probability above 0.88. Smaller values (for example `10`–`20`) use
  proportionally fewer queries and often still succeed in practice, but lose the
  guarantee. The `rounds` argument overrides the number of rounds.
- `algorithm1(dim_slack=0)` adds to the subspace dimension. Each extra unit
  doubles the queries and halves the collision probability. `0` is the paper's
  setting.

## Notes

- The setting is exact, noiseless, exactly `k`-sparse recovery, as in the paper.
  Values are compared with a tolerance of `1e-9` in place of exact arithmetic.
- The oracle only returns values of `f`. The residual `f − Σ D(α) χ_α` is
  subtracted by the algorithm in the bucket domain, so that cost is counted as
  algorithm time, as in the paper's analysis.
- `preprocess` mode builds a `2^n` table, so it is memory-bound; use `dynamic`
  for larger `n`.
