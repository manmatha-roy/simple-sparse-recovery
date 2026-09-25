"""
Experiments reported in the paper.  Run:  python3 experiments.py [--quick | --tables-only]

Writes, to results/:
  experiments.json   raw numbers
  tables.tex         LaTeX tables (booktabs), ready to \\input into the paper

Experiments (all times exclude function-evaluation time):
  E1  scaling in k      n = 20, k = 4..256, both algorithms, preprocess oracle
  E2  scaling in n      k = 32, n = 12..60, both algorithms, dynamic oracle
  E3  iso_const sweep   n = 24, k = 64, Algorithm 2, iso_const = 3..100
  E4  structured inputs n = 24, k = 64, Algorithm 2, four support families
  E5  small iso_const   n = 24, k in {64, 256}, iso_const in {1, 1.5, 2, 3},
                        with R and R+2 rounds

E1-E4 share one random generator with seed 0, and E5 uses seed 7, so the
default run reproduces the paper's numbers exactly on the same numpy version.
--quick uses fewer trials and smaller sizes, as a smoke test (about 10 s).
--tables-only rebuilds results/tables.tex from results/experiments.json.
"""

import json
import os
import sys
import time

import numpy as np

from oracle import (Oracle, make_sparse_spectrum, random_full_rank_basis,
                    enumerate_subspace)
from algorithm1 import algorithm1
from algorithm2 import algorithm2, num_rounds
from sysinfo import system_info


def exact(rec, ts, tol=1e-6):
    return set(rec) == set(ts) and all(abs(rec[a] - ts[a]) <= tol for a in ts)


def run(alg, n, k, sup, co, mode, rng, ts, **kw):
    o = Oracle(n, sup, co, mode)
    t0 = time.perf_counter()
    rec = alg(o, n, k, rng, **kw)
    return (time.perf_counter() - t0) - o.oracle_time, o.count, exact(rec, ts)


def config(n, k, trials, mode, rng, algs=("a1", "a2"), **kw2):
    """Run the given algorithms on the same `trials` random instances."""
    out = {}
    insts = [make_sparse_spectrum(n, k, rng) for _ in range(trials)]
    for name in algs:
        alg = algorithm1 if name == "a1" else algorithm2
        kw = kw2 if name == "a2" else {}
        T = Q = S = 0
        for sup, co, ts in insts:
            t, q, s = run(alg, n, k, sup, co, mode, rng, ts, **kw)
            T += t; Q += q; S += s
        out[name] = {"time": T / trials, "queries": Q / trials,
                     "success": S / trials}
    return out


# ---------------------------------------------------------------------------
# Structured support families for E4
# ---------------------------------------------------------------------------

def structured_families(n, m):
    k = 1 << m

    def subspace_equal(rng):
        sup = enumerate_subspace(random_full_rank_basis(n, m, rng), m)
        return sup, np.ones(k)

    def subspace_signed(rng):
        sup = enumerate_subspace(random_full_rank_basis(n, m, rng), m)
        return sup, rng.choice([-1.0, 1.0], size=k)

    def low_degree(rng):
        cand = [1 << i for i in range(n)] + [(1 << i) | (1 << j)
                                             for i in range(n) for j in range(i)]
        sup = rng.choice(np.array(cand, dtype=np.int64), size=k, replace=False)
        return sup, np.ones(k)

    def int_coeffs(rng):
        sup = rng.choice(1 << n, size=k, replace=False).astype(np.int64)
        co = rng.integers(1, 4, size=k).astype(float) * rng.choice([-1.0, 1.0], size=k)
        return sup, co

    return [("Subspace, equal coefficients", subspace_equal),
            ("Subspace, random signs", subspace_signed),
            (r"Degree $\le 2$, equal coefficients", low_degree),
            ("Random support, small integer coefficients", int_coeffs)]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def main(quick=False):
    res = {"system": system_info(), "quick": quick}
    rng = np.random.default_rng(0)

    ks = (4, 8, 16) if quick else (4, 8, 16, 32, 64, 128, 256)
    tr = 3 if quick else 20
    print(f"E1: scaling in k (n=20, {tr} trials)", flush=True)
    res["E1"] = []
    for k in ks:
        r = config(20, k, tr, "preprocess", rng)
        res["E1"].append({"k": k, **r})
        print(f"  k={k:4d}  A1 {r['a1']}  A2 {r['a2']}", flush=True)

    ns = (12, 16) if quick else (12, 16, 20, 24, 32, 40, 48, 60)
    print(f"E2: scaling in n (k=32, {tr} trials)", flush=True)
    res["E2"] = []
    for n in ns:
        r = config(n, 32, tr, "dynamic", rng)
        res["E2"].append({"n": n, **r})
        print(f"  n={n:3d}  A1 {r['a1']}  A2 {r['a2']}", flush=True)

    isos = (3, 100) if quick else (3, 5, 10, 20, 50, 100)
    tr3 = 3 if quick else 50
    print(f"E3: iso_const sweep (n=24, k=64, {tr3} trials)", flush=True)
    res["E3"] = []
    for iso in isos:
        r = config(24, 64, tr3, "dynamic", rng, algs=("a2",), iso_const=iso)
        res["E3"].append({"iso_const": iso, **r["a2"]})
        print(f"  iso={iso:4}  {r['a2']}", flush=True)

    n, m = 24, 6
    k = 1 << m
    print(f"E4: structured supports (n={n}, k={k}, {tr3} trials)", flush=True)
    res["E4"] = []
    for name, gen in structured_families(n, m):
        S = Q = 0
        for _ in range(tr3):
            sup, co = gen(rng)
            ts = {int(a): float(c) for a, c in zip(sup, co)}
            _, q, s = run(algorithm2, n, k, sup, co, "dynamic", rng, ts)
            S += s; Q += q
        res["E4"].append({"family": name, "success": S / tr3, "queries": Q / tr3})
        print(f"  {name:45s} success={S / tr3:.2f}", flush=True)

    rng = np.random.default_rng(7)
    n = 24
    tr5 = 5 if quick else 100
    print(f"E5: small iso_const (n={n}, {tr5} trials)", flush=True)
    res["E5"] = []
    for k in ((64,) if quick else (64, 256)):
        for iso in (1, 1.5, 2, 3):
            for extra in (0, 2):
                S = Q = 0
                for _ in range(tr5):
                    sup, co, ts = make_sparse_spectrum(n, k, rng)
                    o = Oracle(n, sup, co, "dynamic")
                    rec = algorithm2(o, n, k, rng, iso_const=iso,
                                     rounds=num_rounds(k) + extra)
                    S += exact(rec, ts); Q += o.count
                res["E5"].append({"k": k, "iso_const": iso, "extra_rounds": extra,
                                  "success": S / tr5, "queries": Q / tr5})
                print(f"  k={k:4d} iso={iso:4} rounds=R+{extra}  "
                      f"success={S / tr5:.2f}  queries={Q / tr5:.0f}", flush=True)
    return res


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _q(x):
    return f"{x:,.0f}".replace(",", "{,}")


def _p(x):
    return f"{100 * x:.0f}\\%"


def latex_tables(res, note=""):
    out = [f"% Generated by experiments.py. {note}",
           "% Requires \\usepackage{booktabs}.", ""]

    # Table 1: scaling in k
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Scaling in the sparsity $k$, with $n = 20$. Averages over "
            r"20 random instances per row; times exclude function evaluation.}",
            r"\label{tab:scaling-k}",
            r"\begin{tabular}{r rrr rrr}", r"\toprule",
            r" & \multicolumn{3}{c}{Algorithm~\ref{algo:warmup}} "
            r"& \multicolumn{3}{c}{Algorithm~\ref{alg:main}} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"$k$ & queries & time (s) & success & queries & time (s) & success \\",
            r"\midrule"]
    for r in res["E1"]:
        a1, a2 = r["a1"], r["a2"]
        out.append(f"{r['k']} & {_q(a1['queries'])} & {a1['time']:.3f} & {_p(a1['success'])} "
                   f"& {_q(a2['queries'])} & {a2['time']:.3f} & {_p(a2['success'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table 2: scaling in n
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Scaling in the dimension $n$, with $k = 32$. Averages over "
            r"20 random instances per row.}",
            r"\label{tab:scaling-n}",
            r"\begin{tabular}{r rrr rrr}", r"\toprule",
            r" & \multicolumn{3}{c}{Algorithm~\ref{algo:warmup}} "
            r"& \multicolumn{3}{c}{Algorithm~\ref{alg:main}} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"$n$ & queries & time (s) & success & queries & time (s) & success \\",
            r"\midrule"]
    for r in res["E2"]:
        a1, a2 = r["a1"], r["a2"]
        out.append(f"{r['n']} & {_q(a1['queries'])} & {a1['time']:.3f} & {_p(a1['success'])} "
                   f"& {_q(a2['queries'])} & {a2['time']:.3f} & {_p(a2['success'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table 3: iso_const (E3 and E5 combined)
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Effect of the bucket constant (\texttt{iso\_const}; the analysis "
            r"uses $100$) on Algorithm~\ref{alg:main}, with $n = 24$. Top: $k = 64$, "
            r"$R$ rounds, 50 instances per row. Bottom: $R$ or $R+2$ rounds, "
            r"100 instances per row. Values $1.5$ and $2$ give the same subspace "
            r"size after rounding to a power of $2$.}",
            r"\label{tab:iso-const}",
            r"\begin{tabular}{r r rr rr}", r"\toprule",
            r"\multicolumn{6}{l}{\emph{$k = 64$, $R$ rounds}} \\",
            r"\texttt{iso\_const} & & queries & success & & \\",
            r"\midrule"]
    for r in res["E3"]:
        out.append(f"{r['iso_const']} & & {_q(r['queries'])} & {_p(r['success'])} & & \\\\")
    out += [r"\midrule",
            r"\multicolumn{6}{l}{\emph{Small constants}} \\",
            r" & & \multicolumn{2}{c}{$k = 64$} & \multicolumn{2}{c}{$k = 256$} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"\texttt{iso\_const} & rounds & queries & success & queries & success \\",
            r"\midrule"]
    e5 = {(r["k"], r["iso_const"], r["extra_rounds"]): r for r in res["E5"]}
    for iso in (1, 1.5, 2, 3):
        for extra in (0, 2):
            cells = []
            for k in (64, 256):
                r = e5.get((k, iso, extra))
                cells += ([_q(r["queries"]), _p(r["success"])] if r else ["--", "--"])
            rounds = "$R$" if extra == 0 else f"$R+{extra}$"
            out.append(f"{iso} & {rounds} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table 4: structured supports
    out += [r"\begin{table}[t]", r"\centering",
            r"\caption{Structured inputs for Algorithm~\ref{alg:main}, with $n = 24$ "
            r"and $k = 64$. 50 instances per row.}",
            r"\label{tab:structured}",
            r"\begin{tabular}{l rr}", r"\toprule",
            r"Support and coefficients & queries & success \\", r"\midrule"]
    for r in res["E4"]:
        out.append(f"{r['family']} & {_q(r['queries'])} & {_p(r['success'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        with open("results/experiments.json") as fh:
            res = json.load(fh)
        with open("results/tables.tex", "w") as fh:
            fh.write(latex_tables(res))
        sys.exit("Rewrote results/tables.tex from results/experiments.json")
    quick = "--quick" in sys.argv
    res = main(quick=quick)
    os.makedirs("results", exist_ok=True)
    suffix = "_quick" if quick else ""
    with open(f"results/experiments{suffix}.json", "w") as fh:
        json.dump(res, fh, indent=2)
    note = "QUICK MODE: smoke test only, not the paper's numbers." if quick else ""
    with open(f"results/tables{suffix}.tex", "w") as fh:
        fh.write(latex_tables(res, note))
    print(f"\nWrote results/experiments{suffix}.json and results/tables{suffix}.tex")
