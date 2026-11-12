#!/usr/bin/env python3
"""
validate_local.py -- free, local checks of the IQM sequential-QPUF programs.

The Braket LocalSimulator cannot run measure_ff / cc_prx ("Gate measure_ff is
not defined"), so the submitted OpenQASM itself is parsed back into ops and run
in Aer with measure_ff -> mid-circuit measure and cc_prx -> if_test. The
program text that is simulated is the exact text that submit_qpuf_mcm.py sends.

Checks (per d in 2,3):
  1. known answer (phase 1/2^d, target |1>): m1 = m2 = 1 with probability 1, all arms
  2. arm B: ancilla copy == re-read P1 with probability 1 (noiseless)
  3. noiseless P(m1,m2): arm A == arm B == arm Breuse (deferred-measurement theorem)
  4. arm A vs the full-circuit statevector reference (make_notes_figs.sim_counts,
     fixed circuit, per-stage bit reversal) -- validates routing + decode
"""
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../../docs/plot_ledger"))
from iqm_qpuf_common import *
from make_notes_figs import sim_counts

SHOTS = 100000; DEV = sys.argv[1] if len(sys.argv) > 1 else "garnet"
caps = load_caps(DEV); ok = True


def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")


def run(arm, d, U, init):
    p = build_program(arm, d, 1, U, init, caps)
    body, order = ops_from_qasm(to_qasm(p))            # the text that would be submitted
    assert order == p["order"] and len(body) == len(p["body"])
    meta = dict(arm=arm, d=d, measure=p["measure"], order=p["order"])
    M, agree = decode(simulate_ops(body, order, SHOTS), meta)
    return M, agree, p


for d in (2, 3):
    print(f"\n=== d={d}, n_t=1, {DEV} ({'real caps' if caps['real'] else 'PLACEHOLDER lattice'}) ===")
    Uk = known_phase_unitary(1, 1 / 2 ** d)
    for arm in ("A", "B", "Breuse"):
        M, ag, _ = run(arm, d, Uk, "ones")
        check(f"known answer arm {arm}: P(m1=1,m2=1)", M[1, 1] > 1 - 1e-9, f"= {M[1, 1]:.6f}")
        if ag is not None: check(f"known answer arm {arm}: ancilla == re-read P1", ag > 1 - 1e-9, f"= {ag:.6f}")
    U = haar_random_unitary(2, np.random.default_rng(100))
    Ms, ags = {}, {}
    for arm in ("A", "B", "Breuse"):
        Ms[arm], ags[arm], p = run(arm, d, U, 99)
    check("Haar arm B: ancilla == re-read P1", ags["B"] > 1 - 1e-9, f"= {ags['B']:.6f}")
    tol = 4 * np.sqrt(2 ** (2 * d) / SHOTS)             # generous TVD sampling bound
    tv = lambda a, b: 0.5 * np.abs(Ms[a] - Ms[b]).sum()
    ref = np.zeros((2 ** d, 2 ** d))
    for k, v in sim_counts(d, 1, U, 99, True, True).items(): ref[int(k[d:][::-1], 2), int(k[:d][::-1], 2)] += v
    for a, b in (("A", "B"), ("A", "Breuse")):
        check(f"noiseless TVD({a},{b}) within sampling", tv(a, b) < tol, f"= {tv(a, b):.4f} (tol {tol:.4f})")
    for arm in Ms:
        t = 0.5 * np.abs(Ms[arm] - ref).sum()
        check(f"arm {arm} vs full-circuit reference", t < tol, f"TVD = {t:.4f} (tol {tol:.4f})")
print("\nALL PASS" if ok else "\nFAILURES"); sys.exit(0 if ok else 1)
