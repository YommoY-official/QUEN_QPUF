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
  4. arm A vs the full-circuit statevector reference (reference_pairs: one
     statevector of the whole circuit, no routing) -- validates routing + decode
"""
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from iqm_qpuf_common import *
from iqm_qpuf_common import _group_of

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


print("\n=== smoke layout (nonadj), d=2, known answer, arm B ===")
_p = build_program("B", 2, 1, known_phase_unitary(1, 1 / 4), "ones", caps, layout="nonadj")
_adj_e = {tuple(sorted(e)) for e in caps["edges"]}
check("copy pairs are non-couplers", all(tuple(sorted(x)) not in _adj_e for x in zip(_p["p1_stage1"], _p["anc"])), f"{list(zip(_p['p1_stage1'], _p['anc']))}")
_b, _o = ops_from_qasm(to_qasm(_p)); _M, _ag = decode(simulate_ops(_b, _o, SHOTS), dict(arm="B", d=2, measure=_p["measure"], order=_p["order"]))
check("smoke: P(m1=1,m2=1)", _M[1, 1] > 1 - 1e-9, f"= {_M[1, 1]:.6f}"); check("smoke: ancilla == re-read P1", _ag > 1 - 1e-9)
check("smoke: dynamic-circuit rules", not check_dynamic_rules(_p, caps)[0])


print("\n=== T1 filter + idle calibration + analysis (SYNTHETIC T1 and flip rates; logic check only) ===")
from checkRetrieve import marginals, unconfuse, expected_agreement, load_cal
_c2 = dict(caps, t1={str(q): (3.0 if q in (17, 6) else 45.0) for q in caps["nodes"]})   # SYNTHETIC: 17 and 6 made bad
_sc = record_scores(_c2); _best = set(sorted(_sc, key=_sc.get)[:K_BEST])
_pt = build_program("B", 2, 1, known_phase_unitary(1, 1 / 4), "ones", _c2, layout="nonadj")
check("T1 score: record qubits are among the K best and avoid the synthetic-bad 17, 6", set(_pt["p1_stage1"] + _pt["anc"]) <= _best - {17, 6}, f"P1 {_pt['p1_stage1']} anc {_pt['anc']}")
_raw = {"provider": {"properties": {"one_qubit": {"4": {"T1": 1.2e-5, "T2": 6e-6, "T2_echo": 9e-6, "fRO": 0.98, "readout_error_0_to_1": 0.01, "readout_error_1_to_0": 0.03}}}}, "updatedAt": "2026-11-13T02:40:18Z"}
_lv = parse_live_props(_raw)
check("parse_live_props: units, echo vs Ramsey, timestamp", abs(_lv["t1"]["4"] - 12.0) < 1e-9 and abs(_lv["t2"]["4"] - 9.0) < 1e-9 and abs(_lv["t2_ramsey"]["4"] - 6.0) < 1e-9 and _lv["calibration_updatedAt"].startswith("2026-11-13"))
check("T1 filter: copy pairs still non-couplers, same group",
      all(tuple(sorted(x)) not in _adj_e and _group_of(caps, x[0]) == _group_of(caps, x[1]) for x in zip(_pt["p1_stage1"], _pt["anc"])))
rng = np.random.default_rng(5); E0, E1 = 0.02, 0.15          # asymmetric flips: a T1-like signature
def noisy(prog, flip_q):
    body, order = ops_from_qasm(to_qasm(prog)); raw = simulate_ops(body, order, SHOTS)
    keys = list(raw); X = np.array([[int(ch) for ch in k] for k in keys]); n = np.array([raw[k] for k in keys])
    X = np.repeat(X, n, axis=0); u = rng.random(X.shape)
    for q in flip_q:
        j = order.index(q); X[:, j] ^= np.where(X[:, j] == 1, u[:, j] < E1, u[:, j] < E0).astype(int)
    out = {}
    for row in X: k = "".join(map(str, row)); out[k] = out.get(k, 0) + 1
    return out
rec_q = _pt["p1_stage1"] + _pt["anc"]
cal = {}
for st, pc in idle_cal_programs(_pt).items():
    b, o = ops_from_qasm(to_qasm(pc)); assert pc["body"] and not any(x[0] in ("measure_ff", "cc_prx") for x in pc["body"])
    cnt = noisy(pc, rec_q); one = st == "1"
    for q in rec_q: cal.setdefault(q, {})[st] = sum(v for k, v in cnt.items() if (k[o.index(q)] == "1") != one) / SHOTS
check("idle cal recovers injected flip rates", all(abs(cal[q]["0"] - E0) < 0.005 and abs(cal[q]["1"] - E1) < 0.01 for q in rec_q), str({q: tuple(round(v, 3) for v in cal[q].values()) for q in rec_q}))
rb = dict(arm="B", d=2, measure=_pt["measure"], order=_pt["order"]); cnt = noisy(_pt, rec_q)
mg = marginals(cnt, rb); _, ag = decode(cnt, rb)
anc_q, p1_q = _pt["anc"], _pt["p1_stage1"]
ca = unconfuse(mg["m1_ancilla"], anc_q, cal, 2)
check("unconfuse recovers m1=1 (known answer)", ca[1] > 0.99, f"corrected P(m1=1) {ca[1]:.3f} vs raw {mg['m1_ancilla'][1]:.3f}")
ex = expected_agreement(rb, cal, ca)
check("predicted witness agreement matches the synthetic-noise measurement", abs(ex - ag) < 0.01, f"measured {ag:.3f} predicted {ex:.3f}")

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
    ref = reference_pairs(d, 1, U, 99)
    for a, b in (("A", "B"), ("A", "Breuse")):
        check(f"noiseless TVD({a},{b}) within sampling", tv(a, b) < tol, f"= {tv(a, b):.4f} (tol {tol:.4f})")
    for arm in Ms:
        t = 0.5 * np.abs(Ms[arm] - ref).sum()
        check(f"arm {arm} vs full-circuit reference", t < tol, f"TVD = {t:.4f} (tol {tol:.4f})")
print("\nALL PASS" if ok else "\nFAILURES"); sys.exit(0 if ok else 1)
