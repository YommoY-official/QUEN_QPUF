#!/usr/bin/env python3
"""
submit_test.py
==============
Two-part check for submit_QPUF_ntarg.py:

  1. Production circuit cost preview — build the full circuit with the
     N_PREC / N_TARG / seeds defined in submit_QPUF_ntarg.py, transpile
     against Forte-1, and print depth + gate count. This shows what the
     real job will actually cost on the QPU.

  2. Small-scale AerSim smoke test — build a tiny 3 prec + 2 targ version
     with the same MCM+reset construction and run it on AerSimulator to
     confirm the circuit compiles and executes. No QPU submission, no
     logging — the result is only used to check the construction runs
     without errors.
"""

# ── CONFIGURATION (small-scale AerSim test) ───────────────────────────────────
SMOKE_N_PREC    = 3
SMOKE_N_TARG    = 2
SMOKE_SEED      = 7
SMOKE_INIT_SEED = 99
SMOKE_SIM_SHOTS = 100
# ──────────────────────────────────────────────────────────────────────────────

import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

from submit_QPUF_ntarg import (
    DEVICE_NAME,
    DEVICE_ARN,
    RES_ARN,
    N_PREC           as PROD_N_PREC,
    N_TARG           as PROD_N_TARG,
    N_SHOTS          as PROD_N_SHOTS,
    SEED             as PROD_SEED,
    TARGET_INIT_SEED as PROD_INIT_SEED,
    haar_random_unitary,
    build_full_circuit,
)


def _parse_outcome(bitstring: str, n_prec: int) -> tuple[int, int]:
    """Parse a counts key into (m1, m2). Handles both 'c2 c1' and compact layouts."""
    parts = bitstring.split(" ")
    if len(parts) == 2:
        return int(parts[1], 2), int(parts[0], 2)
    return int(bitstring[n_prec:], 2), int(bitstring[:n_prec], 2)


def _cyclic_distance(a: int, b: int, n_prec: int) -> int:
    M = 2 ** n_prec
    diff = abs(a - b)
    return min(diff, M - diff)


def _resolve_backend():
    try:
        from qiskit_braket_provider import BraketProvider
    except ImportError:
        print("\nERROR: qiskit-braket-provider not installed.")
        sys.exit(1)

    provider = BraketProvider()
    for b in provider.backends():
        dev = getattr(b, "_device", None)
        if dev and getattr(dev, "arn", None) == DEVICE_ARN:
            return b
    print(f"\nERROR: backend for {DEVICE_ARN} not found.")
    sys.exit(1)


def production_cost_preview():
    """Build + transpile the production-size circuit and report depth/gates."""
    print("=" * 72)
    print("PRODUCTION CIRCUIT COST PREVIEW (from submit_QPUF_ntarg.py config)")
    print("=" * 72)
    print(f"DEVICE      : {DEVICE_NAME}")
    print(f"RES_ARN     : {RES_ARN}")
    print(f"N_PREC      : {PROD_N_PREC}")
    print(f"N_TARG      : {PROD_N_TARG}")
    print(f"Total qubits: {PROD_N_PREC + PROD_N_TARG}")
    print(f"SEED        : {PROD_SEED}")

    rng = np.random.default_rng(seed=PROD_SEED)
    d = 2 ** PROD_N_TARG
    U = haar_random_unitary(d, rng=rng)

    qc = build_full_circuit(PROD_N_PREC, PROD_N_TARG, U,
                            target_init_seed=PROD_INIT_SEED)
    print(f"Logical ops : {qc.size()}")

    backend = _resolve_backend()

    # IonQ Forte's qiskit target has no equivalence path from UnitaryGate to
    # gpi/gpi2/rzz and doesn't advertise 'reset'. Transpile to rz/rx/rxx
    # (rxx ≈ IonQ MS), so gate/depth counts roughly match native cost; Braket
    # Direct's server-side compiler maps to gpi/gpi2/rzz at submission time.
    ionq_basis = ['rz', 'rx', 'rxx', 'measure', 'reset']

    print("\n--- One controlled-Haar-U probe ---")
    probe = QuantumCircuit(PROD_N_TARG + 1, name="cU_probe")
    cU = UnitaryGate(U, label="U").control(1)
    probe.append(cU, list(range(PROD_N_TARG + 1)))
    probe_hw = transpile(probe, basis_gates=ionq_basis, optimization_level=1)
    per_U_gates = probe_hw.size()
    per_U_depth = probe_hw.depth()
    per_U_2q    = probe_hw.num_nonlocal_gates()
    print(f"Native gates : {per_U_gates}")
    print(f"2-qubit gates: {per_U_2q}")
    print(f"Depth        : {per_U_depth}")
    print(f"Projected per-QPE-stage ctrl-U cost: "
          f"~{per_U_gates * PROD_N_PREC} gates "
          f"(~{per_U_2q * PROD_N_PREC} 2-qubit gates) "
          f"({PROD_N_PREC} ctrl-U gates per stage)")

    print("\n--- Full production circuit transpile (this may take a minute) ---")
    qc_hw = transpile(qc, basis_gates=ionq_basis, optimization_level=1)
    n_total = qc_hw.size()
    n_2q    = qc_hw.num_nonlocal_gates()
    print(f"Transpiled depth   : {qc_hw.depth()}")
    print(f"Transpiled gates   : {n_total}")
    print(f"2-qubit gates      : {n_2q}")
    print(f"Op breakdown       : {dict(qc_hw.count_ops())}")
    print("\n--- Total number of Gates ---")
    print(f"Total gates        : {n_total * PROD_N_SHOTS}")
    print(f"Total 2-qubit gates: {n_2q * PROD_N_SHOTS}")


def small_scale_smoke_test():
    """Run the small QPUF circuit on AerSimulator to confirm it executes."""
    print("\n" + "=" * 72)
    print("SMALL-SCALE AERSIM SMOKE TEST")
    print("=" * 72)
    rng = np.random.default_rng(seed=SMOKE_SEED)
    d = 2 ** SMOKE_N_TARG
    U = haar_random_unitary(d, rng=rng)

    err = float(np.max(np.abs(U.conj().T @ U - np.eye(d))))
    print(f"N_PREC      : {SMOKE_N_PREC}")
    print(f"N_TARG      : {SMOKE_N_TARG}  (U is {d}×{d})")
    print(f"Total qubits: {SMOKE_N_PREC + SMOKE_N_TARG}")
    print(f"SEED        : {SMOKE_SEED}")
    print(f"|U†U − I|   : {err:.2e}")

    qc = build_full_circuit(SMOKE_N_PREC, SMOKE_N_TARG, U,
                            target_init_seed=SMOKE_INIT_SEED)
    print(f"Logical circuit: {qc.num_qubits} qubits, {qc.size()} high-level ops")

    eigvals = np.linalg.eig(U)[0]
    phases = np.mod(np.angle(eigvals) / (2 * np.pi), 1.0)
    ideal_bins = sorted({round(p * 2 ** SMOKE_N_PREC) % 2 ** SMOKE_N_PREC for p in phases})
    print(f"Ideal QPE bins from eig(U): {ideal_bins}  "
          f"(phases = {[round(float(p), 4) for p in phases]})")

    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        print("qiskit-aer not installed; cannot run smoke test.")
        return

    sim = AerSimulator()
    qc_sim = transpile(qc, sim)
    counts_sim = sim.run(qc_sim, shots=SMOKE_SIM_SHOTS).result().get_counts()

    agree = 0
    for k, v in counts_sim.items():
        m1, m2 = _parse_outcome(k, SMOKE_N_PREC)
        if _cyclic_distance(m1, m2, SMOKE_N_PREC) <= 1:
            agree += v
    total = sum(counts_sim.values())
    print(f"Sim acceptance (|m1-m2|_cyclic ≤ 1): "
          f"{agree}/{total} = {agree/total:.3f}")
    print("Top 5 sim outcomes (bitstring → m1, m2, count):")
    for k, v in sorted(counts_sim.items(), key=lambda x: -x[1])[:5]:
        m1, m2 = _parse_outcome(k, SMOKE_N_PREC)
        d_cyc = _cyclic_distance(m1, m2, SMOKE_N_PREC)
        print(f"  {k!r:24s}  m1={m1:3d}  m2={m2:3d}  dist={d_cyc:2d}  count={v}")

    print("\nSmoke test completed without errors.")


def main():
    production_cost_preview()
    small_scale_smoke_test()


if __name__ == "__main__":
    main()
