#!/usr/bin/env python3
"""
submit_test.py
==============
Small smoke-test version of submit_QPUF_ntarg.py.

Goals:
  1. Validate the Haar-random controlled-unitary construction end-to-end on a
     small circuit (3 prec + 2 targ = 5 qubits) using the same MCM+reset
     pattern as the full job — so any failure here flags the same failure
     mode the larger job would hit.
  2. Print the transpiled gate count for ONE controlled-Haar-U, so the cost
     of the full N_PREC=32 job can be projected.
  3. AerSimulator pre-check before any QPU shots: print acceptance rate and
     compare against the eigenphase bins from np.linalg.eig(U).
  4. Submit to the QPU and log to the shared job_log.txt with
     circuit_type="QPUF_ntarg_test" so the analysis notebook can filter.

Reuses helpers from submit_QPUF_ntarg.py (haar_random_unitary,
build_full_circuit, append_job_log, _encode_unitary).
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DEVICE_NAME = "Forte-1"
DEVICE_ARN  = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
N_PREC      = 3
N_TARG      = 2             # total qubits = N_PREC + N_TARG = 5
N_SHOTS     = 100
SEED        = 7
TARGET_INIT_SEED = 99
SIM_SHOTS    = 2000         # local AerSim shots for the pre-check
SUBMIT_TO_QPU = True        # False → run sim + transpile probe only
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

from submit_QPUF_ntarg import (
    haar_random_unitary,
    build_full_circuit,
    append_job_log,
    _encode_unitary,
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


def main():
    # ── Build the Haar unitary ────────────────────────────────────────────────
    rng = np.random.default_rng(seed=SEED)
    d = 2 ** N_TARG
    U = haar_random_unitary(d, rng=rng)

    err = float(np.max(np.abs(U.conj().T @ U - np.eye(d))))
    print(f"DEVICE      : {DEVICE_NAME}")
    print(f"N_PREC      : {N_PREC}")
    print(f"N_TARG      : {N_TARG}  (U is {d}×{d})")
    print(f"Total qubits: {N_PREC + N_TARG}")
    print(f"SEED        : {SEED}")
    print(f"|U†U − I|   : {err:.2e}")

    # ── Build the full two-stage circuit ──────────────────────────────────────
    qc = build_full_circuit(N_PREC, N_TARG, U, target_init_seed=TARGET_INIT_SEED)
    print(f"\nLogical circuit: {qc.num_qubits} qubits, {qc.size()} high-level ops")

    # ── Ideal eigenphase bins from np.linalg.eig(U) ──────────────────────────
    eigvals = np.linalg.eig(U)[0]
    phases = np.mod(np.angle(eigvals) / (2 * np.pi), 1.0)
    ideal_bins = sorted({round(p * 2 ** N_PREC) % 2 ** N_PREC for p in phases})
    print(f"Ideal QPE bins from eig(U): {ideal_bins}  "
          f"(phases = {[round(float(p), 4) for p in phases]})")

    # ── Local AerSim pre-check ────────────────────────────────────────────────
    print("\n--- Local AerSimulator pre-check ---")
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        print("qiskit-aer not installed; skipping pre-check.")
    else:
        sim = AerSimulator()
        qc_sim = transpile(qc, sim)
        counts_sim = sim.run(qc_sim, shots=SIM_SHOTS).result().get_counts()

        agree = 0
        for k, v in counts_sim.items():
            m1, m2 = _parse_outcome(k, N_PREC)
            if _cyclic_distance(m1, m2, N_PREC) <= 1:
                agree += v
        total = sum(counts_sim.values())
        print(f"Sim acceptance (|m1-m2|_cyclic ≤ 1): "
              f"{agree}/{total} = {agree/total:.3f}")
        print("Top 5 sim outcomes (bitstring → m1, m2, count):")
        for k, v in sorted(counts_sim.items(), key=lambda x: -x[1])[:5]:
            m1, m2 = _parse_outcome(k, N_PREC)
            d_cyc = _cyclic_distance(m1, m2, N_PREC)
            print(f"  {k!r:24s}  m1={m1:3d}  m2={m2:3d}  dist={d_cyc:2d}  count={v}")

    # ── Stop here if not submitting ──────────────────────────────────────────
    if not SUBMIT_TO_QPU:
        print("\nSUBMIT_TO_QPU=False — skipping QPU submission.")
        return

    # ── QPU backend lookup ───────────────────────────────────────────────────
    try:
        from qiskit_braket_provider import BraketProvider
    except ImportError:
        print("\nERROR: qiskit-braket-provider not installed.")
        sys.exit(1)

    provider = BraketProvider()
    backend = None
    for b in provider.backends():
        dev = getattr(b, "_device", None)
        if dev and getattr(dev, "arn", None) == DEVICE_ARN:
            backend = b
            break
    if backend is None:
        print(f"\nERROR: backend for {DEVICE_ARN} not found.")
        sys.exit(1)

    device_n_qubits = getattr(backend, "num_qubits", None)
    if device_n_qubits is None:
        dev = getattr(backend, "_device", None)
        caps = getattr(dev, "properties", None) if dev else None
        device_n_qubits = caps.paradigm.qubitCount if caps else None
    if device_n_qubits is not None and qc.num_qubits > device_n_qubits:
        print(f"\nERROR: circuit needs {qc.num_qubits} qubits but {DEVICE_NAME} "
              f"only has {device_n_qubits}.")
        sys.exit(1)

    # ── Per-controlled-U probe ───────────────────────────────────────────────
    # Transpile a circuit with exactly one controlled-Haar-U to see how many
    # native gates each ctrl-U costs — useful for cost-projecting larger runs.
    print("\n--- Gates per controlled-Haar-U probe ---")
    probe = QuantumCircuit(N_TARG + 1, name="cU_probe")
    cU = UnitaryGate(U, label="U").control(1)
    probe.append(cU, list(range(N_TARG + 1)))
    probe_hw = transpile(probe, backend=backend, optimization_level=1)
    per_U_gates = probe_hw.size()
    per_U_depth = probe_hw.depth()
    print(f"One controlled-U (1 ctrl + {N_TARG} targets):")
    print(f"  Native gates : {per_U_gates}")
    print(f"  Depth        : {per_U_depth}")
    print(f"  Projected per-QPE-stage ctrl-U cost: "
          f"~{per_U_gates * N_PREC} gates ({N_PREC} ctrl-U gates per stage)")

    # ── Transpile full circuit ───────────────────────────────────────────────
    print("\n--- Full circuit transpile ---")
    qc_hw = transpile(qc, backend=backend, optimization_level=1)
    n_gates = qc_hw.size()
    print(f"Transpiled depth: {qc_hw.depth()}")
    print(f"Transpiled gates: {n_gates}")

    GATE_SHOT_LIMIT = 1_000_000
    n_shots = N_SHOTS
    if n_gates * n_shots > GATE_SHOT_LIMIT:
        n_shots = max(1, GATE_SHOT_LIMIT // n_gates)
        print(f"WARNING: gates × shots > {GATE_SHOT_LIMIT:,}; "
              f"reducing shots → {n_shots}")

    # ── Submit ────────────────────────────────────────────────────────────────
    print(f"\nSubmitting {n_shots} shots to {DEVICE_NAME} ...")
    job = backend.run(qc_hw, shots=n_shots)
    job_id = job.job_id()
    submitted_at = datetime.now(timezone.utc).isoformat()
    print(f"Job ID    : {job_id}")
    print(f"Timestamp : {submitted_at}")

    record = {
        "job_id":            job_id,
        "datetime":          submitted_at,
        "qpu":               DEVICE_NAME,
        "device_arn":        DEVICE_ARN,
        "circuit_type":      "QPUF_ntarg_test",
        "n_prec":            N_PREC,
        "n_targ":            N_TARG,
        "n_shots":           n_shots,
        "n_shots_requested": N_SHOTS,
        "n_gates":           n_gates,
        "per_U_gates":       per_U_gates,
        "per_U_depth":       per_U_depth,
        "seed":              SEED,
        "target_init_seed":  TARGET_INIT_SEED,
        "unitary":           _encode_unitary(U),
    }
    append_job_log(record)


if __name__ == "__main__":
    main()
