#!/usr/bin/env python3
"""
submit_job.py
=============
Submits the two-stage QPE QPUF circuit to IQM Garnet on AWS.
On success, appends a JSON record to `job_results/jobs_log.txt` for
retrieval by checkRetrieve_job.py and result_analysis.ipynb.
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DEVICE_NAME = "Garnet"
DEVICE_ARN  = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
N_SHOTS     = 500
N_PREC      = 9      # precision qubits per QPE stage
SEED        = 42     # RNG seed for Haar-random unitary
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate

JOB_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "jobs_log.txt")


# ── Helper functions ───────────────────────────────────────────────────────────

def haar_random_1qubit_matrix(rng=None):
    """
    Returns (U, angles) where U is a 2×2 Haar-random unitary matrix
    via ZYZ Euler decomposition: U = Rz(phi) @ Ry(theta) @ Rz(lam).

    Sampling:
      phi, lam ~ Uniform[0, 2π)
      theta = 2 * arccos(sqrt(u)),  u ~ Uniform[0, 1]   (Haar-correct)
    """
    if rng is None:
        rng = np.random.default_rng()

    phi   = rng.uniform(0, 2 * np.pi)
    lam   = rng.uniform(0, 2 * np.pi)
    u     = rng.uniform(0, 1)
    theta = 2 * np.arccos(np.sqrt(u))

    def Rz(a):
        return np.array([[np.exp(-1j * a / 2), 0.0],
                         [0.0,                 np.exp(1j * a / 2)]])

    def Ry(a):
        return np.array([[ np.cos(a / 2), -np.sin(a / 2)],
                         [ np.sin(a / 2),  np.cos(a / 2)]])

    U = Rz(phi) @ Ry(theta) @ Rz(lam)
    return U, dict(phi=phi, theta=theta, lam=lam)


def build_qpe_circuit(n_prec: int, angles: dict) -> QuantumCircuit:
    """
    Build a 1-qubit QPE sub-circuit using direct controlled-RzRyRz gates.

    U = Rz(phi) @ Ry(theta) @ Rz(lam)  (Haar-random, ZYZ decomposition)
    U^(2^k) is realised by repeating the gate sequence 2^k times.

    Qubit layout: [prec[0], ..., prec[n_prec-1], target]
    """
    n_targ = 1
    total  = n_prec + n_targ

    qc   = QuantumCircuit(total, name='QPE')
    prec = list(range(n_prec))
    targ = n_prec

    phi   = angles['phi']
    theta = angles['theta']
    lam   = angles['lam']

    qc.h(prec)

    for k in range(n_prec):
        ctrl = prec[k]
        for _ in range(2 ** k):
            qc.crz(lam,   ctrl, targ)
            qc.cry(theta, ctrl, targ)
            qc.crz(phi,   ctrl, targ)

    iqft = QFTGate(n_prec).inverse()
    qc.append(iqft, prec)

    return qc


def build_full_circuit(n_prec: int, angles: dict) -> QuantumCircuit:
    """Build the two-stage QPE circuit with mid-circuit measurement."""
    n_targ = 1

    targ_reg  = QuantumRegister(n_targ, 'target')
    prec1_reg = QuantumRegister(n_prec, 'prec1')
    prec2_reg = QuantumRegister(n_prec, 'prec2')
    c1        = ClassicalRegister(n_prec, 'c1')
    c2        = ClassicalRegister(n_prec, 'c2')

    qc = QuantumCircuit(targ_reg, prec1_reg, prec2_reg, c1, c2)

    # Target: random single-qubit initial state
    init_rng = np.random.default_rng(seed=99)
    theta0, phi0 = init_rng.uniform(0, np.pi), init_rng.uniform(0, 2 * np.pi)
    qc.ry(theta0, targ_reg[0])
    qc.rz(phi0,   targ_reg[0])

    # Stage 1 QPE
    qpe1 = build_qpe_circuit(n_prec, angles)
    qc.append(qpe1, list(prec1_reg) + list(targ_reg))

    # Mid-circuit measurement → collapses target onto eigenstate of U
    qc.measure(prec1_reg, c1)
    qc.barrier(label='collapse')

    # Stage 2 QPE on the collapsed target
    qpe2 = build_qpe_circuit(n_prec, angles)
    qc.append(qpe2, list(prec2_reg) + list(targ_reg))

    # Final measurement
    qc.measure(prec2_reg, c2)

    return qc


def append_job_log(record: dict):
    """Append a JSON record (one per line) to jobs_log.txt."""
    os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f"Job record written to: {LOG_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(seed=SEED)
    unitary, angles = haar_random_1qubit_matrix(rng=rng)

    print(f"Device      : {DEVICE_NAME}  ({DEVICE_ARN})")
    print(f"N_PREC      : {N_PREC}")
    print(f"N_SHOTS     : {N_SHOTS}")
    print(f"SEED        : {SEED}")
    print(f"Euler angles: phi={angles['phi']:.6f}  "
          f"theta={angles['theta']:.6f}  lam={angles['lam']:.6f}")

    qc = build_full_circuit(N_PREC, angles)
    print(f"\nCircuit qubits : {qc.num_qubits}  (IQM Garnet has 20)")

    try:
        from qiskit_braket_provider import BraketProvider
    except ImportError:
        print("\nERROR: qiskit-braket-provider not installed.")
        print("       pip install qiskit-braket-provider")
        sys.exit(1)

    provider    = BraketProvider()
    iqm_backend = provider.get_backend(DEVICE_NAME)

    print(f"\nTranspiling for {iqm_backend.name} ...")
    qc_hw = transpile(qc, backend=iqm_backend, optimization_level=1)
    print(f"Transpiled depth : {qc_hw.depth()}")
    print(f"CZ gates         : {qc_hw.count_ops().get('cz', 0)}")

    print(f"\nSubmitting {N_SHOTS} shots to {DEVICE_NAME} ...")
    job          = iqm_backend.run(qc_hw, shots=N_SHOTS)
    job_id       = job.job_id()
    submitted_at = datetime.now(timezone.utc).isoformat()

    print(f"Job submitted successfully.")
    print(f"Job ID    : {job_id}")
    print(f"Timestamp : {submitted_at}")

    record = {
        "job_id":     job_id,
        "datetime":   submitted_at,
        "device":     DEVICE_NAME,
        "device_arn": DEVICE_ARN,
        "n_prec":     N_PREC,
        "n_shots":    N_SHOTS,
        "seed":       SEED,
        "angles": {
            "phi":   angles['phi'],
            "theta": angles['theta'],
            "lam":   angles['lam'],
        },
    }
    append_job_log(record)


if __name__ == "__main__":
    main()