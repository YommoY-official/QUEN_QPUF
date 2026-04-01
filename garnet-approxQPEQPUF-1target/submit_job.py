#!/usr/bin/env python3
"""
submit_job.py
=============
Submits the two-stage QPE QPUF circuit to IonQ Forte on AWS Braket.
On success, appends a JSON record to `job_results/jobs_log.txt` for
retrieval by checkRetrieve_job.py and result_analysis.ipynb.

IonQ Forte is a trapped-ion device with all-to-all qubit connectivity,
so no custom routing layout is required. Mid-circuit measurements are
supported natively.
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DEVICE_NAME = "Forte-1"
DEVICE_ARN  = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
N_SHOTS     = 400
N_PREC      = 7       # precision qubits per QPE stage
SEED        = 33     # RNG seed for Haar-random unitary
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


def zyz_angles(M: np.ndarray):
    """
    Extract (phi, theta, lam) from a 2×2 SU(2) matrix M = Rz(phi)@Ry(theta)@Rz(lam).

    From the ZYZ matrix elements:
      M[0,0] = e^{-i(phi+lam)/2} cos(theta/2)  →  theta = 2*arccos(|M[0,0]|)
      M[0,1] = -e^{-i(phi-lam)/2} sin(theta/2)
    Solving:
      phi = -(angle(M[0,0]) + angle(-M[0,1]))
      lam =   angle(-M[0,1]) - angle(M[0,0])
    """
    a = M[0, 0]
    b = M[0, 1]
    cos_half = float(np.clip(np.abs(a), 0.0, 1.0))
    theta = 2.0 * np.arccos(cos_half)
    if np.abs(np.sin(theta / 2.0)) < 1e-10:
        # Near-identity: only phi+lam is determined; set lam=0
        phi = -2.0 * float(np.angle(a))
        lam = 0.0
    else:
        phi = -(float(np.angle(a)) + float(np.angle(-b)))
        lam =   float(np.angle(-b)) - float(np.angle(a))
    return phi, theta, lam


def _build_U(angles: dict) -> np.ndarray:
    """Reconstruct the 2×2 unitary matrix from ZYZ angles."""
    phi, theta, lam = angles['phi'], angles['theta'], angles['lam']

    def Rz(a):
        return np.array([[np.exp(-1j * a / 2), 0.0],
                         [0.0,                 np.exp(1j * a / 2)]])

    def Ry(a):
        return np.array([[ np.cos(a / 2), -np.sin(a / 2)],
                         [ np.sin(a / 2),  np.cos(a / 2)]])

    return Rz(phi) @ Ry(theta) @ Rz(lam)


def build_qpe_circuit(n_prec: int, angles: dict) -> QuantumCircuit:
    """
    Build a 1-qubit QPE sub-circuit using matrix squaring.

    U = Rz(phi) @ Ry(theta) @ Rz(lam)  (Haar-random, ZYZ decomposition)
    U^(2^k) is computed via repeated squaring and re-decomposed into ZYZ angles,
    so each precision qubit requires exactly 3 controlled gates (O(n_prec) total)
    instead of repeating U 2^k times (O(2^n_prec) total).

    Qubit layout: [prec[0], ..., prec[n_prec-1], target]
    """
    n_targ = 1
    total  = n_prec + n_targ

    qc   = QuantumCircuit(total, name='QPE')
    prec = list(range(n_prec))
    targ = n_prec

    qc.h(prec)

    M = _build_U(angles)   # M = U^(2^0) = U^1

    for k in range(n_prec):
        ctrl = prec[k]
        p, t, l = zyz_angles(M)   # decompose current U^(2^k)
        qc.crz(l, ctrl, targ)
        qc.cry(t, ctrl, targ)
        qc.crz(p, ctrl, targ)
        M = M @ M                 # M → U^(2^(k+1)) for next qubit

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
    # barrier removed — not supported on IonQ Forte via Braket

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

    print(f"QPU         : {DEVICE_NAME}  ({DEVICE_ARN})")
    print(f"N_PREC      : {N_PREC}")
    print(f"N_SHOTS     : {N_SHOTS}")
    print(f"SEED        : {SEED}")
    print(f"Euler angles: phi={angles['phi']:.6f}  "
          f"theta={angles['theta']:.6f}  lam={angles['lam']:.6f}")

    qc = build_full_circuit(N_PREC, angles)
    print(f"\nCircuit qubits : {qc.num_qubits}")

    try:
        from qiskit_braket_provider import BraketProvider
    except ImportError:
        print("\nERROR: qiskit-braket-provider not installed.")
        print("       pip install qiskit-braket-provider")
        sys.exit(1)

    # Match by ARN against the underlying AwsDevice — avoids name-matching issues.
    provider = BraketProvider()
    backend  = None
    for b in provider.backends():
        dev = getattr(b, '_device', None)
        if dev and getattr(dev, 'arn', None) == DEVICE_ARN:
            backend = b
            break

    if backend is None:
        print(f"\nERROR: Could not find backend for ARN: {DEVICE_ARN}")
        print("Available backends:")
        for b in provider.backends():
            dev = getattr(b, '_device', None)
            arn = getattr(dev, 'arn', 'N/A') if dev else 'N/A'
            print(f"  {b.name}  →  {arn}")
        sys.exit(1)

    # IonQ Forte is all-to-all connected — no routing SWAPs needed.
    print(f"\nTranspiling for {backend.name} ...")
    qc_hw = transpile(qc, backend=backend, optimization_level=1)
    n_gates = qc_hw.size()
    print(f"Transpiled depth : {qc_hw.depth()}")
    print(f"Transpiled gates : {n_gates}")

    GATE_SHOT_LIMIT = 1_000_000
    n_shots = N_SHOTS
    if n_gates * n_shots > GATE_SHOT_LIMIT:
        n_shots = GATE_SHOT_LIMIT // n_gates
        print(f"\nWARNING: gates × shots would exceed {GATE_SHOT_LIMIT:,}.")
        print(f"         Reducing shots: {N_SHOTS} → {n_shots}  "
              f"({n_gates} gates × {n_shots} shots = {n_gates * n_shots:,})")

    print(f"\nSubmitting {n_shots} shots to {DEVICE_NAME} ...")
    job          = backend.run(qc_hw, shots=n_shots)
    job_id       = job.job_id()
    submitted_at = datetime.now(timezone.utc).isoformat()

    print(f"Job submitted successfully.")
    print(f"Job ID    : {job_id}")
    print(f"Timestamp : {submitted_at}")

    record = {
        "job_id":          job_id,
        "datetime":        submitted_at,
        "qpu":             DEVICE_NAME,
        "device_arn":      DEVICE_ARN,
        "n_prec":          N_PREC,
        "n_shots":         n_shots,
        "n_shots_requested": N_SHOTS,
        "n_gates":         n_gates,
        "seed":            SEED,
        "angles": {
            "phi":   angles['phi'],
            "theta": angles['theta'],
            "lam":   angles['lam'],
        },
    }
    append_job_log(record)


if __name__ == "__main__":
    main()
