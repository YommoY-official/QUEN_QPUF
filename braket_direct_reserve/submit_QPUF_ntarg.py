#!/usr/bin/env python3
"""
submit_QPUF_ntarg.py
====================
Submits a two-stage QPE QPUF circuit to IonQ Forte-1 on AWS Braket, generalized
to n_targ > 1 target qubits using exact Haar-random n-qubit unitaries.

Circuit shape (one shared precision register, reused via MCM + reset):

    target[n_targ]    : Haar-random initial state (one RY+RZ per target qubit)
    prec[n_prec]      : QPE ancillae (reused across the two stages)

      H^{⊗n_prec} ── ctrl-U^{2^{n_prec-1-k}} ── invQFT ── measure → c1 ── reset
      H^{⊗n_prec} ── ctrl-U^{2^{n_prec-1-k}} ── invQFT ── measure → c2

The same Haar-random U is used in both stages (this is the PE-QPUF
construction — the second-stage outcomes verify the eigenstate collapsed by
stage 1).

Haar sampling: complex Ginibre → QR → phase-fix diag(R)/|diag(R)|
(Mezzadri, arXiv:math-ph/0609050).

Controlled-U synthesis: U^{2^k} computed classically via np.linalg.matrix_power,
wrapped as qiskit UnitaryGate, then .control(1). Qiskit's transpiler decomposes
the controlled arbitrary unitary into native gates at submission time.

IonQ Forte-1: 36 qubits, all-to-all connectivity, supports MCM + reset via
Braket Direct.
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DEVICE_NAME = "Forte-Enterprise-1"
DEVICE_ARN  = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"
N_PREC      = 30            # precision qubits (shared across both stages)
N_TARG      = 3             # target qubits — Haar-random unitary acts on these
N_SHOTS     = 10000
SEED        = 100           # RNG seed for the Haar-random unitary
TARGET_INIT_SEED = 99       # RNG seed for the target-state initialisation
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate

JOB_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")


# ── Haar-random unitary ────────────────────────────────────────────────────────

def haar_random_unitary(d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample a d x d Haar-distributed unitary matrix.

    Method (Mezzadri, arXiv:math-ph/0609050):
      1. Draw a complex Ginibre matrix Z ~ (N(0,1) + i N(0,1)) / sqrt(2).
      2. QR-decompose Z = Q R.
      3. Phase-fix: Lambda = diag(R) / |diag(R)|; return Q @ diag(Lambda).
         (Raw QR is NOT Haar — the diagonal phase fix is mandatory.)
    """
    if rng is None:
        rng = np.random.default_rng()

    Z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag_R = np.diag(R)
    Lambda = diag_R / np.abs(diag_R)
    return Q * Lambda  # broadcast: multiplies each column of Q by Lambda[j]


def _stable_matrix_power(U: np.ndarray, power: int) -> np.ndarray:
    """
    U**power, re-projected onto the nearest unitary via SVD (polar
    decomposition). Repeated squaring in float64 squares the error each
    step, so at N_PREC=32 the raw U^{2^31} drifts ~1e-6 from unitary and
    Qiskit's UnitaryGate unitarity check rejects it.
    """
    M = np.linalg.matrix_power(U, power)
    W, _, Vh = np.linalg.svd(M)
    return W @ Vh


# ── QPE stage builder ──────────────────────────────────────────────────────────

def build_qpe_stage(n_prec: int, U: np.ndarray) -> QuantumCircuit:
    """
    Build one QPE stage: H on every precision qubit, controlled-U^{2^{n_prec-1-k}}
    for precision qubit k (qubit 0 is the MSB of the measured integer), then
    inverse QFT on the precision register.

    Qubit layout: [prec[0], ..., prec[n_prec-1], targ[0], ..., targ[n_targ-1]].
    """
    n_targ = int(round(np.log2(U.shape[0])))
    assert 2 ** n_targ == U.shape[0], "U dimension must be a power of 2"
    total = n_prec + n_targ

    qc = QuantumCircuit(total, name="QPE")
    prec = list(range(n_prec))
    targ = list(range(n_prec, n_prec + n_targ))

    qc.h(prec)

    # Stage exponent: precision qubit k (k=0 first) controls U^{2^{n_prec-1-k}}.
    # This puts the most significant bit at qubit 0 of the precision register.
    for k in range(n_prec):
        power = 2 ** (n_prec - 1 - k)
        U_pow = _stable_matrix_power(U, power)
        cU = UnitaryGate(U_pow, label=f"U^{power}").control(1)
        # Order: [control, target_0, target_1, ...]
        qc.append(cU, [prec[k]] + targ)

    iqft = QFTGate(n_prec).inverse()
    qc.append(iqft, prec)

    return qc


def build_full_circuit(n_prec: int, n_targ: int, U: np.ndarray,
                       target_init_seed: int) -> QuantumCircuit:
    """
    Two-stage QPE QPUF with one shared precision register, reused across stages
    via measure → reset.

    Registers:
      target[n_targ]  — Haar-random initial state (one RY+RZ per qubit)
      prec[n_prec]    — QPE ancillae, measured + reset between stages
      c1, c2          — classical bits for stage 1 and stage 2 outcomes
    """
    targ_reg = QuantumRegister(n_targ, "target")
    prec_reg = QuantumRegister(n_prec, "prec")
    c1 = ClassicalRegister(n_prec, "c1")
    c2 = ClassicalRegister(n_prec, "c2")

    qc = QuantumCircuit(targ_reg, prec_reg, c1, c2)

    # Target initialisation: one RY+RZ per qubit, seeded independently of U.
    init_rng = np.random.default_rng(seed=target_init_seed)
    for q in targ_reg:
        theta0 = init_rng.uniform(0, np.pi)
        phi0   = init_rng.uniform(0, 2 * np.pi)
        qc.ry(theta0, q)
        qc.rz(phi0, q)

    # Stage 1 QPE
    qpe1 = build_qpe_stage(n_prec, U)
    qc.append(qpe1, list(prec_reg) + list(targ_reg))

    # MCM and reset the precision register so it can be reused for stage 2.
    qc.measure(prec_reg, c1)
    for q in prec_reg:
        qc.reset(q)

    # Stage 2 QPE on the collapsed target, reusing the same prec qubits.
    qpe2 = build_qpe_stage(n_prec, U)
    qc.append(qpe2, list(prec_reg) + list(targ_reg))

    qc.measure(prec_reg, c2)

    return qc


# ── Logging ────────────────────────────────────────────────────────────────────

def append_job_log(record: dict):
    """Append a JSON record (one per line) to job_log.txt."""
    os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Job record written to: {LOG_FILE}")


def _encode_unitary(U: np.ndarray) -> dict:
    """Serialize a complex matrix to JSON-friendly real/imag lists."""
    return {
        "shape": list(U.shape),
        "real":  U.real.tolist(),
        "imag":  U.imag.tolist(),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Build the Haar-random unitary ─────────────────────────────────────────
    rng = np.random.default_rng(seed=SEED)
    d = 2 ** N_TARG
    U = haar_random_unitary(d, rng=rng)

    # Sanity check: U should be unitary to within numerical precision.
    err = float(np.max(np.abs(U.conj().T @ U - np.eye(d))))
    if err > 1e-10:
        print(f"WARNING: |U†U − I|_max = {err:.2e} (expected ~1e-15)")

    print(f"QPU         : {DEVICE_NAME}  ({DEVICE_ARN})")
    print(f"N_PREC      : {N_PREC}")
    print(f"N_TARG      : {N_TARG}  (U is {d}×{d})")
    print(f"N_SHOTS     : {N_SHOTS}")
    print(f"SEED        : {SEED}")
    print(f"|U†U − I|   : {err:.2e}")

    # ── Build the circuit ──────────────────────────────────────────────────────
    qc = build_full_circuit(N_PREC, N_TARG, U, target_init_seed=TARGET_INIT_SEED)
    print(f"\nCircuit qubits   : {qc.num_qubits}  "
          f"(prec={N_PREC} + targ={N_TARG})")

    # ── Resolve backend ────────────────────────────────────────────────────────
    try:
        from qiskit_braket_provider import BraketProvider
    except ImportError:
        print("\nERROR: qiskit-braket-provider not installed.")
        print("       pip install qiskit-braket-provider")
        sys.exit(1)

    provider = BraketProvider()
    backend = None
    for b in provider.backends():
        dev = getattr(b, "_device", None)
        if dev and getattr(dev, "arn", None) == DEVICE_ARN:
            backend = b
            break

    if backend is None:
        print(f"\nERROR: Could not find backend for ARN: {DEVICE_ARN}")
        print("Available backends:")
        for b in provider.backends():
            dev = getattr(b, "_device", None)
            arn = getattr(dev, "arn", "N/A") if dev else "N/A"
            print(f"  {b.name}  →  {arn}")
        sys.exit(1)

    # ── Qubit-count check ──────────────────────────────────────────────────────
    device_n_qubits = getattr(backend, "num_qubits", None)
    if device_n_qubits is None:
        dev = getattr(backend, "_device", None)
        caps = getattr(dev, "properties", None) if dev else None
        device_n_qubits = caps.paradigm.qubitCount if caps is not None else None

    if device_n_qubits is not None and qc.num_qubits > device_n_qubits:
        print(f"\nERROR: circuit needs {qc.num_qubits} qubits but {DEVICE_NAME} "
              f"only has {device_n_qubits}.")
        print(f"       Reduce N_PREC and/or N_TARG so that "
              f"N_PREC + N_TARG ≤ {device_n_qubits}.")
        sys.exit(1)
    print(f"Device qubits    : {device_n_qubits}  "
          f"(circuit uses {qc.num_qubits})")

    # ── Transpile ──────────────────────────────────────────────────────────────
    # Don't pass backend=backend: IonQ Forte's qiskit target advertises only
    # {gpi, gpi2, rzz, measure, ...} and has no equivalence path from
    # UnitaryGate/QFT decompositions, plus 'reset' isn't in the target. We
    # transpile to rz/rx/rxx (≈ IonQ MS), and Braket Direct's server-side
    # compiler maps these to native gpi/gpi2/rzz at submission time.
    print(f"\nTranspiling for {backend.name} ...")
    qc_hw = transpile(
        qc,
        basis_gates=['rz', 'rx', 'rxx', 'measure', 'reset'],
        optimization_level=1,
    )
    n_gates = qc_hw.size()
    print(f"Transpiled depth : {qc_hw.depth()}")
    print(f"Transpiled gates : {n_gates}")

    # ── Shots × gates budget (Braket limit is 1M for IonQ) ─────────────────────
    GATE_SHOT_LIMIT = 1_000_000
    n_shots = N_SHOTS
    if n_gates * n_shots > GATE_SHOT_LIMIT:
        n_shots = max(1, GATE_SHOT_LIMIT // n_gates)
        print(f"\nWARNING: gates × shots would exceed {GATE_SHOT_LIMIT:,}.")
        print(f"         Reducing shots: {N_SHOTS} → {n_shots}  "
              f"({n_gates} gates × {n_shots} shots = {n_gates * n_shots:,})")

    # ── Submit ─────────────────────────────────────────────────────────────────
    print(f"\nSubmitting {n_shots} shots to {DEVICE_NAME} ...")
    job          = backend.run(qc_hw, shots=n_shots)
    job_id       = job.job_id()
    submitted_at = datetime.now(timezone.utc).isoformat()

    print(f"Job submitted successfully.")
    print(f"Job ID    : {job_id}")
    print(f"Timestamp : {submitted_at}")

    record = {
        "job_id":            job_id,
        "datetime":          submitted_at,
        "qpu":               DEVICE_NAME,
        "device_arn":        DEVICE_ARN,
        "circuit_type":      "QPUF_ntarg",
        "n_prec":            N_PREC,
        "n_targ":            N_TARG,
        "n_shots":           n_shots,
        "n_shots_requested": N_SHOTS,
        "n_gates":           n_gates,
        "seed":              SEED,
        "target_init_seed":  TARGET_INIT_SEED,
        "unitary":           _encode_unitary(U),
    }
    append_job_log(record)


if __name__ == "__main__":
    main()
