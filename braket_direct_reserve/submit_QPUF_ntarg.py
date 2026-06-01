#!/usr/bin/env python3
"""
submit_QPUF_ntarg.py
====================
Submits a two-stage QPE QPUF circuit to IonQ Forte-Enterprise-1 on AWS
Braket, generalised to n_targ > 1 target qubits using exact Haar-random
n-qubit unitaries.

Circuit shape (single quantum + single classical register, as required by
Braket OpenQASM 3). Stage 2 uses a fresh precision-ancilla block because
Forte-Enterprise-1 does NOT expose `reset` via its OpenQASM submission
path (verified against `device.properties.action.supportedOperations`):

    target[n_targ]    : Haar-random initial state (one RY+RZ per target qubit)
    prec_a[n_prec]    : QPE ancillae for stage 1
    prec_b[n_prec]    : QPE ancillae for stage 2 (disjoint from prec_a)

      H^{⊗n_prec} ── ctrl-U^{2^{n_prec-1-k}} ── invQFT ── measure → c[0..n_prec-1]
                                                                      (MCM — target collapses)
      H^{⊗n_prec} ── ctrl-U^{2^{n_prec-1-k}} ── invQFT ── measure → c[n_prec..2n_prec-1]

The same Haar-random U is used in both stages (this is the PE-QPUF
construction — the second-stage outcomes verify the eigenstate collapsed by
the stage-1 mid-circuit measurement).

Qubit budget: N_TARG + 2 * N_PREC must fit Forte-Enterprise-1 (35 qubits).

Haar sampling: complex Ginibre → QR → phase-fix diag(R)/|diag(R)|
(Mezzadri, arXiv:math-ph/0609050).

Controlled-U synthesis: U^{2^k} computed classically via np.linalg.matrix_power,
wrapped as qiskit UnitaryGate, then .control(1). Qiskit's transpiler decomposes
the controlled arbitrary unitary into native gates at submission time.
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DEVICE_NAME = "Forte-Enterprise-1"
DEVICE_ARN  = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"
RES_ARN = "arn:aws:braket:us-east-1:767397707562:reservation/08fda262-2902-4a28-b88e-0969df8830c7"
N_PREC      = 10          # precision qubits (shared across both stages)
N_TARG      = 2          # target qubits — Haar-random unitary acts on these
N_SHOTS     = 1000

SEED        = 10           # RNG seed for the Haar-random unitary
TARGET_INIT_SEED = 99       # RNG seed for the target-state initialisation

# IonQ Forte-1 gate-time model (used only for the wall-clock estimate)
ONE_QUBIT_TIME_S = 130e-6           # per 1q native gate
TWO_QUBIT_TIME_S = 970e-6           # per 2q native gate
READOUT_TIME_S   = 150e-6 + 50e-6   # 200 µs per readout event
STARTUP_TIME_S   = 0.5              # fixed per-task overhead
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
    Two-stage QPE QPUF, flattened to one quantum + one classical register
    (Braket OpenQASM 3 allows only one of each).

    Forte-Enterprise-1 does NOT expose `reset` via the OpenQASM submission
    path (verified by querying its `supportedOperations`), so we cannot
    reuse the precision register across stages. Instead we allocate two
    disjoint precision blocks. MCM on the stage-1 block still collapses
    the target (`supportsUnassignedMeasurements: True`), so the stage-2
    block performs QPE on the same collapsed eigenstate — preserving the
    PE-QPUF construction.

        q[0 .. n_targ-1]                            → target
        q[n_targ .. n_targ+n_prec-1]                → prec_a  (stage 1)
        q[n_targ+n_prec .. n_targ+2*n_prec-1]       → prec_b  (stage 2)
        c[0 .. n_prec-1]                            → stage-1 outcomes (low bits)
        c[n_prec .. 2*n_prec-1]                     → stage-2 outcomes (high bits)
    """
    n_total    = n_targ + 2 * n_prec
    q          = QuantumRegister(n_total, "q")
    c          = ClassicalRegister(2 * n_prec, "c")
    qc         = QuantumCircuit(q, c)

    targ_idx   = list(range(0, n_targ))
    prec_a_idx = list(range(n_targ,             n_targ + n_prec))
    prec_b_idx = list(range(n_targ + n_prec,    n_targ + 2 * n_prec))

    # Target initialisation: one RY+RZ per qubit, seeded independently of U.
    init_rng = np.random.default_rng(seed=target_init_seed)
    for i in targ_idx:
        theta0 = init_rng.uniform(0, np.pi)
        phi0   = init_rng.uniform(0, 2 * np.pi)
        qc.ry(theta0, q[i])
        qc.rz(phi0,   q[i])

    # Stage 1 QPE on prec_a + targ. build_qpe_stage expects the precision
    # qubits first, then the target qubits.
    qpe1 = build_qpe_stage(n_prec, U)
    qc.append(qpe1, prec_a_idx + targ_idx)

    # Mid-circuit measurement of stage-1 precision register. This collapses
    # the target onto an eigenstate of U; the stage-2 QPE that follows then
    # acts on that collapsed state.
    qc.measure([q[i] for i in prec_a_idx], [c[k] for k in range(n_prec)])

    # Stage 2 QPE on prec_b + (already-collapsed) targ.
    qpe2 = build_qpe_stage(n_prec, U)
    qc.append(qpe2, prec_b_idx + targ_idx)

    qc.measure([q[i] for i in prec_b_idx], [c[n_prec + k] for k in range(n_prec)])

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


def _rewrite_qasm_for_braket(qasm: str) -> str:
    """
    Make qiskit's OpenQASM 3 output acceptable to AWS Braket.

    Braket's parser accepts only its own built-in gates: no `include`
    statements and no user-defined `gate` blocks. We:
      1. Strip any `gate rxx(...) { ... }` definition (matched by brace
         depth — robust to multi-line bodies).
      2. Rename every `rxx(<angle>) ...` call site to `xx(<angle>) ...`
         (Braket's native Ising-XX gate; matrix is identical to qiskit
         RXX, so no parameter rescaling is needed).
    """
    import re

    out_lines  = []
    skip_until = -1
    lines      = qasm.splitlines()
    i          = 0
    while i < len(lines):
        line = lines[i]

        # Drop "gate rxx(...) { ... }" block (balanced braces, any depth).
        if re.match(r'\s*gate\s+rxx\b', line):
            depth = line.count('{') - line.count('}')
            i += 1
            while i < len(lines) and depth > 0:
                depth += lines[i].count('{') - lines[i].count('}')
                i += 1
            continue

        # Rename call sites: `rxx(angle) q0, q1;` → `xx(angle) q0, q1;`.
        # Use a word boundary so we don't accidentally rewrite something
        # like `myrxx(...)` if it ever appears.
        line = re.sub(r'\brxx\s*\(', 'xx(', line)

        out_lines.append(line)
        i += 1

    return '\n'.join(out_lines) + ('\n' if qasm.endswith('\n') else '')


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
    print(f"RES_ARN     : {RES_ARN}")
    print(f"N_PREC      : {N_PREC}")
    print(f"N_TARG      : {N_TARG}  (U is {d}×{d})")
    print(f"N_SHOTS     : {N_SHOTS}")
    print(f"SEED        : {SEED}")
    print(f"|U†U − I|   : {err:.2e}")

    # ── Build the circuit ──────────────────────────────────────────────────────
    qc = build_full_circuit(N_PREC, N_TARG, U, target_init_seed=TARGET_INIT_SEED)
    print(f"\nCircuit qubits   : {qc.num_qubits}  "
          f"(prec={N_PREC} + targ={N_TARG})")

    # ── Resolve device (amazon-braket-sdk; bypass qiskit-braket-provider) ─────
    # qiskit-braket-provider's adapter refuses to translate `reset`
    # (NotImplementedError in providers/adapter.py), so we can't go through it
    # for an MCM+reset circuit. Instead we build/transpile in Qiskit, export
    # OpenQASM 3, and submit via the Braket SDK directly. IonQ Forte-1 +
    # Braket Direct accept `reset` at the QASM level.
    try:
        from braket.aws import AwsDevice, DirectReservation
        from braket.experimental_capabilities import EnableExperimentalCapability
        from braket.ir.openqasm import Program as OpenQasmProgram
        from qiskit import qasm3
    except ImportError as e:
        print(f"\nERROR: missing dependency: {e}")
        print("       pip install amazon-braket-sdk qiskit")
        sys.exit(1)

    device = AwsDevice(DEVICE_ARN)
    print(f"Device resolved  : {device.name}")

    # ── Qubit-count check ──────────────────────────────────────────────────────
    device_n_qubits = device.properties.paradigm.qubitCount
    if qc.num_qubits > device_n_qubits:
        print(f"\nERROR: circuit needs {qc.num_qubits} qubits but {DEVICE_NAME} "
              f"only has {device_n_qubits}.")
        print(f"       Reduce N_PREC and/or N_TARG so that "
              f"N_TARG + 2 * N_PREC ≤ {device_n_qubits}.")
        sys.exit(1)
    print(f"Device qubits    : {device_n_qubits}  "
          f"(circuit uses {qc.num_qubits})")

    # ── Transpile ──────────────────────────────────────────────────────────────
    # rz/rx/rxx ≈ IonQ MS-basis; Braket Direct's server-side compiler maps
    # these to native gpi/gpi2/zz at submission time. `reset` survives the
    # transpile and is preserved in the QASM 3 export below.
    print(f"\nTranspiling for {DEVICE_NAME} ...")
    qc_hw = transpile(
        qc,
        basis_gates=['rz', 'rx', 'rxx', 'measure', 'reset'],
        optimization_level=1,
    )
    n_gates = qc_hw.size()
    ops     = qc_hw.count_ops()
    n_1q    = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q    = ops.get('rxx', 0)
    n_meas  = ops.get('measure', 0)
    print(f"Transpiled depth : {qc_hw.depth()}")
    print(f"Transpiled gates : {n_gates}")
    print(f"  1q gates (rz+rx) : {n_1q}")
    print(f"  2q gates (rxx)   : {n_2q}")
    print(f"  measurements     : {n_meas}")

    # ── Wall-clock estimate (IonQ Forte-1 gate-time model) ────────────────────
    # (a) literal — readout charged once per shot, as written in the formula
    # (b) per-meas — readout charged per measurement event in the circuit
    per_shot_gates    = n_1q * ONE_QUBIT_TIME_S + n_2q * TWO_QUBIT_TIME_S
    t_est_literal_s   = STARTUP_TIME_S + N_SHOTS * (per_shot_gates + READOUT_TIME_S)
    t_est_permeas_s   = STARTUP_TIME_S + N_SHOTS * (per_shot_gates + n_meas * READOUT_TIME_S)

    def _fmt(t_s):
        return f"{t_s:.1f} s  ({t_s/60:.2f} min, {t_s/3600:.3f} h)"

    print(f"\nEstimated runtime (literal readout/shot)    : {_fmt(t_est_literal_s)}")
    print(f"Estimated runtime (readout × n_measurements): {_fmt(t_est_permeas_s)}")

    # ── Confirm before submitting ─────────────────────────────────────────────
    try:
        resp = input("\nSubmit this job to the QPU? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted — no job submitted.")
        return

    # ── Submit ─────────────────────────────────────────────────────────────────
    # Export to OpenQASM 3, then submit via the Braket SDK directly.
    # The DirectReservation context manager (recommended by Braket) routes
    # every task submitted inside its `with` block to the reservation, so it
    # bills against the reserved window instead of on-demand QPU time.
    #
    # Braket's OpenQASM 3 parser is restrictive: it rejects both `include`
    # statements AND user-defined `gate` declarations — only its built-in
    # gates are allowed. Qiskit's `rxx` isn't a Braket built-in, but Braket's
    # `xx(angle)` is the same matrix (verified: xx(θ) = exp(-iθ/2 X⊗X) =
    # qiskit RXX(θ)). So we (a) declare Braket built-ins as basis_gates so
    # qiskit emits no defs for them, (b) drop the include line, and (c)
    # rewrite the `rxx` definition + every `rxx(...)` call to `xx(...)`.
    BRAKET_BUILTIN_GATES = ['h', 'cx', 'cnot',
                            'rx', 'ry', 'rz', 'xx',
                            'x',  'y',  'z',
                            's',  't',  'swap', 'i']
    qasm_src = qasm3.dumps(qc_hw,
                            includes=(),
                            basis_gates=BRAKET_BUILTIN_GATES)
    qasm_src = _rewrite_qasm_for_braket(qasm_src)

    # `reset` is rejected by Braket's default OpenQASM parser; opting in to
    # experimental capabilities turns on the MCM+reset feature that Forte-1
    # supports under Braket Direct.
    print(f"\nSubmitting {N_SHOTS} shots to {DEVICE_NAME} under reservation {RES_ARN} ...")
    with DirectReservation(device, reservation_arn=RES_ARN), \
         EnableExperimentalCapability():
        task = device.run(
            OpenQasmProgram(source=qasm_src),
            shots=N_SHOTS,
        )
    job_id       = task.id    # Braket task ARN — same format as the old code
    submitted_at = datetime.now(timezone.utc).isoformat()

    print(f"Job submitted successfully.")
    print(f"Job ID    : {job_id}")
    print(f"Timestamp : {submitted_at}")

    record = {
        "job_id":               job_id,
        "datetime":             submitted_at,
        "qpu":                  DEVICE_NAME,
        "device_arn":           DEVICE_ARN,
        "reservation_arn":      RES_ARN,
        "circuit_type":         "QPUF_ntarg",
        "n_prec":               N_PREC,
        "n_targ":               N_TARG,
        "n_shots":              N_SHOTS,
        "n_gates":              n_gates,
        "n_1q_gates":           n_1q,
        "n_2q_gates":           n_2q,
        "n_measurements":       n_meas,
        "est_runtime_s_literal": t_est_literal_s,
        "est_runtime_s_permeas": t_est_permeas_s,
        "seed":                 SEED,
        "target_init_seed":     TARGET_INIT_SEED,
        "unitary":              _encode_unitary(U),
    }
    append_job_log(record)


if __name__ == "__main__":
    main()
