#!/usr/bin/env python3
"""
ionq_qpuf_two_stage.py
======================
Submits the TWO-STAGE PE-QPUF (two QPE stages on the same Haar-random unitary,
comparing the two precision registers) to IonQ Forte-Enterprise-1 on AWS
Braket, with IonQ native debiasing toggleable.

This is the *real* QPUF: a single QPE stage is just plain phase estimation.
Each shot yields TWO measured integers:
    m1 = stage-1 precision register (prec_a)  -> c[0 .. n_prec-1]
    m2 = stage-2 precision register (prec_b)  -> c[n_prec .. 2*n_prec-1]
Ideally m1 == m2 (both QPE stages read the same collapsed eigenphase), so the
QPUF response / consistency check is the per-shot value m2 - m1.

Circuit + helpers are reused from ionq_noise_mitig.py (build_qpuf_two_stage,
Haar sampler, QASM rewrite, gate-time + fidelity models). Results are written
to a SEPARATE directory (job_results_2stage/) so they never collide with the
single-stage QPE results that ionq_noise_mitigation_compare.ipynb consumes.

Retrieve completed jobs with:
    python checkRetrieve.py job_results_2stage
"""

import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
from qiskit import transpile

# Reuse everything from the single-stage script (same directory).
from ionq_noise_mitig import (
    DEVICE_NAME, DEVICE_ARN,
    SEED, TARGET_INIT_SEED,
    ONE_QUBIT_TIME_S, TWO_QUBIT_TIME_S, READOUT_TIME_S, STARTUP_TIME_S,
    F2Q, F1Q, MAX_TOTAL_GATES, EM_MIN_SHOTS, IONQ_BASIS,
    haar_random_unitary, build_qpuf_two_stage, _rewrite_qasm_for_braket,
    _encode_unitary, _prompt, _parse_bool,
)

# -- CONFIGURATION (defaults; overridable via the interactive prompts) ----------
N_PREC      = 4           # precision qubits PER stage (two-stage => 2*N_PREC + N_TARG qubits)
N_TARG      = 1
USE_DEBIAS  = False       # toggle IonQ native debiasing
N_SHOTS     = 2500        # if USE_DEBIAS, must be >= EM_MIN_SHOTS

DEVICE_QUBITS = 35        # Forte-Enterprise-1
# ------------------------------------------------------------------------------

JOB_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "job_results_2stage")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")


def two_stage_counts(n_prec, n_targ):
    """Native-gate counts of the transpiled two-stage QPUF."""
    rng = np.random.default_rng(seed=SEED)
    U = haar_random_unitary(2 ** n_targ, rng=rng)
    qc = build_qpuf_two_stage(n_prec, n_targ, U, TARGET_INIT_SEED)
    qc_hw = transpile(qc, basis_gates=IONQ_BASIS, optimization_level=1)
    ops = qc_hw.count_ops()
    n_1q = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q = ops.get('rxx', 0)
    n_meas = ops.get('measure', 0)
    return qc, qc_hw, n_1q, n_2q, n_meas


def viability_table(n_targ, n_prec_max=10):
    """Print the gate-cap / fidelity feasibility of the two-stage QPUF."""
    max_gpc = MAX_TOTAL_GATES // N_SHOTS
    print("=" * 88)
    print(f"Two-stage QPUF viability  (N_TARG={n_targ}, N_SHOTS={N_SHOTS}, "
          f"debias={'ON' if USE_DEBIAS else 'OFF'})")
    print(f"  gate cap = {MAX_TOTAL_GATES:,} = (gates/circuit) x shots "
          f"=> <= {max_gpc} gates/circuit at {N_SHOTS} shots")
    print("=" * 88)
    header = (f"{'N_PREC':>6} | {'qubits':>6} | {'1q':>5} | {'2q':>5} | "
              f"{'g/cir':>6} | {'tot_gates':>11} | {'cap':>5} | {'estF':>6}")
    print(header); print("-" * len(header))
    best = 0
    for npr in range(1, n_prec_max + 1):
        _, _, n1, n2, _ = two_stage_counts(npr, n_targ)
        g = n1 + n2
        tot = g * N_SHOTS
        nq = n_targ + 2 * npr
        ok = tot <= MAX_TOTAL_GATES and nq <= DEVICE_QUBITS
        if ok:
            best = npr
        fid = (F2Q ** n2) * (F1Q ** n1)
        print(f"{npr:>6} | {nq:>6} | {n1:>5} | {n2:>5} | {g:>6} | "
              f"{tot:>11,} | {'ok' if ok else 'OVER':>5} | {fid:>6.3f}")
    print("-" * len(header))
    print(f"Max feasible N_PREC at {N_SHOTS} shots: {best}")
    print("=" * 88)
    return best


def prompt_config():
    """Interactive overrides for the four run parameters."""
    global USE_DEBIAS, N_SHOTS, N_PREC, N_TARG

    def _pos_int(s):
        v = int(s)
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    print("=== Two-stage QPUF configuration (Enter to keep [default]) ===")
    USE_DEBIAS = _prompt("Use IonQ debiasing? (y/n)", USE_DEBIAS, _parse_bool)
    N_SHOTS    = _prompt("Number of shots", N_SHOTS, _pos_int)
    N_PREC     = _prompt("Precision qubits per stage  N_PREC", N_PREC, _pos_int)
    N_TARG     = _prompt("Target qubits               N_TARG", N_TARG, _pos_int)
    print(f"--> USE_DEBIAS={USE_DEBIAS}, N_SHOTS={N_SHOTS}, "
          f"N_PREC={N_PREC}, N_TARG={N_TARG}\n")


def append_job_log(record):
    os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
    needs_nl = os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0
    if needs_nl:
        with open(LOG_FILE, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b"\n"
    with open(LOG_FILE, "a") as f:
        if needs_nl:
            f.write("\n")
        f.write(json.dumps(record) + "\n")
    print(f"Job record written to: {LOG_FILE}")


def main():
    prompt_config()
    em_label = "debias" if USE_DEBIAS else "none"

    if USE_DEBIAS and N_SHOTS < EM_MIN_SHOTS:
        print(f"ERROR: USE_DEBIAS=True requires N_SHOTS >= {EM_MIN_SHOTS}; got {N_SHOTS}.")
        sys.exit(1)

    viability_table(N_TARG, n_prec_max=10)

    # Build + transpile the chosen two-stage QPUF.
    rng = np.random.default_rng(seed=SEED)
    d = 2 ** N_TARG
    U = haar_random_unitary(d, rng=rng)
    qc, qc_hw, n_1q, n_2q, n_meas = two_stage_counts(N_PREC, N_TARG)
    n_qubits = N_TARG + 2 * N_PREC
    gates_per_circ = n_1q + n_2q
    total_gates = gates_per_circ * N_SHOTS
    est_fid = (F2Q ** n_2q) * (F1Q ** n_1q)

    print(f"\nQPU         : {DEVICE_NAME}")
    print(f"Circuit     : TWO-STAGE QPUF (2 QPEs), N_PREC={N_PREC}, N_TARG={N_TARG}")
    print(f"Qubits      : {n_qubits}  (n_targ + 2*n_prec)")
    print(f"Mitigation  : {em_label}   |   N_SHOTS={N_SHOTS}")
    print(f"Native gates: {n_1q} x 1q, {n_2q} x 2q  ({gates_per_circ}/circuit), {n_meas} measurements")
    print(f"Total gates : {total_gates:,}  [cap {MAX_TOTAL_GATES:,}]")
    print(f"Est. fidelity: {est_fid:.4f}")

    per_shot = n_1q * ONE_QUBIT_TIME_S + n_2q * TWO_QUBIT_TIME_S
    t_est = STARTUP_TIME_S + N_SHOTS * (per_shot + READOUT_TIME_S)
    print(f"Est. runtime: {t_est:.1f} s ({t_est/60:.2f} min)")

    # Hard feasibility gates.
    if n_qubits > DEVICE_QUBITS:
        print(f"\nERROR: needs {n_qubits} qubits > {DEVICE_QUBITS}.")
        sys.exit(1)
    if total_gates > MAX_TOTAL_GATES:
        print(f"\nERROR: total gates {total_gates:,} exceeds cap {MAX_TOTAL_GATES:,}. "
              f"Reduce N_PREC or N_SHOTS.")
        sys.exit(1)

    try:
        resp = input(f"\nSubmit two-stage QPUF (mitigation: {em_label})? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted -- no job submitted.")
        return

    try:
        from braket.aws import AwsDevice
        from braket.ir.openqasm import Program as OpenQasmProgram
        from braket.error_mitigation import Debias
        from qiskit import qasm3
    except ImportError as e:
        print(f"\nERROR: missing dependency: {e}")
        sys.exit(1)

    device = AwsDevice(DEVICE_ARN)
    BRAKET_BUILTIN_GATES = ['h', 'cx', 'cnot', 'rx', 'ry', 'rz', 'xx',
                            'x', 'y', 'z', 's', 't', 'swap', 'i']
    qasm_src = qasm3.dumps(qc_hw, includes=(), basis_gates=BRAKET_BUILTIN_GATES)
    qasm_src = _rewrite_qasm_for_braket(qasm_src)

    print(f"\nSubmitting {N_SHOTS} shots (mitigation: {em_label}) to {DEVICE_NAME} ...")
    run_kwargs = {"shots": N_SHOTS}
    if USE_DEBIAS:
        run_kwargs["device_parameters"] = {"errorMitigation": Debias()}
    task = device.run(OpenQasmProgram(source=qasm_src), **run_kwargs)
    job_id = task.id
    submitted_at = datetime.now(timezone.utc).isoformat()

    print(f"Job submitted. Job ID: {job_id}")

    record = {
        "job_id":            job_id,
        "datetime":          submitted_at,
        "qpu":               DEVICE_NAME,
        "device_arn":        DEVICE_ARN,
        "submission":        "on-demand",
        "circuit_type":      f"QPUF2stage_{N_TARG}targ_{em_label}",
        "n_stages":          2,
        "n_prec":            N_PREC,
        "n_targ":            N_TARG,
        "n_qubits":          n_qubits,
        "n_shots":           N_SHOTS,
        "error_mitigation":  em_label,
        "em_min_shots":      EM_MIN_SHOTS if USE_DEBIAS else None,
        "n_1q_gates":        n_1q,
        "n_2q_gates":        n_2q,
        "gates_per_circuit": gates_per_circ,
        "total_gates":       total_gates,
        "n_measurements":    n_meas,
        "est_fidelity":      est_fid,
        "seed":              SEED,
        "target_init_seed":  TARGET_INIT_SEED,
        "unitary":           _encode_unitary(U),
    }
    append_job_log(record)


if __name__ == "__main__":
    main()
