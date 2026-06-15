#!/usr/bin/env python3
"""
ionq_noise_mitig.py
===================
Submits a SINGLE-TARGET-QUBIT, single-stage QPE QPUF circuit to IonQ
Forte-Enterprise-1 on AWS Braket WITH IonQ's native debiasing error
mitigation enabled.

Debiasing distributes the requested shots across symmetrization variants --
it does NOT multiply total QPU time beyond the shot count. IonQ requires
>= 2500 shots for debiasing, which is why N_SHOTS = 2500 and the per-shot
gate budget is tight: the floor is a shot floor, not a runtime floor.

Verified Braket debiasing API (introspected against the installed
amazon-braket-sdk):
    from braket.error_mitigation import Debias
    device.run(OpenQasmProgram(source=qasm_src), shots=N_SHOTS,
               device_parameters={"errorMitigation": Debias()})
AwsDevice.run forwards device_parameters via **aws_quantum_task_kwargs to
AwsQuantumTask.create, which serializes Debias() into
IonqDeviceParameters.errorMitigation (field verified present).

Haar sampling: complex Ginibre -> QR -> phase-fix (Mezzadri, math-ph/0609050).
"""

# -- CONFIGURATION --------------------------------------------------------------
DEVICE_NAME = "Forte-Enterprise-1"
DEVICE_ARN  = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"
#RES_ARN = "arn:aws:braket:us-east-1:767397707562:reservation/08fda262-2902-4a28-b88e-0969df8830c7"
N_PREC      = 6           # precision qubits -- easy to change (see calculator)
N_TARG      = 1           # single target qubit (Haar-random unitary acts here)

USE_DEBIAS  = False        # toggle IonQ native debiasing error mitigation ON/OFF
N_SHOTS     = 2500        # shots; if USE_DEBIAS, must be >= EM_MIN_SHOTS (checked in main)

SEED        = 10
TARGET_INIT_SEED = 99

ONE_QUBIT_TIME_S = 130e-6
TWO_QUBIT_TIME_S = 970e-6
READOUT_TIME_S   = 150e-6 + 50e-6
STARTUP_TIME_S   = 0.5

# F = (F2Q)**n_2q * (F1Q)**n_1q -- IonQ Forte typical; TUNE to your calibration.
F2Q = 0.985
F1Q = 0.9998

RUNTIME_BUDGET_S = 3600.0   # default 1 hour per job
FIDELITY_FLOOR   = 0.5

# Hard cap: (gates per circuit) * N_SHOTS must stay below this. At N_SHOTS=2500
# this means <= 400 gates/circuit. Counts 1q + 2q gate operations (not readout).
MAX_TOTAL_GATES = 1_000_000

EM_MIN_SHOTS = 2500
# ------------------------------------------------------------------------------

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate

JOB_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")
IONQ_BASIS = ['rz', 'rx', 'rxx', 'measure']


def haar_random_unitary(d, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    Z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag_R = np.diag(R)
    Lambda = diag_R / np.abs(diag_R)
    return Q * Lambda


def _stable_matrix_power(U, power):
    M = np.linalg.matrix_power(U, power)
    W, _, Vh = np.linalg.svd(M)
    return W @ Vh


def build_qpe_stage(n_prec, U):
    n_targ = int(round(np.log2(U.shape[0])))
    assert 2 ** n_targ == U.shape[0], "U dimension must be a power of 2"
    total = n_prec + n_targ
    qc = QuantumCircuit(total, name="QPE")
    prec = list(range(n_prec))
    targ = list(range(n_prec, n_prec + n_targ))
    qc.h(prec)
    for k in range(n_prec):
        power = 2 ** (n_prec - 1 - k)
        U_pow = _stable_matrix_power(U, power)
        cU = UnitaryGate(U_pow, label=f"U^{power}").control(1)
        qc.append(cU, [prec[k]] + targ)
    qc.append(QFTGate(n_prec).inverse(), prec)
    return qc


def build_full_circuit(n_prec, n_targ, U, target_init_seed):
    n_total = n_targ + n_prec
    q = QuantumRegister(n_total, "q")
    c = ClassicalRegister(n_prec, "c")
    qc = QuantumCircuit(q, c)
    targ_idx = list(range(0, n_targ))
    prec_idx = list(range(n_targ, n_targ + n_prec))
    init_rng = np.random.default_rng(seed=target_init_seed)
    for i in targ_idx:
        qc.ry(init_rng.uniform(0, np.pi), q[i])
        qc.rz(init_rng.uniform(0, 2 * np.pi), q[i])
    qc.append(build_qpe_stage(n_prec, U), prec_idx + targ_idx)
    qc.measure([q[i] for i in prec_idx], [c[k] for k in range(n_prec)])
    return qc


def _transpiled_counts(n_prec, n_targ, U, target_init_seed):
    qc = build_full_circuit(n_prec, n_targ, U, target_init_seed=target_init_seed)
    qc_hw = transpile(qc, basis_gates=IONQ_BASIS, optimization_level=1)
    ops = qc_hw.count_ops()
    n_1q = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q = ops.get('rxx', 0)
    n_meas = ops.get('measure', 0)
    return n_1q, n_2q, n_meas


def _runtime_estimate_s(n_1q, n_2q, n_shots):
    per_shot_gates = n_1q * ONE_QUBIT_TIME_S + n_2q * TWO_QUBIT_TIME_S
    return STARTUP_TIME_S + n_shots * (per_shot_gates + READOUT_TIME_S)


def affordable_nprec_table(n_targ, target_init_seed=TARGET_INIT_SEED, n_prec_max=15):
    # Note: debiasing distributes 2500 shots across symmetrization variants;
    # it does NOT multiply total QPU time beyond the shot count. The >=2500
    # shot floor is the reason the per-shot gate budget is tight.
    #
    # Three feasibility constraints per N_PREC:
    #   (1) runtime  <= RUNTIME_BUDGET_S
    #   (2) fidelity >= FIDELITY_FLOOR
    #   (3) (gates/circuit) * N_SHOTS <= MAX_TOTAL_GATES  (gates/circuit = n_1q + n_2q)
    rng = np.random.default_rng(seed=SEED)
    U = haar_random_unitary(2 ** n_targ, rng=rng)
    max_gates_per_circ = MAX_TOTAL_GATES // N_SHOTS
    print("=" * 90)
    print(f"Affordable-N_PREC calculator  (N_TARG={n_targ}, N_SHOTS={N_SHOTS}, debiasing ON)")
    print(f"  runtime budget = {RUNTIME_BUDGET_S/60:.1f} min, fidelity floor = {FIDELITY_FLOOR}")
    print(f"  total-gate cap = {MAX_TOTAL_GATES:,} = (gates/circuit) * shots "
          f"=> <= {max_gates_per_circ} gates/circuit at {N_SHOTS} shots")
    print(f"  fidelity model: F = (F2Q={F2Q})^n_2q * (F1Q={F1Q})^n_1q  (tune to calib)")
    print("=" * 90)
    header = (f"{'N_PREC':>6} | {'n_1q':>5} | {'n_2q':>5} | {'gates/cir':>9} | "
              f"{'tot_gates':>11} | {'runtime(min)':>12} | {'fidelity':>9}")
    print(header)
    print("-" * len(header))
    best_feasible = 0
    for n_prec in range(1, n_prec_max + 1):
        n_1q, n_2q, _ = _transpiled_counts(n_prec, n_targ, U, target_init_seed)
        gates_per_circ = n_1q + n_2q
        total_gates = gates_per_circ * N_SHOTS
        rt_s = _runtime_estimate_s(n_1q, n_2q, N_SHOTS)
        fid = (F2Q ** n_2q) * (F1Q ** n_1q)
        ok_rt = rt_s <= RUNTIME_BUDGET_S
        ok_fid = fid >= FIDELITY_FLOOR
        ok_gates = total_gates <= MAX_TOTAL_GATES
        feasible = ok_rt and ok_fid and ok_gates
        if feasible:
            best_feasible = n_prec
        # Flag which constraint(s) fail, so the binding limit is obvious.
        if feasible:
            flag = "  <- feasible"
        else:
            fails = []
            if not ok_gates: fails.append("gates")
            if not ok_fid:   fails.append("fidelity")
            if not ok_rt:    fails.append("runtime")
            flag = "  X " + ",".join(fails)
        print(f"{n_prec:>6} | {n_1q:>5} | {n_2q:>5} | {gates_per_circ:>9} | "
              f"{total_gates:>11,} | {rt_s/60:>12.2f} | {fid:>9.4f}{flag}")
    print("-" * len(header))
    if best_feasible > 0:
        print(f"Recommended MAX feasible N_PREC (N_TARG={n_targ}): {best_feasible}  "
              f"[runtime<= {RUNTIME_BUDGET_S/60:.0f} min, fidelity>= {FIDELITY_FLOOR}, "
              f"tot_gates<= {MAX_TOTAL_GATES:,}]")
    else:
        print(f"No N_PREC in range satisfies all constraints for N_TARG={n_targ}.")
    print("=" * 90)
    return best_feasible


def append_job_log(record):
    os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Job record written to: {LOG_FILE}")


def _encode_unitary(U):
    return {"shape": list(U.shape), "real": U.real.tolist(), "imag": U.imag.tolist()}


def _rewrite_qasm_for_braket(qasm):
    import re
    out_lines, lines, i = [], qasm.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'\s*gate\s+rxx\b', line):
            depth = line.count('{') - line.count('}')
            i += 1
            while i < len(lines) and depth > 0:
                depth += lines[i].count('{') - lines[i].count('}')
                i += 1
            continue
        line = re.sub(r'\brxx\s*\(', 'xx(', line)
        out_lines.append(line)
        i += 1
    return '\n'.join(out_lines) + ('\n' if qasm.endswith('\n') else '')


def main():
    # Debiasing requires >= EM_MIN_SHOTS shots; without mitigation any count is fine.
    if USE_DEBIAS and N_SHOTS < EM_MIN_SHOTS:
        print(f"ERROR: USE_DEBIAS=True requires N_SHOTS >= {EM_MIN_SHOTS}; got {N_SHOTS}.")
        sys.exit(1)

    em_label = "debias" if USE_DEBIAS else "none"

    # Affordability sweep for BOTH 1- and 2-target-qubit QPUFs.
    for nt in (1, 2):
        affordable_nprec_table(nt, target_init_seed=TARGET_INIT_SEED, n_prec_max=15)

    rng = np.random.default_rng(seed=SEED)
    d = 2 ** N_TARG
    U = haar_random_unitary(d, rng=rng)
    err = float(np.max(np.abs(U.conj().T @ U - np.eye(d))))
    if err > 1e-10:
        print(f"WARNING: |UdagU - I|_max = {err:.2e}")

    print(f"\nQPU         : {DEVICE_NAME}  ({DEVICE_ARN})")
    print(f"N_PREC      : {N_PREC}")
    print(f"N_TARG      : {N_TARG}  (U is {d}x{d})")
    print(f"N_SHOTS     : {N_SHOTS}")
    print(f"MITIGATION  : {'debias (ON)' if USE_DEBIAS else 'none (OFF)'}"
          f"{f'  (debiasing min = {EM_MIN_SHOTS})' if USE_DEBIAS else ''}")
    print(f"SEED        : {SEED}")

    qc = build_full_circuit(N_PREC, N_TARG, U, target_init_seed=TARGET_INIT_SEED)
    print(f"\nCircuit qubits   : {qc.num_qubits}  (prec={N_PREC} + targ={N_TARG})")

    try:
        from braket.aws import AwsDevice
        from braket.ir.openqasm import Program as OpenQasmProgram
        from braket.error_mitigation import Debias
        from qiskit import qasm3
    except ImportError as e:
        print(f"\nERROR: missing dependency: {e}")
        print("       pip install amazon-braket-sdk qiskit")
        sys.exit(1)

    device = AwsDevice(DEVICE_ARN)
    print(f"Device resolved  : {device.name}")

    device_n_qubits = device.properties.paradigm.qubitCount
    if qc.num_qubits > device_n_qubits:
        print(f"\nERROR: circuit needs {qc.num_qubits} qubits but {DEVICE_NAME} "
              f"only has {device_n_qubits}.")
        sys.exit(1)
    print(f"Device qubits    : {device_n_qubits}  (circuit uses {qc.num_qubits})")

    print(f"\nTranspiling for {DEVICE_NAME} ...")
    qc_hw = transpile(qc, basis_gates=IONQ_BASIS, optimization_level=1)
    n_gates = qc_hw.size()
    ops = qc_hw.count_ops()
    n_1q = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q = ops.get('rxx', 0)
    n_meas = ops.get('measure', 0)
    print(f"Transpiled depth : {qc_hw.depth()}")
    print(f"Transpiled gates : {n_gates}")
    print(f"  1q gates (rz+rx) : {n_1q}")
    print(f"  2q gates (rxx)   : {n_2q}")
    print(f"  measurements     : {n_meas}")

    est_fid = (F2Q ** n_2q) * (F1Q ** n_1q)
    print(f"  est. fidelity    : {est_fid:.4f}")

    per_shot_gates = n_1q * ONE_QUBIT_TIME_S + n_2q * TWO_QUBIT_TIME_S
    t_est_literal_s = STARTUP_TIME_S + N_SHOTS * (per_shot_gates + READOUT_TIME_S)
    t_est_permeas_s = STARTUP_TIME_S + N_SHOTS * (per_shot_gates + n_meas * READOUT_TIME_S)

    def _fmt(t_s):
        return f"{t_s:.1f} s  ({t_s/60:.2f} min, {t_s/3600:.3f} h)"

    print(f"\nEstimated runtime (literal readout/shot)    : {_fmt(t_est_literal_s)}")
    print(f"Estimated runtime (readout x n_measurements): {_fmt(t_est_permeas_s)}")
    if USE_DEBIAS:
        print("Note: debiasing distributes the shots across symmetrization variants;")
        print("      it does not multiply total QPU time beyond the shot count.")

    # -- Total-gate hard cap: (gates/circuit) * shots must be < MAX_TOTAL_GATES --
    gates_per_circ = n_1q + n_2q
    total_gates = gates_per_circ * N_SHOTS
    print(f"\nTotal gates      : {total_gates:,}  "
          f"({gates_per_circ} gates/circuit x {N_SHOTS} shots)  "
          f"[cap {MAX_TOTAL_GATES:,}]")
    if total_gates > MAX_TOTAL_GATES:
        print(f"\nERROR: total gates {total_gates:,} exceeds the cap {MAX_TOTAL_GATES:,}. "
              f"Reduce N_PREC or N_SHOTS.")
        sys.exit(1)

    try:
        resp = input(f"\nSubmit this job (mitigation: {em_label}) to the QPU? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted -- no job submitted.")
        return

    BRAKET_BUILTIN_GATES = ['h', 'cx', 'cnot', 'rx', 'ry', 'rz', 'xx',
                            'x', 'y', 'z', 's', 't', 'swap', 'i']
    qasm_src = qasm3.dumps(qc_hw, includes=(), basis_gates=BRAKET_BUILTIN_GATES)
    qasm_src = _rewrite_qasm_for_braket(qasm_src)

    # VERIFIED debiasing API: device_parameters={"errorMitigation": Debias()}
    # Submitted on-demand directly to DEVICE_ARN (no reservation). When
    # USE_DEBIAS is False we omit device_parameters entirely -> no mitigation.
    print(f"\nSubmitting {N_SHOTS} shots (mitigation: {em_label}) on-demand to {DEVICE_NAME} ...")
    run_kwargs = {"shots": N_SHOTS}
    if USE_DEBIAS:
        run_kwargs["device_parameters"] = {"errorMitigation": Debias()}
    task = device.run(OpenQasmProgram(source=qasm_src), **run_kwargs)
    job_id = task.id
    submitted_at = datetime.now(timezone.utc).isoformat()

    print(f"Job submitted successfully.")
    print(f"Job ID    : {job_id}")
    print(f"Timestamp : {submitted_at}")

    record = {
        "job_id": job_id,
        "datetime": submitted_at,
        "qpu": DEVICE_NAME,
        "device_arn": DEVICE_ARN,
        "submission": "on-demand",
        "circuit_type": f"QPUF_{N_TARG}targ_{em_label}",
        "n_prec": N_PREC,
        "n_targ": N_TARG,
        "n_shots": N_SHOTS,
        "error_mitigation": em_label,
        "em_min_shots": EM_MIN_SHOTS if USE_DEBIAS else None,
        "n_gates": n_gates,
        "n_1q_gates": n_1q,
        "n_2q_gates": n_2q,
        "gates_per_circuit": gates_per_circ,
        "total_gates": total_gates,
        "n_measurements": n_meas,
        "est_fidelity": est_fid,
        "est_runtime_s_literal": t_est_literal_s,
        "est_runtime_s_permeas": t_est_permeas_s,
        "seed": SEED,
        "target_init_seed": TARGET_INIT_SEED,
        "unitary": _encode_unitary(U),
    }
    append_job_log(record)


if __name__ == "__main__":
    main()
