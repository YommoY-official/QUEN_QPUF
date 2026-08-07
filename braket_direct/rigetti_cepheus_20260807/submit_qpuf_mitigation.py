#!/usr/bin/env python3
"""
submit_qpuf_mitigation.py
=========================
Submit the two-stage PE-QPUF to Rigetti Cepheus-1-108Q under a Braket Direct
reservation, with client-side noise mitigation.

Interactive: asks for N_PREC (precision qubits per stage), N_TARG (target
qubits), shots, and which mitigation to apply -- then prints the complete
circuit cost (qubit count, 1q/2q gate counts, depth, per-task and total
runtime estimate) and waits for confirmation BEFORE anything is submitted.

Mitigation
----------
Rigetti has no vendor-side error mitigation (IonQ's `Debias` is IonQ-only),
so both techniques here are client-side and cost extra TASKS:

  ZNE  Zero-noise extrapolation. Submits the same circuit at noise scale
       lambda = 1, 3, 5, ... via global folding C -> C (C^dag C)^k. The
       folded circuit has exactly lambda times the gates, so the expectation
       value is sampled at lambda x the physical noise and extrapolated back
       to lambda = 0 offline. Cost: one task per lambda.

       Folding MUST survive to the hardware. A non-verbatim submission is
       recompiled by Rigetti's Quilc, which cancels C^dag C and silently
       turns every lambda back into 1. That is why VERBATIM defaults to ON --
       run submit_test.py with verbatim first to confirm the device accepts
       it.

  REM  Readout-error mitigation. Two extra tasks preparing |0...0> and
       |1...1> on exactly the physical qubits the QPUF measures, giving a
       per-qubit 2x2 confusion matrix to invert at analysis time. Cheap
       (2 tasks, trivial circuits) and on superconducting hardware readout
       is usually the single largest error source -- this is the highest
       value-per-task mitigation available here.

Everything is logged to job_results/job_log.txt with the `role` and
`zne_scale` fields the analysis needs to tell the circuits apart. Retrieve
with:  python checkRetrieve.py
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigetti_qpuf_common import (
    DEVICE_NAME, DEVICE_ARN, RES_ARN,
    F1Q, F2Q,
    haar_random_unitary, build_qpuf_two_stage,
    load_device_caps, transpile_for_rigetti, count_native,
    check_qubits_on_device, native_gate_violations,
    measured_physical_qubits, fold_global, readout_calibration_circuits,
    to_braket_qasm, append_job_log, encode_unitary,
    task_tags, report_reservation,
    estimate_runtime_s, estimate_fidelity, fmt_time, print_circuit_report,
)

# -- DEFAULTS (all overridable at the prompt) ----------------------------------
N_PREC      = 3            # precision qubits PER stage
N_TARG      = 1            # target qubits (Haar-random U acts here)
N_SHOTS     = 1000
ZNE_SCALES  = [1, 3, 5]    # odd integers; [1] means "no ZNE"
USE_REM     = True
USE_VERBATIM = True

SEED             = 10
TARGET_INIT_SEED = 99

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_results")
# ------------------------------------------------------------------------------


# -- Prompt helpers ------------------------------------------------------------

def _pos_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise ValueError("must be >= 1")
    return v


def _parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ("y", "yes", "t", "true", "1", "on"):
        return True
    if s in ("n", "no", "f", "false", "0", "off"):
        return False
    raise ValueError(f"expected yes/no, got {s!r}")


def _parse_scales(s: str) -> list[int]:
    vals = [int(x) for x in s.replace(",", " ").split()]
    for v in vals:
        if v < 1 or v % 2 == 0:
            raise ValueError(f"ZNE scales must be ODD positive integers; got {v}")
    return sorted(set(vals))


def _prompt(label, default, cast):
    """
    Prompt for a value; blank input or EOF keeps the [default].

    `default` must ALREADY be the target type -- it is returned as-is and is
    never passed through `cast`. Handing this the display spelling instead
    ("1 3 5" for a list of ints) silently leaks a str into a typed variable on
    the Enter path only, and the failure surfaces far away: iterating the
    "list" then yields characters, and len() counts characters, not scales.

    Formatting the default for display is this function's job, not the
    caller's -- that is what removes the temptation to pass a string.
    """
    shown = " ".join(map(str, default)) if isinstance(default, (list, tuple)) else default
    while True:
        try:
            raw = input(f"{label} [{shown}]: ").strip()
        except EOFError:
            return default
        if raw == "":
            return default
        try:
            return cast(raw)
        except Exception as e:
            print(f"  invalid input ({e}); try again or press Enter for {shown}.")


def prompt_config():
    global N_PREC, N_TARG, N_SHOTS, ZNE_SCALES, USE_REM, USE_VERBATIM
    print("=" * 78)
    print(f"QPUF + NOISE MITIGATION on {DEVICE_NAME}")
    print("=" * 78)
    print("Press Enter to keep the [default].\n")
    N_PREC  = _prompt("Precision qubits per stage  N_PREC", N_PREC, _pos_int)
    N_TARG  = _prompt("Target qubits               N_TARG", N_TARG, _pos_int)
    N_SHOTS = _prompt("Shots per circuit           N_SHOTS", N_SHOTS, _pos_int)
    ZNE_SCALES = _prompt("ZNE noise scales (odd ints; '1' = no ZNE)",
                         ZNE_SCALES, _parse_scales)
    USE_REM = _prompt("Readout-error mitigation calibration? (y/n)", USE_REM, _parse_bool)
    USE_VERBATIM = _prompt("Verbatim box? (required for ZNE to survive Quilc) (y/n)",
                           USE_VERBATIM, _parse_bool)

    print(f"\n--> N_PREC={N_PREC}  N_TARG={N_TARG}  N_SHOTS={N_SHOTS}  "
          f"ZNE={ZNE_SCALES}  REM={USE_REM}  VERBATIM={USE_VERBATIM}\n")

    if len(ZNE_SCALES) > 1 and not USE_VERBATIM:
        print("WARNING: ZNE with more than one scale factor but VERBATIM=off.")
        print("         Rigetti's compiler will cancel the C^dag C folds and every")
        print("         lambda will execute as lambda=1. The extrapolation will be")
        print("         meaningless. Turn verbatim on, or run ZNE scales = 1 only.\n")


# -- Feasibility sweep ---------------------------------------------------------

def viability_table(n_targ: int, caps: dict, n_shots: int, n_prec_max: int = 6):
    """
    Sweep N_PREC and show what each costs AFTER routing onto the real lattice.

    This is the table that decides N_PREC. Note what it does NOT show: a
    routing catastrophe. On a 108-qubit lattice SABRE only pays ~1.0-1.35x
    the all-to-all 2q count, because there is room to spread out. The column
    that actually bites is `est F` -- the QPUF dies of gate infidelity long
    before it runs out of qubits or connectivity.
    """
    rng = np.random.default_rng(seed=SEED)
    U   = haar_random_unitary(2 ** n_targ, rng=rng)

    print("=" * 92)
    print(f"Feasibility sweep -- two-stage QPUF, N_TARG={n_targ}, {n_shots} shots/circuit")
    print(f"  routed onto {caps['device_name']} ({caps['qubit_count']} qubits), "
          f"fidelity model F1Q={F1Q} F2Q={F2Q}")
    print("=" * 92)
    hdr = (f"{'N_PREC':>6} | {'logical q':>9} | {'phys q':>6} | {'1q':>7} | {'2q':>7} | "
           f"{'depth':>7} | {'2q depth':>8} | {'est F':>7} | {'runtime':>10}")
    print(hdr)
    print("-" * len(hdr))

    for npr in range(1, n_prec_max + 1):
        qc    = build_qpuf_two_stage(npr, n_targ, U, TARGET_INIT_SEED)
        qc_hw = transpile_for_rigetti(qc, caps)
        prof  = count_native(qc_hw)
        rt    = estimate_runtime_s(prof, n_shots)
        fid   = estimate_fidelity(prof)
        flag  = "" if fid >= 0.05 else "  <- noise floor"
        print(f"{npr:>6} | {n_targ + 2*npr:>9} | {prof['n_qubits_used']:>6} | "
              f"{prof['n_1q']:>7,} | {prof['n_2q']:>7,} | {prof['depth']:>7,} | "
              f"{prof['depth_2q']:>8,} | {fid:>7.4f} | "
              f"{rt['t_parallel_s']:>9.1f}s{flag}")
    print("-" * len(hdr))
    print("est F = product of gate fidelities: a rough survival probability for the")
    print("whole circuit, NOT a QPUF success rate. Below ~0.05 the output is noise.")
    print("=" * 92 + "\n")


# -- Main ----------------------------------------------------------------------

def main():
    prompt_config()

    caps, caps_are_real = load_device_caps()
    if not caps_are_real:
        print("NOTE: device_caps.json not found -- using a PLACEHOLDER square lattice.")
        print("      Run `python query_device_caps.py` on the DCV first; gate counts")
        print("      below are the right scale but not the real chip.\n")

    if _prompt("Show the N_PREC feasibility sweep first? (y/n)", True, _parse_bool):
        viability_table(N_TARG, caps, N_SHOTS)

    n_logical = N_TARG + 2 * N_PREC
    if n_logical > caps["qubit_count"]:
        print(f"ERROR: needs {n_logical} logical qubits, device has {caps['qubit_count']}.")
        sys.exit(1)

    # -- Build the base circuit ------------------------------------------------
    rng = np.random.default_rng(seed=SEED)
    d   = 2 ** N_TARG
    U   = haar_random_unitary(d, rng=rng)
    err = float(np.max(np.abs(U.conj().T @ U - np.eye(d))))
    if err > 1e-10:
        print(f"WARNING: |U'U - I|_max = {err:.2e} (expected ~1e-15)")

    print(f"Building + routing the base circuit (N_PREC={N_PREC}, N_TARG={N_TARG}) ...")
    qc    = build_qpuf_two_stage(N_PREC, N_TARG, U, TARGET_INIT_SEED)
    qc_hw = transpile_for_rigetti(qc, caps)
    phys_qubits = measured_physical_qubits(qc_hw)

    print("\n" + "=" * 78)
    print("CONFIGURATION")
    print("=" * 78)
    print(f"  device            : {DEVICE_NAME}  ({DEVICE_ARN})")
    print(f"  submission        : {'reservation ' + RES_ARN if RES_ARN else 'ON-DEMAND (no reservation)'}")
    print(f"  circuit           : two-stage PE-QPUF, deferred measurement")
    print(f"  N_PREC / N_TARG   : {N_PREC} / {N_TARG}   (U is {d}x{d})")
    print(f"  logical qubits    : {n_logical}  = N_TARG + 2*N_PREC")
    print(f"  physical qubits   : {sorted(phys_qubits)}")
    print(f"  shots per circuit : {N_SHOTS:,}")
    print(f"  ZNE scales        : {ZNE_SCALES}")
    print(f"  REM calibration   : {'on (2 extra tasks)' if USE_REM else 'off'}")
    print(f"  verbatim          : {USE_VERBATIM}")
    print(f"  seeds             : U={SEED}, target init={TARGET_INIT_SEED}")

    # -- Assemble every circuit that will be submitted -------------------------
    jobs: list[dict] = []

    for scale in ZNE_SCALES:
        folded = qc_hw if scale == 1 else fold_global(qc_hw, scale)
        jobs.append({
            "role":      "qpuf",
            "label":     f"QPUF lambda={scale}",
            "zne_scale": scale,
            "circuit":   folded,
            "profile":   count_native(folded),
        })

    if USE_REM:
        for label, cal in readout_calibration_circuits(phys_qubits, qc_hw.num_qubits):
            jobs.append({
                "role":      "readout_cal",
                "label":     f"readout calibration {label}",
                "zne_scale": 1,
                "cal_state": label,
                "circuit":   cal,
                "profile":   count_native(cal),
            })

    # -- Report ----------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"CIRCUIT SPECIFICS -- {len(jobs)} task(s) will be submitted")
    print("=" * 78)

    total_runtime = 0.0
    total_gates   = 0
    for job in jobs:
        rt = print_circuit_report(job["label"], job["profile"], N_SHOTS,
                                  n_logical, caps_are_real)
        job["runtime"] = rt
        total_runtime += rt["t_parallel_s"]
        total_gates   += job["profile"]["n_gates"] * N_SHOTS

    print("-" * 78)
    print("TOTALS")
    print("-" * 78)
    print(f"  tasks                        : {len(jobs)}")
    print(f"  shots submitted              : {len(jobs) * N_SHOTS:,}")
    print(f"  total gate-ops executed      : {total_gates:,}")
    print(f"  est. total runtime           : {fmt_time(total_runtime)}")
    print(f"  est. total runtime (+30% pad): {fmt_time(total_runtime * 1.3)}")
    print()
    print("  The runtime estimate is dominated by per-task startup and the")
    print("  per-shot reset delay, both of which are PLACEHOLDER constants in")
    print("  rigetti_qpuf_common.py. Run submit_test.py once and let")
    print("  checkRetrieve.py compare them against Rigetti's own")
    print("  qpuRuntimeEstimation before you size the reservation window.")

    if not RES_ARN:
        print("\n  WARNING: RES_ARN is empty -- these tasks would be submitted")
        print("           ON-DEMAND and billed per task, not against a reservation.")

    try:
        resp = input(f"\nSubmit {len(jobs)} task(s) to {DEVICE_NAME}? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted -- nothing submitted.")
        return

    # -- Submit ----------------------------------------------------------------
    try:
        from braket.aws import AwsDevice, DirectReservation
        from braket.ir.openqasm import Program as OpenQasmProgram
    except ImportError as e:
        print(f"\nERROR: missing dependency: {e}\n       pip install amazon-braket-sdk qiskit")
        sys.exit(1)

    device = AwsDevice(DEVICE_ARN)
    print(f"\nDevice resolved : {device.name}  (status {device.status})")
    # Check the qubits the circuit TOUCHES, not its width -- see
    # check_qubits_on_device(). qc_hw.num_qubits is 108 for every routed
    # circuit (top physical index 107 + 1), which says nothing about whether
    # 107 real qubits are enough.
    try:
        used = check_qubits_on_device(qc_hw, caps)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Physical qubits : {used}  ({len(used)} of "
          f"{device.properties.paradigm.qubitCount} live)")

    # Verbatim pre-flight. A verbatim box is executed EXACTLY as written, so a
    # single non-native gate anywhere gets the whole program rejected. Catch it
    # here rather than burning a round trip inside the reservation window.
    if USE_VERBATIM:
        bad = sorted({v for job in jobs
                      for v in native_gate_violations(job["circuit"], caps)})
        if bad:
            print("\nERROR: verbatim submission requested, but these are not "
                  "natively executable:")
            for v in bad:
                print(f"    {v}")
            print("  (transpile_for_rigetti(native_rx=True) should have lowered "
                  "these -- do not submit verbatim until this is empty.)")
            sys.exit(1)
        print(f"Verbatim check  : OK -- all {len(jobs)} circuit(s) native")

    # DirectReservation routes every task submitted inside the `with` block to
    # the reserved window instead of billing on-demand QPU time.
    ctx = DirectReservation(device, reservation_arn=RES_ARN) if RES_ARN else None

    submitted = []
    log_file  = os.path.join(RESULTS_DIR, "job_log.txt")
    try:
        if ctx is not None:
            ctx.__enter__()
        for job in jobs:
            qasm_src = to_braket_qasm(job["circuit"], verbatim=USE_VERBATIM,
                                      physical=True)
            print(f"\nSubmitting: {job['label']} ({N_SHOTS} shots) ...")
            task = device.run(
                OpenQasmProgram(source=qasm_src), shots=N_SHOTS,
                tags=task_tags("qpuf", {"Role": job["role"],
                                        "ZneScale": job["zne_scale"],
                                        "Verbatim": USE_VERBATIM}),
            )
            submitted_at = datetime.now(timezone.utc).isoformat()
            print(f"  Task ARN : {task.id}")
            if RES_ARN and not submitted:      # first task tells us if it stuck
                report_reservation(task, RES_ARN)

            prof = job["profile"]
            record = {
                "job_id":            task.id,
                "datetime":          submitted_at,
                "qpu":               DEVICE_NAME,
                "device_arn":        DEVICE_ARN,
                "submission":        "reservation" if RES_ARN else "on-demand",
                "reservation_arn":   RES_ARN or None,
                "circuit_type":      f"QPUF2stage_{N_TARG}targ",
                "role":              job["role"],
                "label":             job["label"],
                "cal_state":         job.get("cal_state"),
                "zne_scale":         job["zne_scale"],
                "zne_scales_all":    ZNE_SCALES,
                "error_mitigation":  ("zne" if len(ZNE_SCALES) > 1 else "none")
                                     + ("+rem" if USE_REM else ""),
                "verbatim":          USE_VERBATIM,
                "n_stages":          2,
                "n_prec":            N_PREC,
                "n_targ":            N_TARG,
                "n_qubits":          n_logical,
                "physical_qubits":   sorted(phys_qubits),
                "n_shots":           N_SHOTS,
                "n_1q_gates":        prof["n_1q"],
                "n_2q_gates":        prof["n_2q"],
                "n_gates":           prof["n_gates"],
                "n_measurements":    prof["n_meas"],
                "depth":             prof["depth"],
                "depth_2q":          prof["depth_2q"],
                "est_fidelity":      estimate_fidelity(prof),
                "est_runtime_parallel_s": job["runtime"]["t_parallel_s"],
                "est_runtime_serial_s":   job["runtime"]["t_serial_s"],
                "seed":              SEED,
                "target_init_seed":  TARGET_INIT_SEED,
                "unitary":           encode_unitary(U),
            }
            log_file = append_job_log(RESULTS_DIR, record)
            submitted.append(task.id)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    print(f"\n{len(submitted)}/{len(jobs)} task(s) submitted.")
    print(f"Logged to: {log_file}")
    print("Retrieve with:  python checkRetrieve.py")


if __name__ == "__main__":
    main()
