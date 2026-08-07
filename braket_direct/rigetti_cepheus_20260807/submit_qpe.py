#!/usr/bin/env python3
"""
submit_qpe.py
=============
Submit plain single-stage Quantum Phase Estimation to Rigetti
Cepheus-1-108Q, with or without noise mitigation.

Interactive: asks for precision qubits, target qubits, shots, which input
state to use, and whether to mitigate -- then prints the full circuit cost
and the noiseless reference distribution, and waits for confirmation BEFORE
submitting anything.

Why QPE rather than the two-stage QPUF
--------------------------------------
QPE has ONE terminal measurement of the precision register and nothing else.
There is no mid-circuit measurement, no second stage, and no question about
what the readout means: the measured integer m gives the phase estimate
m / 2^n_prec directly. That makes it the clean vehicle for a mitigation
study -- you can score the hardware output against an exactly known answer.

    qubits used = N_TARG + N_PREC          (the QPUF needs N_TARG + 2*N_PREC)

Input state
-----------
  known : U is a diagonal unitary whose |1...1> eigenstate carries exactly
          the phase PHI (default 1/8), and the target register is prepared
          in that eigenstate. Noiseless QPE then puts ALL its weight in bin
          round(PHI * 2^n_prec). Any spread you measure is error, so the
          effect of mitigation is directly readable. This is the default,
          and with N_TARG=1, PHI=1/8 it is the textbook T-gate QPE.

  haar  : the same seeded Haar-random U and Haar-random target state the
          QPUF scripts use, so the QPE run is directly comparable to the
          QPUF run on the same unitary. A generic input is a superposition
          of eigenvectors, so the ideal output already spreads over several
          eigenphases -- still exactly computable, just a weaker reference.

Mitigation (client-side; Rigetti has no vendor-side EM)
-------------------------------------------------------
  none  1 task.
  rem   + 2 readout-calibration tasks (all-|0>, all-|1>) on the same
        physical qubits, giving per-qubit confusion matrices to invert.
  zne   + one task per extra noise scale, via global folding
        C -> C (C^dag C)^k. NEEDS VERBATIM, or Rigetti's Quilc cancels the
        folds and every lambda runs as lambda=1.
  both  zne + rem.

Results go to job_results_qpe/. Retrieve with:
    python checkRetrieve.py job_results_qpe
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigetti_qpuf_common import (
    DEVICE_NAME, DEVICE_ARN, RES_ARN,
    haar_random_unitary, known_phase_unitary,
    build_qpe_circuit, ideal_qpe_distribution,
    load_device_caps, transpile_for_rigetti, count_native,
    check_qubits_on_device, native_gate_violations,
    measured_physical_qubits, fold_global, readout_calibration_circuits,
    to_braket_qasm, append_job_log, encode_unitary,
    task_tags, report_reservation,
    estimate_fidelity, fmt_time, print_circuit_report,
)

# -- DEFAULTS (all overridable at the prompt) ----------------------------------
N_PREC      = 5            # precision qubits
N_TARG      = 1            # target qubits
N_SHOTS     = 1000

MITIGATION  = "both"       # none | rem | zne | both
STATE_MODE  = "known"      # known | haar
PHI         = 1 / 8        # phase to estimate, `known` mode only
ZNE_SCALES  = [1, 3, 5]    # odd integers; used when MITIGATION includes zne
USE_VERBATIM = True

SEED             = 10      # Haar unitary seed   (matches the QPUF scripts)
TARGET_INIT_SEED = 99      # Haar target-state seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "job_results_qpe")
# ------------------------------------------------------------------------------


# -- Prompt helpers ------------------------------------------------------------

def _pos_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise ValueError("must be >= 1")
    return v


def _fraction(s: str) -> float:
    """Accept 0.125 or 1/8."""
    s = s.strip()
    if "/" in s:
        num, den = s.split("/", 1)
        return float(num) / float(den)
    return float(s)


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


def _one_of(options: tuple[str, ...]):
    def cast(s: str) -> str:
        s = s.strip().lower()
        if s not in options:
            raise ValueError(f"expected one of {'/'.join(options)}")
        return s
    return cast


def _prompt(label, default, cast):
    """
    Prompt for a value; blank input or EOF keeps the [default].

    `default` must ALREADY be the target type -- it is returned as-is and is
    never passed through `cast`. Handing this the display spelling instead
    (the string "1/8" for a float, "1 3 5" for a list) silently leaks a str
    into a typed variable on the Enter path only, and the failure surfaces far
    away at the first format of that variable, as
    "unknown format code 'g' for object type 'str'".

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
    global N_PREC, N_TARG, N_SHOTS, MITIGATION, STATE_MODE, PHI, ZNE_SCALES, USE_VERBATIM

    print("=" * 78)
    print(f"QUANTUM PHASE ESTIMATION on {DEVICE_NAME}")
    print("=" * 78)
    print("Press Enter to keep the [default].\n")

    N_PREC  = _prompt("Precision qubits  N_PREC", N_PREC, _pos_int)
    N_TARG  = _prompt("Target qubits     N_TARG", N_TARG, _pos_int)
    N_SHOTS = _prompt("Shots per circuit N_SHOTS", N_SHOTS, _pos_int)

    print("\nError mitigation:")
    print("  none  no mitigation           (1 task)")
    print("  rem   readout calibration     (+2 tasks)")
    print("  zne   zero-noise extrapolation(+1 task per extra scale)")
    print("  both  zne + rem")
    MITIGATION = _prompt("Mitigation", MITIGATION, _one_of(("none", "rem", "zne", "both")))

    if MITIGATION in ("zne", "both"):
        ZNE_SCALES = _prompt("  ZNE noise scales (odd ints)",
                             ZNE_SCALES, _parse_scales)
    else:
        ZNE_SCALES = [1]

    print("\nInput state:")
    print("  known  exact eigenstate of a known-phase U -> single sharp bin")
    print("  haar   seeded Haar U + Haar target state (comparable to the QPUF run)")
    STATE_MODE = _prompt("State mode", STATE_MODE, _one_of(("known", "haar")))
    if STATE_MODE == "known":
        PHI = _prompt("  Phase PHI to estimate (e.g. 1/8)", PHI, _fraction)

    USE_VERBATIM = _prompt("\nVerbatim box? (required for ZNE to survive Quilc) (y/n)",
                           USE_VERBATIM, _parse_bool)

    print(f"\n--> N_PREC={N_PREC}  N_TARG={N_TARG}  N_SHOTS={N_SHOTS}  "
          f"MITIGATION={MITIGATION}  STATE={STATE_MODE}"
          + (f" PHI={PHI:g}" if STATE_MODE == "known" else "")
          + f"  VERBATIM={USE_VERBATIM}\n")

    if MITIGATION in ("zne", "both") and len(ZNE_SCALES) > 1 and not USE_VERBATIM:
        print("WARNING: ZNE requested with VERBATIM=off. Rigetti's compiler will")
        print("         cancel the C^dag C folds, every lambda will execute as")
        print("         lambda=1, and the extrapolation will be meaningless.\n")


# -- Main ----------------------------------------------------------------------

def main():
    prompt_config()

    use_zne = MITIGATION in ("zne", "both")
    use_rem = MITIGATION in ("rem", "both")
    scales  = ZNE_SCALES if use_zne else [1]

    caps, caps_are_real = load_device_caps()
    if not caps_are_real:
        print("NOTE: device_caps.json not found -- using a PLACEHOLDER square lattice.")
        print("      Run `python query_device_caps.py` on the DCV first; gate counts")
        print("      below are the right scale but not the real chip.\n")

    n_logical = N_TARG + N_PREC
    if n_logical > caps["qubit_count"]:
        print(f"ERROR: needs {n_logical} qubits, device has {caps['qubit_count']}.")
        sys.exit(1)

    # -- Unitary + circuit -----------------------------------------------------
    if STATE_MODE == "known":
        U = known_phase_unitary(N_TARG, PHI)
        ideal_bin = PHI * 2 ** N_PREC
        exact = abs(ideal_bin - round(ideal_bin)) < 1e-9
        qc = build_qpe_circuit(N_PREC, N_TARG, U, eigenstate=True)
    else:
        rng = np.random.default_rng(seed=SEED)
        U = haar_random_unitary(2 ** N_TARG, rng=rng)
        ideal_bin, exact = None, False
        qc = build_qpe_circuit(N_PREC, N_TARG, U, target_init_seed=TARGET_INIT_SEED)

    err = float(np.max(np.abs(U.conj().T @ U - np.eye(2 ** N_TARG))))
    if err > 1e-10:
        print(f"WARNING: |U'U - I|_max = {err:.2e} (expected ~1e-15)")

    print(f"Building + routing QPE (N_PREC={N_PREC}, N_TARG={N_TARG}) ...")
    qc_hw       = transpile_for_rigetti(qc, caps)
    phys_qubits = measured_physical_qubits(qc_hw)

    # -- Noiseless reference ---------------------------------------------------
    ideal = ideal_qpe_distribution(qc, N_PREC, N_TARG)

    print("\n" + "=" * 78)
    print("CONFIGURATION")
    print("=" * 78)
    print(f"  device            : {DEVICE_NAME}  ({DEVICE_ARN})")
    print(f"  submission        : {'reservation ' + RES_ARN if RES_ARN else 'ON-DEMAND (no reservation)'}")
    print(f"  circuit           : single-stage QPE, terminal measurement only")
    print(f"  N_PREC / N_TARG   : {N_PREC} / {N_TARG}   (U is {2**N_TARG}x{2**N_TARG})")
    print(f"  logical qubits    : {n_logical}  = N_TARG + N_PREC")
    print(f"  physical qubits   : {sorted(phys_qubits)}")
    print(f"  shots per circuit : {N_SHOTS:,}")
    print(f"  mitigation        : {MITIGATION}"
          + (f"   ZNE scales {scales}" if use_zne else "")
          + ("   REM 2 tasks" if use_rem else ""))
    print(f"  verbatim          : {USE_VERBATIM}")
    print(f"  state mode        : {STATE_MODE}")
    if STATE_MODE == "known":
        print(f"  phase PHI         : {PHI:.10g}  -> ideal bin {ideal_bin:g}"
              f"  ({'exactly representable' if exact else 'NOT exact in this many bits'})")
        if not exact:
            print(f"    PHI * 2^N_PREC is not an integer, so even a noiseless run")
            print(f"    spreads over neighbouring bins. Use a dyadic PHI (1/2, 1/4,")
            print(f"    1/8, ...) with N_PREC >= log2(1/PHI) for a single sharp bin.")
    else:
        print(f"  seeds             : U={SEED}, target init={TARGET_INIT_SEED}")

    if ideal is not None:
        print("\n  noiseless reference distribution (top bins):")
        for m, p in sorted(ideal.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    bin {m:>6}  (phase {m / 2**N_PREC:.6f})  p = {p:.4f}")
    else:
        print(f"\n  noiseless reference: skipped ({n_logical} qubits too many to simulate)")

    # -- Assemble the task list ------------------------------------------------
    jobs: list[dict] = []
    for scale in scales:
        folded = qc_hw if scale == 1 else fold_global(qc_hw, scale)
        jobs.append({"role": "qpe", "label": f"QPE lambda={scale}",
                     "zne_scale": scale, "circuit": folded,
                     "profile": count_native(folded)})

    if use_rem:
        for label, cal in readout_calibration_circuits(phys_qubits, qc_hw.num_qubits):
            jobs.append({"role": "readout_cal", "label": f"readout calibration {label}",
                         "zne_scale": 1, "cal_state": label, "circuit": cal,
                         "profile": count_native(cal)})

    # -- Report ----------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"CIRCUIT SPECIFICS -- {len(jobs)} task(s) will be submitted")
    print("=" * 78)

    total_runtime = 0.0
    total_gates   = 0
    for job in jobs:
        rt = print_circuit_report(job["label"], job["profile"], N_SHOTS,
                                  n_logical, caps_are_real,
                                  qubit_formula="n_targ + n_prec")
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
    print("  Runtime is dominated by per-task startup and the per-shot reset")
    print("  delay, both PLACEHOLDER constants in rigetti_qpuf_common.py. Run")
    print("  submit_test.py once and let checkRetrieve.py compare them against")
    print("  Rigetti's qpuRuntimeEstimation before sizing the reservation.")

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

    # DirectReservation routes every task submitted inside the block to the
    # reserved window instead of billing on-demand QPU time.
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
                tags=task_tags("qpe", {"Role": job["role"],
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
                "circuit_type":      f"QPE_{N_TARG}targ",
                "role":              job["role"],
                "label":             job["label"],
                "cal_state":         job.get("cal_state"),
                "zne_scale":         job["zne_scale"],
                "zne_scales_all":    scales,
                "error_mitigation":  MITIGATION,
                "verbatim":          USE_VERBATIM,
                "n_stages":          1,
                "state_mode":        STATE_MODE,
                "phi":               PHI if STATE_MODE == "known" else None,
                "ideal_bin":         ideal_bin if STATE_MODE == "known" else None,
                "ideal_distribution": ({str(k): v for k, v in ideal.items()}
                                       if ideal is not None else None),
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
    print("Retrieve with:  python checkRetrieve.py job_results_qpe")


if __name__ == "__main__":
    main()
