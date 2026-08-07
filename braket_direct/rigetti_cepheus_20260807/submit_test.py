#!/usr/bin/env python3
"""
submit_test.py
==============
End-to-end pipeline smoke test for the Rigetti Cepheus-1-108Q QPUF run.

This submits a DELIBERATELY TINY version of the real circuit (default
N_PREC=2, N_TARG=1 -> 5 logical qubits) through exactly the same code path
that submit_qpuf_mitigation.py uses: same builder, same transpile, same
OpenQASM 3 export, same job_log format. If this works, the only thing that
changes for the real run is the size.

What it is checking
-------------------
  1. AWS credentials + region resolve, and the device is reachable.
  2. Braket's OpenQASM 3 parser accepts our exported QASM (this is the
     step that historically breaks: no `include`, no user-defined `gate`
     blocks, Braket built-ins only).
  3. If verbatim mode is requested -- that Braket accepts the
     `#pragma braket verbatim box` form on physical qubits. This matters
     because ZNE folding is worthless without it (Quilc would cancel the
     folds).
  4. The reservation ARN routes the task to the reserved window.
  5. job_log.txt is written in the shape checkRetrieve.py expects.

Targets
-------
  local  : LocalSimulator("braket_sv") -- free, instant, validates the QASM
           only (no routing, no verbatim).
  sv1    : Braket SV1 managed simulator -- free-ish, validates the full
           submit/retrieve round trip including task ARNs and S3.
  qpu    : Cepheus on-demand -- real hardware, real cost, no reservation.
  res    : Cepheus inside the Braket Direct reservation (needs RES_ARN).

Run `python checkRetrieve.py job_results_test` afterwards to pull results.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
from qiskit import transpile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigetti_qpuf_common import (
    DEVICE_NAME, DEVICE_ARN, RES_ARN, RIGETTI_BASIS,
    haar_random_unitary, build_qpuf_two_stage,
    load_device_caps, transpile_for_rigetti, count_native,
    check_qubits_on_device, native_gate_violations,
    to_braket_qasm, append_job_log, encode_unitary,
    task_tags, report_reservation,
    print_circuit_report,
)

# -- TEST CONFIGURATION --------------------------------------------------------
N_PREC  = 2        # precision qubits PER stage -> 2*N_PREC + N_TARG total
N_TARG  = 1
N_SHOTS = 100

SEED             = 10
TARGET_INIT_SEED = 99

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "job_results_test")
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
# ------------------------------------------------------------------------------


def choose(label: str, options: dict, default: str) -> str:
    """Prompt for one of `options` (key -> description); Enter keeps default."""
    print(f"\n{label}")
    for k, desc in options.items():
        mark = " (default)" if k == default else ""
        print(f"  {k:<6} {desc}{mark}")
    try:
        raw = input(f"choice [{default}]: ").strip().lower()
    except EOFError:
        return default
    return raw if raw in options else default


def main():
    target = choose(
        "Where should the test job go?",
        {
            "local": "LocalSimulator braket_sv  -- free, validates QASM only",
            "sv1":   "Braket SV1 simulator      -- validates full submit/retrieve",
            "qpu":   f"{DEVICE_NAME} on-demand   -- real hardware, billed per task",
            "res":   f"{DEVICE_NAME} reservation -- needs RES_ARN set",
        },
        default="local",
    )

    on_hardware = target in ("qpu", "res")

    verbatim = False
    if on_hardware:
        vb = choose(
            "Submit inside a verbatim box? (ZNE needs this; test it now)",
            {"n": "no  -- let Quilc compile/route (normal path)",
             "y": "yes -- #pragma braket verbatim, physical qubits, no recompilation"},
            default="n",
        )
        verbatim = (vb == "y")

    if target == "res" and not RES_ARN:
        print("\nERROR: RES_ARN is empty in rigetti_qpuf_common.py -- "
              "fill in the reservation ARN or choose 'qpu'.")
        sys.exit(1)

    # -- Build -----------------------------------------------------------------
    caps, caps_are_real = load_device_caps()
    n_logical = N_TARG + 2 * N_PREC

    rng = np.random.default_rng(seed=SEED)
    U   = haar_random_unitary(2 ** N_TARG, rng=rng)
    err = float(np.max(np.abs(U.conj().T @ U - np.eye(2 ** N_TARG))))
    if err > 1e-10:
        print(f"WARNING: |U'U - I|_max = {err:.2e} (expected ~1e-15)")

    qc = build_qpuf_two_stage(N_PREC, N_TARG, U, target_init_seed=TARGET_INIT_SEED)

    print("\n" + "=" * 78)
    print(f"PIPELINE TEST -- two-stage PE-QPUF, N_PREC={N_PREC}, N_TARG={N_TARG}, "
          f"{N_SHOTS} shots")
    print("=" * 78)
    print(f"target       : {target}"
          + (f"   (verbatim={verbatim})" if on_hardware else ""))
    print(f"device caps  : {'device_caps.json' if caps_are_real else 'PLACEHOLDER lattice'}")

    if on_hardware:
        # Route onto the real lattice; the result addresses physical qubits.
        qc_hw = transpile_for_rigetti(qc, caps)
    else:
        # Simulators have no lattice; routing onto 108 qubits would make the
        # statevector unsimulable. Decompose to the native basis only, so the
        # circuit stays at n_logical qubits and the QASM is still exercised.
        qc_hw = transpile(qc, basis_gates=RIGETTI_BASIS + ["measure"],
                          optimization_level=1)

    profile = count_native(qc_hw)
    print_circuit_report("TEST CIRCUIT", profile, N_SHOTS, n_logical, caps_are_real)

    # Physical qubit addressing on hardware only -- we routed the circuit
    # ourselves, so those indices are the ones we want executed. Simulators
    # have no physical qubits, so they get the virtual register.
    qasm_src = to_braket_qasm(qc_hw, verbatim=verbatim, physical=on_hardware)
    print(f"\nOpenQASM 3 : {len(qasm_src.splitlines())} lines, "
          f"{len(qasm_src)} chars")
    print("--- first 15 lines ---")
    for line in qasm_src.splitlines()[:15]:
        print("  " + line)
    print("  ...")

    try:
        resp = input(f"\nSubmit this test job to '{target}'? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted -- nothing submitted.")
        return

    # -- Submit ----------------------------------------------------------------
    try:
        from braket.ir.openqasm import Program as OpenQasmProgram
    except ImportError as e:
        print(f"\nERROR: missing dependency: {e}\n       pip install amazon-braket-sdk qiskit")
        sys.exit(1)

    program = OpenQasmProgram(source=qasm_src)

    if target == "local":
        from braket.devices import LocalSimulator
        print("\nRunning on LocalSimulator(braket_sv) ...")
        result = LocalSimulator("braket_sv").run(program, shots=N_SHOTS).result()
        counts = dict(result.measurement_counts)
        print(f"OK -- {sum(counts.values())} shots, {len(counts)} unique outcomes.")
        for bits, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {bits} : {n}")
        print("\nLocal run only -- nothing logged (no task ARN to retrieve).")
        return

    from braket.aws import AwsDevice

    if target == "sv1":
        device, dev_arn, dev_name = AwsDevice(SV1_ARN), SV1_ARN, "SV1"
    else:
        device, dev_arn, dev_name = AwsDevice(DEVICE_ARN), DEVICE_ARN, DEVICE_NAME
        print(f"\nDevice resolved : {device.name}  (status {device.status})")
        # Check the qubits the circuit TOUCHES, not its width -- see
        # check_qubits_on_device(). qc_hw.num_qubits is 108 for every routed
        # circuit (top physical index 107 + 1), which says nothing about
        # whether 107 real qubits are enough.
        try:
            used = check_qubits_on_device(qc_hw, caps)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        print(f"Physical qubits : {used}  ({len(used)} of "
              f"{device.properties.paradigm.qubitCount} live)")

        # Verbatim pre-flight -- a verbatim box runs EXACTLY as written, so one
        # non-native gate rejects the whole program. Proving this passes is a
        # main point of the test job: ZNE is worthless without verbatim.
        if verbatim:
            bad = native_gate_violations(qc_hw, caps)
            if bad:
                print("\nERROR: verbatim requested, but these are not natively "
                      "executable:")
                for v in bad:
                    print(f"    {v}")
                sys.exit(1)
            print("Verbatim check  : OK -- circuit is native")

    tags = task_tags("test", {"Target": target, "Verbatim": verbatim})

    print(f"\nSubmitting {N_SHOTS} shots to {dev_name} ...")
    if target == "res":
        from braket.aws import DirectReservation
        with DirectReservation(device, reservation_arn=RES_ARN):
            task = device.run(program, shots=N_SHOTS, tags=tags)
    else:
        task = device.run(program, shots=N_SHOTS, tags=tags)

    submitted_at = datetime.now(timezone.utc).isoformat()
    print(f"Job submitted.\n  Task ARN  : {task.id}\n  Timestamp : {submitted_at}")
    if target == "res":
        report_reservation(task, RES_ARN)

    record = {
        "job_id":            task.id,
        "datetime":          submitted_at,
        "qpu":               dev_name,
        "device_arn":        dev_arn,
        "submission":        "reservation" if target == "res" else "on-demand",
        "reservation_arn":   RES_ARN if target == "res" else None,
        "circuit_type":      "QPUF2stage_TEST",
        "role":              "test",
        "verbatim":          verbatim,
        "zne_scale":         1,
        "error_mitigation":  "none",
        "n_stages":          2,
        "n_prec":            N_PREC,
        "n_targ":            N_TARG,
        "n_qubits":          n_logical,
        "n_shots":           N_SHOTS,
        "n_1q_gates":        profile["n_1q"],
        "n_2q_gates":        profile["n_2q"],
        "n_gates":           profile["n_gates"],
        "n_measurements":    profile["n_meas"],
        "depth":             profile["depth"],
        "depth_2q":          profile["depth_2q"],
        "seed":              SEED,
        "target_init_seed":  TARGET_INIT_SEED,
        "unitary":           encode_unitary(U),
    }
    log_file = append_job_log(RESULTS_DIR, record)
    print(f"  Logged to : {log_file}")
    print(f"\nNext: python checkRetrieve.py job_results_test")


if __name__ == "__main__":
    main()
