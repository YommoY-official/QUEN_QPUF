#!/usr/bin/env python3
"""
submit_ionq_mitigation.py
===========================
Submit single-stage Quantum Phase Estimation to IonQ Forte-Enterprise-1,
with or without REM/DDD/ZNE mitigation. ON-DEMAND submission (bare
AwsDevice(DEVICE_ARN).run(...), no Braket Direct reservation) -- adapted
from braket_direct/rigetti_cepheus_20260807/submit_qpe.py's interactive
structure and noise_mitigation/ionq_noise_mitig.py's proven on-demand
submission pattern.

Interactive: asks for precision qubits, target qubits, shots, which input
state to use, and whether to mitigate -- then prints the full circuit cost
and the noiseless reference distribution, and waits for confirmation BEFORE
submitting anything.

Why QPE rather than the two-stage QPUF
--------------------------------------
Same reasoning as the Rigetti campaign: QPE has ONE terminal measurement
and a directly computable answer (bin round(PHI * 2^n_prec)), so a
mitigation technique's effect is a comparison against ground truth, not a
comparison against another noisy measurement.

    qubits used = N_TARG + N_PREC

Mitigation (client-side; IonQ's own vendor-side EM here is `Debias`, which
is a SEPARATE, ORTHOGONAL mechanism already exercised in
noise_mitigation/ionq_noise_mitig.py and NOT combined with REM/DDD/ZNE in
this script -- mixing untested client-side folding/pulses with an IonQ-side
shot-symmetrization scheme would confound the two, so this campaign keeps
them apart)
-------------------------------------------------------------------------
  none  1 task.
  rem   + 2 readout-calibration tasks (all-|0>, all-|1>).
  zne   + one task per extra noise scale, via random unitary folding
        (fractional lambda). Whether IonQ's server-side compiler cancels
        the folds is UNCONFIRMED -- see MITIGATION_BRIEF_IONQ.md Section
        9.1. This script does not claim ZNE "works" here; it submits the
        circuits and logs everything needed to check afterwards (the
        monotonic-decay-toward-the-floor diagnostic in that section).
  both  zne + rem.

Results go to job_results/. Retrieve with:
    python checkRetrieve.py
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ionq_mitigation_common import (
    DEVICE_NAME, DEVICE_ARN, IONQ_BASIS, MAX_TOTAL_GATES,
    haar_random_unitary, known_phase_unitary,
    build_qpe_circuit, ideal_qpe_distribution,
    load_device_caps, transpile_for_ionq, count_native,
    measured_physical_qubits, to_braket_qasm,
    append_job_log, encode_unitary, task_tags, confirm_submit,
    estimate_fidelity, fmt_time, print_circuit_report,
)
from error_mitig_ionq import ErrorMitigation

# -- DEFAULTS (all overridable at the prompt) ----------------------------------
N_PREC      = 5            # precision qubits
N_TARG      = 1            # target qubits
N_SHOTS     = 1000

MITIGATION  = "both"       # none | rem | zne | both
STATE_MODE  = "known"      # known | haar
PHI         = 1 / 8        # phase to estimate, `known` mode only
ZNE_SCALES  = [1, 1.5, 2, 2.5, 3]   # fractional lambdas via random folding

SEED             = 10      # matches the IonQ QPUF scripts' Haar seed
TARGET_INIT_SEED = 99

ASSUME_YES = False         # set by --yes; skips the final confirmation

# Dynamical decoupling. A CLI FLAG, not a prompt -- same reasoning as
# rigetti's submit_qpe.py: keeps the interactive prompt sequence stable for
# any scripted/heredoc runs. "none" disables it.
DDD_SEQ    = "none"        # none | XX | YY | XY4 | XYXYX | II
DDD_CHOICES = ("none", "XX", "YY", "XY4", "XYXYX", "II")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "job_results")
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


def _parse_scales(s: str) -> list[float]:
    vals = [float(x) for x in s.replace(",", " ").split()]
    for v in vals:
        if v < 1:
            raise ValueError(f"ZNE scales must be >= 1; got {v}")
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

    `default` must already be the target type -- see the identical function
    in rigetti's submit_qpe.py for why passing a display string instead is
    a silent-str-leak trap.
    """
    shown = " ".join(map(str, default)) if isinstance(default, (list, tuple)) else default
    while True:
        try:
            raw = input(f"{label} [{shown}]: ").strip()
        except EOFError:
            print(f"\n  [no input available -- using default {shown}]")
            return default
        if raw == "":
            return default
        try:
            return cast(raw)
        except Exception as e:
            print(f"  invalid input ({e}); try again or press Enter for {shown}.")


def prompt_config():
    global N_PREC, N_TARG, N_SHOTS, MITIGATION, STATE_MODE, PHI, ZNE_SCALES

    print("=" * 78)
    print(f"QUANTUM PHASE ESTIMATION on {DEVICE_NAME}  (ON-DEMAND, no reservation)")
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
        ZNE_SCALES = _prompt("  ZNE noise scales (>= 1, fractional OK)",
                             ZNE_SCALES, _parse_scales)
    else:
        ZNE_SCALES = [1]

    print("\nInput state:")
    print("  known  exact eigenstate of a known-phase U -> single sharp bin")
    print("  haar   seeded Haar U + Haar target state (comparable to the QPUF run)")
    STATE_MODE = _prompt("State mode", STATE_MODE, _one_of(("known", "haar")))
    if STATE_MODE == "known":
        PHI = _prompt("  Phase PHI to estimate (e.g. 1/8)", PHI, _fraction)

    print(f"\n--> N_PREC={N_PREC}  N_TARG={N_TARG}  N_SHOTS={N_SHOTS}  "
          f"MITIGATION={MITIGATION}  STATE={STATE_MODE}"
          + (f" PHI={PHI:g}" if STATE_MODE == "known" else "") + "\n")

    if MITIGATION in ("zne", "both") or ZNE_SCALES != [1]:
        print("NOTE: whether IonQ's server-side compiler preserves ZNE folds")
        print("      and DD pulses unchanged is an OPEN, UNCONFIRMED question")
        print("      for this device -- MITIGATION_BRIEF_IONQ.md Section 9.1.")
        print("      This run submits and logs everything needed to check that")
        print("      afterwards; it does not assume the answer either way.\n")


# -- Main ----------------------------------------------------------------------

def main():
    global ASSUME_YES, DDD_SEQ
    ap = argparse.ArgumentParser(
        description="Submit single-stage QPE to Forte-Enterprise-1, on-demand.")
    ap.add_argument("--yes", action="store_true",
                    help="skip the final confirmation prompt and submit")
    ap.add_argument("--ddd", choices=DDD_CHOICES, default=DDD_SEQ,
                    help="dynamical-decoupling sequence to insert into idle "
                         "windows (default: none). XY4 is the recommended "
                         "sequence; II is the mandatory null control on this "
                         "device (see error_mitig_ionq.apply_ddd docstring). "
                         "Costs NO extra tasks.")
    args = ap.parse_args()
    ASSUME_YES = args.yes
    DDD_SEQ    = args.ddd

    # One id per invocation, stamped onto every task it submits -- lets the
    # analysis tell repeat batches of an identical config apart (same
    # reasoning as rigetti's submit_qpe.py).
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    prompt_config()

    use_zne = MITIGATION in ("zne", "both")
    use_rem = MITIGATION in ("rem", "both")
    scales  = ZNE_SCALES if use_zne else [1]

    caps, caps_are_real = load_device_caps()
    if not caps_are_real:
        print("NOTE: device_caps.json not found in this directory -- using the")
        print("      CLAUDE.md-stated 36-qubit all-to-all placeholder. Run")
        print("      query_device_caps.py to confirm against the live device.\n")

    n_logical = N_TARG + N_PREC
    if n_logical > caps["qubit_count"]:
        print(f"ERROR: needs {n_logical} qubits, device caps say {caps['qubit_count']}.")
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

    print(f"Building + transpiling QPE (N_PREC={N_PREC}, N_TARG={N_TARG}) ...")
    # ONE transpile pass, to native rz/rx/rxx, BEFORE any DDD/ZNE transform.
    # No coupling map / no placement stage needed here: Forte-Enterprise-1
    # is all-to-all (module docstring point 1 of ionq_mitigation_common.py).
    # Do NOT re-transpile after this point at optimization_level > 0 -- that
    # is exactly the ordering bug flagged in MITIGATION_BRIEF_IONQ.md
    # Section 9.1, "what would settle it", item 3: our own pipeline already
    # transpiles at level 1 elsewhere (ionq_noise_mitig.py, submit_QPUF_
    # ntarg.py), and folding/DD insertion must happen strictly after the
    # final transpile.
    qc_hw = transpile_for_ionq(qc, optimization_level=1)

    # -- DDD, BEFORE folding ---------------------------------------------------
    # Order matters: decoupling the base circuit and then folding it means
    # every lambda scales the noise of ONE well-defined circuit C_ddd.
    # Folding first would decouple the folded circuit's own idle structure
    # instead, making DDD and ZNE stop being independent factors.
    ddd_report = None
    if DDD_SEQ != "none":
        em = ErrorMitigation()
        before = qc_hw
        qc_hw = em.apply_ddd(qc_hw, sequence=DDD_SEQ, verbose=True)
        ddd_report = em.ddd_report(qc_hw)
        check = em.verify_ddd(before, qc_hw)
        print(f"  DDD {DDD_SEQ}: {check}")
        if not check.get("ok", True):
            print("ERROR: DDD verification failed -- the transform did not add "
                  "the gates it claims. Not submitting.")
            sys.exit(1)
        print("  NOTE: this only confirms the pulses are in the SUBMITTED QASM.")
        print("        Whether they survive IonQ's server-side compilation is")
        print("        unverifiable with this repo's current metadata retrieval")
        print("        (MITIGATION_BRIEF_IONQ.md Section 9.1).")

    phys_qubits = measured_physical_qubits(qc_hw)

    # -- Noiseless reference ---------------------------------------------------
    ideal = ideal_qpe_distribution(qc, N_PREC, N_TARG)

    print("\n" + "=" * 78)
    print("CONFIGURATION")
    print("=" * 78)
    print(f"  device            : {DEVICE_NAME}  ({DEVICE_ARN})")
    print(f"  submission        : ON-DEMAND (no Braket Direct reservation)")
    print(f"  circuit           : single-stage QPE, terminal measurement only")
    print(f"  N_PREC / N_TARG   : {N_PREC} / {N_TARG}   (U is {2**N_TARG}x{2**N_TARG})")
    print(f"  logical qubits    : {n_logical}  = N_TARG + N_PREC")
    print(f"  shots per circuit : {N_SHOTS:,}")
    print(f"  mitigation        : {MITIGATION}"
          + (f"   ZNE scales {scales}" if use_zne else "")
          + ("   REM 2 tasks" if use_rem else ""))
    print(f"  DDD               : {DDD_SEQ}"
          + (f"   {ddd_report.get('n_pulses', '?')} pulses in "
             f"{ddd_report.get('n_windows', '?')} idle windows (0 extra tasks)"
             if ddd_report else "   (no decoupling)"))
    print(f"  batch id          : {batch_id}")
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
    em = ErrorMitigation()
    jobs: list[dict] = []
    for scale in scales:
        folded = qc_hw if scale == 1 else em.fold(qc_hw, scale)
        jobs.append({"role": "qpe", "label": f"QPE lambda={scale}",
                     "zne_scale": scale, "circuit": folded,
                     "profile": count_native(folded)})

    if use_rem:
        for label, cal in em.rem_calibration_circuits(qc_hw, model="tensored"):
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
        job["runtime_s"] = rt["runtime_s"]
        total_runtime += rt["runtime_s"]
        total_gates   += rt["total_gates"]
        if rt["total_gates"] > MAX_TOTAL_GATES:
            print(f"\nERROR: task '{job['label']}' has {rt['total_gates']:,} "
                  f"total gates, over the self-imposed cap {MAX_TOTAL_GATES:,} "
                  f"(see ionq_mitigation_common.MAX_TOTAL_GATES -- this cap is "
                  f"a project budget guess, not a cited AWS/IonQ platform "
                  f"limit). Reduce N_PREC, N_SHOTS, or the ZNE scale.")
            sys.exit(1)

    print("-" * 78)
    print("TOTALS")
    print("-" * 78)
    print(f"  tasks                        : {len(jobs)}")
    print(f"  shots submitted              : {len(jobs) * N_SHOTS:,}")
    print(f"  total gate-ops executed      : {total_gates:,}  [cap {MAX_TOTAL_GATES:,}]")
    print(f"  est. total runtime           : {fmt_time(total_runtime)}")
    print(f"  est. total runtime (+30% pad): {fmt_time(total_runtime * 1.3)}")
    print()
    print("  This is an ON-DEMAND submission -- each task is billed per task,")
    print("  and on-demand tasks may queue behind other users' work (unlike a")
    print("  reserved window). Timing constants above are UNVALIDATED estimates")
    print("  (MITIGATION_BRIEF_IONQ.md Section 9.2), not a calibration fit.")

    if not confirm_submit(
            f"Submit {len(jobs)} task(s) to {DEVICE_NAME}? [y/N]: ", ASSUME_YES):
        return

    # -- Submit ----------------------------------------------------------------
    try:
        from braket.aws import AwsDevice
        from braket.ir.openqasm import Program as OpenQasmProgram
    except ImportError as e:
        print(f"\nERROR: missing dependency: {e}\n       pip install amazon-braket-sdk qiskit")
        sys.exit(1)

    device = AwsDevice(DEVICE_ARN)
    print(f"\nDevice resolved : {device.name}  (status {device.status})")

    device_n_qubits = device.properties.paradigm.qubitCount
    if n_logical > device_n_qubits:
        print(f"\nERROR: circuit needs {n_logical} qubits but {DEVICE_NAME} "
              f"only has {device_n_qubits}.")
        sys.exit(1)
    print(f"Device qubits    : {device_n_qubits}  (circuit uses {n_logical})")

    submitted = []
    log_file  = os.path.join(RESULTS_DIR, "job_log.txt")
    for job in jobs:
        # verbatim=False by default: see to_braket_qasm's docstring for why
        # this campaign does not default to a verbatim box (Section 9.1).
        qasm_src = to_braket_qasm(job["circuit"], verbatim=False)
        print(f"\nSubmitting: {job['label']} ({N_SHOTS} shots) ...")
        task = device.run(
            OpenQasmProgram(source=qasm_src), shots=N_SHOTS,
            tags=task_tags("qpe_mitigation", {"Role": job["role"],
                                              "ZneScale": job["zne_scale"],
                                              "Ddd": DDD_SEQ}),
        )
        submitted_at = datetime.now(timezone.utc).isoformat()
        print(f"  Task ARN : {task.id}")

        prof = job["profile"]
        record = {
            "job_id":            task.id,
            "datetime":          submitted_at,
            "qpu":               DEVICE_NAME,
            "device_arn":        DEVICE_ARN,
            "submission":        "on-demand",
            "circuit_type":      f"QPE_{N_TARG}targ",
            "role":              job["role"],
            "label":             job["label"],
            "cal_state":         job.get("cal_state"),
            "zne_scale":         job["zne_scale"],
            "zne_scales_all":    scales,
            "error_mitigation":  MITIGATION,
            "ddd_sequence":      DDD_SEQ,
            "ddd_report":        ddd_report,
            "batch_id":          batch_id,
            "verbatim":          False,
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
            # UNSORTED, position = clbit index -- see mitig_analysis.py's
            # bit-ordering docstring for why this must be logged explicitly.
            "clbit_to_phys":     list(phys_qubits),
            "n_shots":           N_SHOTS,
            "n_1q_gates":        prof["n_1q"],
            "n_2q_gates":        prof["n_2q"],
            "n_gates":           prof["n_gates"],
            "total_gates":       prof["n_gates"] * N_SHOTS,
            "n_measurements":    prof["n_meas"],
            "est_fidelity":      estimate_fidelity(prof),
            "est_runtime_s":     job["runtime_s"],
            "seed":              SEED,
            "target_init_seed":  TARGET_INIT_SEED,
            "unitary":           encode_unitary(U),
        }
        log_file = append_job_log(RESULTS_DIR, record)
        submitted.append(task.id)

    print(f"\n{len(submitted)}/{len(jobs)} task(s) submitted.")
    print(f"Logged to: {log_file}")
    print("Retrieve with:  python checkRetrieve.py")


if __name__ == "__main__":
    main()
