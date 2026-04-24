#!/usr/bin/env python3
"""
submit_QPE.py  (CORRECTED)
===========================
Fixes two compounding bugs in the original build_qpe():

  BUG 1 — Controlled-power ordering.
    The original code applied controlled-U^(2^k) with counting qubit k as
    control (so qubit 0 was LSB). For QPE where qubit 0 is read as the MSB of
    the output integer (the standard convention, and the convention implicit
    when measurement results are interpreted as integers with qubit 0
    leftmost), counting qubit k must control U^(2^(n_counting-1-k)).

  BUG 2 — Inverse-QFT swap position.
    The original code placed the SWAP cascade at the END of the inverse QFT.
    Because the forward QFT itself ends with SWAPs, its inverse must BEGIN
    with SWAPs. Placing swaps at the end of the inverse QFT left a residual
    bit-reversal in the output.

The two bugs compose to a double bit-reversal, which becomes the identity
permutation only when the output is symmetric under bit-reversal. For
phi = 1/8 the ideal bin is 2^(n_counting) / 8 = 2^(n_counting-3); this
accidentally coincides with the bug-induced bin (which is always 4) only at
n_counting = 5. That explains why the local-simulator sanity check for n=5
passed, and why every hardware run for other n values showed peak_acc ≈ 0.

Usage: python submit_QPE.py <company>
  company: ionq | aqt | rigetti | iqm
"""

import sys
import json
import os
from datetime import datetime, timezone

import numpy as np
import boto3
from braket.aws import AwsDevice
from braket.circuits import Circuit

# ── Configuration ──────────────────────────────────────────────────────────────
S3_PREFIX    = "qpu-benchmark"
SHOTS        = 256
PHASE_TARGET = 1 / 8   # T gate eigenvalue phase: e^(2πi * 1/8)

def get_default_bucket() -> str:
    region     = boto3.session.Session().region_name
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"amazon-braket-{region}-{account_id}"

JOB_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")

COMPANY_DEVICES = {
    "ionq": [
        {"name": "Forte-1",
         "arn":  "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"},
        {"name": "Forte-Enterprise-1",
         "arn":  "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"},
    ],
    "aqt": [
        {"name": "Ibex-Q1",
         "arn":  "arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1"},
    ],
    "rigetti": [
        {"name": "Ankaa-3",
         "arn":  "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"},
        {"name": "Cepheus-1-108Q",
         "arn":  "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"},
    ],
    "iqm": [
        {"name": "Garnet",
         "arn":  "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"},
        {"name": "Emerald",
         "arn":  "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald"},
    ],
}


# ── Gate decompositions ────────────────────────────────────────────────────────
def _cphase(circ: Circuit, control: int, target: int, angle: float):
    """
    CPhaseShift(angle) decomposed into Rz + CNOT. Correct up to global phase.
    Applies e^{i*angle} when both qubits are |1⟩.
    """
    circ.rz(control, angle / 2)
    circ.cnot(control, target)
    circ.rz(target, -angle / 2)
    circ.cnot(control, target)
    circ.rz(target, angle / 2)


# ── Circuit builder (CORRECTED) ────────────────────────────────────────────────
def build_qpe(n_counting: int) -> Circuit:
    """
    Textbook QPE for U = T (phase φ = 1/8), eigenstate |1⟩.

    Layout:
      counting register : qubits 0 .. n_counting-1  (qubit 0 is MSB)
      target qubit      : qubit  n_counting          (initialized to |1⟩)

    After execution, the measurement bitstring read as an integer with qubit 0
    as the most significant bit gives the QPE estimate m, where m / 2^n_counting
    approximates the eigenphase φ. For φ = 1/8 this is exact for n_counting ≥ 3,
    so the ideal peak bin is 2^n_counting / 8.
    """
    target   = n_counting
    counting = list(range(n_counting))

    circ = Circuit()

    # Prepare eigenstate |1⟩ on target qubit
    circ.x(target)

    # Hadamard on all counting qubits
    for k in counting:
        circ.h(k)

    # Controlled-T^(2^(n_counting-1-k)) for counting qubit k.
    # Qubit 0 (MSB) controls the largest power; qubit n_counting-1 (LSB) controls T^1.
    for k in range(n_counting):
        power = 2 ** (n_counting - 1 - k)
        angle = (np.pi / 4) * power % (2 * np.pi)
        _cphase(circ, counting[k], target, angle)

    # Inverse QFT on the counting register.
    # Forward QFT = (H + CR cascade) then SWAPs, so QFT^{-1} = SWAPs then
    # reverse of (H + CR cascade) with negated rotation angles.

    # 1) SWAPs first
    for i in range(n_counting // 2):
        circ.swap(counting[i], counting[n_counting - 1 - i])

    # 2) Reverse cascade: for i from n-1 down to 0, apply CR^dagger with all j>i,
    #    then H on qubit i.
    for i in range(n_counting - 1, -1, -1):
        for j in range(n_counting - 1, i, -1):
            angle = -np.pi / 2 ** (j - i)
            _cphase(circ, counting[j], counting[i], angle)
        circ.h(counting[i])

    return circ


# ── Log helper ─────────────────────────────────────────────────────────────────
def append_log(record: dict):
    os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        print("Usage: python submit_QPE.py <company>")
        print(f"  company: {' | '.join(COMPANY_DEVICES)}")
        sys.exit(1)

    company = sys.argv[1].lower()
    if company not in COMPANY_DEVICES:
        print(f"Unknown company '{company}'. Choose from: {', '.join(COMPANY_DEVICES)}")
        sys.exit(1)

    device_list = COMPANY_DEVICES[company]
    print(f"\nCompany : {company}  ({len(device_list)} device(s))")
    print(f"Querying device status...\n")

    online_devices = []
    for dev_info in device_list:
        name = dev_info["name"]
        arn  = dev_info["arn"]
        try:
            device     = AwsDevice(arn)
            caps       = device.properties
            max_qubits = caps.paradigm.qubitCount if hasattr(caps, "paradigm") else None
            status     = device.status
        except Exception as e:
            print(f"  [{name}] ERROR querying device: {e}")
            continue

        qubit_str = f"{max_qubits} qubits" if max_qubits is not None else "? qubits"
        print(f"  {name:<30} | {status:<8} | {qubit_str}")

        if status == "ONLINE" and max_qubits is not None:
            online_devices.append((name, arn, max_qubits, device))

    if not online_devices:
        print("\nNo online devices found. Exiting.")
        sys.exit(0)

    print(f"\nOnline devices:")
    for idx, (name, arn, max_qubits, _) in enumerate(online_devices, start=1):
        print(f"  [{idx}] {name}  ({max_qubits} qubits)")

    while True:
        try:
            choice = int(input(f"\nSelect device [1-{len(online_devices)}]: "))
            if 1 <= choice <= len(online_devices):
                break
            print(f"  Please enter a number between 1 and {len(online_devices)}.")
        except ValueError:
            print("  Invalid input — enter an integer.")

    sel_name, sel_arn, sel_max_qubits, sel_device = online_devices[choice - 1]
    print(f"\nSelected : {sel_name}  (max {sel_max_qubits} qubits)")
    print(f"Note     : 1 qubit is reserved as the QPE target; "
          f"n_counting = n_total - 1.")

    while True:
        try:
            n_total = int(input(f"Enter number of qubits to use [2-{sel_max_qubits}]: "))
            if 2 <= n_total <= sel_max_qubits:
                break
            print(f"  Must be between 2 and {sel_max_qubits}.")
        except ValueError:
            print("  Invalid input — enter an integer.")

    n_counting = n_total - 1

    print(f"\nSubmitting QPE circuit:")
    print(f"  Device     : {sel_name}")
    print(f"  n_total    : {n_total}  (n_counting={n_counting}, target=1)")
    print(f"  Shots      : {SHOTS}")
    print(f"  Phase φ    : {PHASE_TARGET}  (T gate, ideal bin: "
          f"{round(2**n_counting * PHASE_TARGET)})")

    circ = build_qpe(n_counting)
    submitted_at = datetime.now(timezone.utc).isoformat()

    try:
        s3_bucket = get_default_bucket()
        task      = sel_device.run(
            circ,
            shots=SHOTS,
            s3_destination_folder=(s3_bucket, f"{S3_PREFIX}/{sel_name}"),
        )
        task_id = task.id
        print(f"\n  ✓ Task ID      : {task_id}")
        print(f"    Submitted at : {submitted_at}")

        record = {
            "task_id":      task_id,
            "submitted_at": submitted_at,
            "device":       sel_name,
            "device_arn":   sel_arn,
            "circuit_type": "QPE",
            "n_qubits":     n_total,
            "n_counting":   n_counting,
            "n_shots":      SHOTS,
            "phase_target": PHASE_TARGET,
        }
        append_log(record)
        print(f"    Logged       → {LOG_FILE}")

    except Exception as e:
        print(f"\n  ✗ ERROR submission failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()