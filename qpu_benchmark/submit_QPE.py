#!/usr/bin/env python3
"""
submit_QPE.py
=============
Usage: python submit_QPE.py <company>
  company: ionq | aqt | rigetti | iqm

Interactive flow:
  1. Lists all devices for the chosen company with status and qubit count.
  2. Prompts user to select an ONLINE device.
  3. Prompts user to enter the number of qubits (n_total; 1 qubit is reserved
     as the target, so n_counting = n_total - 1).
  4. Submits one QPE job and logs the result.

Quantum Phase Estimation circuit:
  Unitary  : T gate  (U = T, phase φ = 1/8, eigenstate |1⟩)
  Counting qubits : n_counting = n_total - 1
  Target qubit    : 1 qubit (initialized to |1⟩, eigenstate of T)

  Circuit steps:
    1. X on target qubit  → prepare eigenstate |1⟩
    2. H on each counting qubit
    3. Controlled-T^(2^k) for counting qubit k  (phase kickback)
    4. Inverse QFT on counting qubits
    5. Measure all (counting qubits encode the estimated phase)

  All gates decomposed into H, X, CNOT, Rz — supported on all Braket QPUs.
  CPhaseShift(θ) decomposition:
    Rz(θ/2)  on control
    CNOT(control, target)
    Rz(-θ/2) on target
    CNOT(control, target)
    Rz(θ/2)  on target

Appends one JSON record per submitted task to:
  qpu_benchmark/job_results/job_log.txt

Each record contains: task_id, submitted_at, device, device_arn,
circuit_type (QPE), n_qubits, n_counting, n_shots, phase_target.
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
    """Return the default Braket S3 bucket for this account and region."""
    region     = boto3.session.Session().region_name
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"amazon-braket-{region}-{account_id}"

JOB_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")

# ── Company → Device map ───────────────────────────────────────────────────────
COMPANY_DEVICES = {
    "ionq": [
        {
            "name": "Forte-1",
            "arn":  "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
        },
        {
            "name": "Forte-Enterprise-1",
            "arn":  "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1",
        },
    ],
    "aqt": [
        {
            "name": "Ibex-Q1",
            "arn":  "arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1",
        },
    ],
    "rigetti": [
        {
            "name": "Ankaa-3",
            "arn":  "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
        },
        {
            "name": "Cepheus-1-108Q",
            "arn":  "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
        },
    ],
    "iqm": [
        {
            "name": "Garnet",
            "arn":  "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
        },
        {
            "name": "Emerald",
            "arn":  "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald",  # ← verify
        },
    ],
}

# ── Circuit builder ────────────────────────────────────────────────────────────
def _cphase(circ: Circuit, control: int, target: int, angle: float):
    """
    Decomposed CPhaseShift(angle) using CNOT and Rz only.
    Applies phase e^(i*angle) when both qubits are |1⟩.
    """
    circ.rz(control, angle / 2)
    circ.cnot(control, target)
    circ.rz(target, -angle / 2)
    circ.cnot(control, target)
    circ.rz(target, angle / 2)


def build_qpe(n_counting: int) -> Circuit:
    """
    QPE circuit for the T gate (phase φ = 1/8).

    Layout:
      qubits 0 .. n_counting-1  : counting register
      qubit  n_counting          : target (eigenstate qubit)
    """
    target   = n_counting
    counting = list(range(n_counting))

    circ = Circuit()

    # Step 1 — Prepare eigenstate |1⟩ on target qubit
    circ.x(target)

    # Step 2 — Hadamard on all counting qubits
    for k in counting:
        circ.h(k)

    # Step 3 — Controlled-T^(2^k) for each counting qubit k
    for k in range(n_counting):
        angle = (np.pi / 4) * (2 ** k) % (2 * np.pi)
        _cphase(circ, counting[k], target, angle)

    # Step 4 — Inverse QFT on counting qubits
    # Rotations + Hadamards FIRST
    for i in range(n_counting - 1, -1, -1):
        for j in range(n_counting - 1, i, -1):
            angle = -np.pi / 2 ** (j - i)
            _cphase(circ, counting[j], counting[i], angle)
        circ.h(counting[i])

    # Bit-reversal swaps LAST
    for i in range(n_counting // 2):
        circ.swap(counting[i], counting[n_counting - 1 - i])

    return circ


# ── Log helper ─────────────────────────────────────────────────────────────────
def append_log(record: dict):
    """Append one JSON record (one line) to job_log.txt."""
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

    # ── Phase 1: discover devices ──────────────────────────────────────────────
    online_devices = []   # list of (name, arn, max_qubits, AwsDevice)

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

    # ── Phase 2: select device ─────────────────────────────────────────────────
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

    # ── Phase 3: select qubit count ────────────────────────────────────────────
    while True:
        try:
            n_total = int(input(f"Enter number of qubits to use [2-{sel_max_qubits}]: "))
            if 2 <= n_total <= sel_max_qubits:
                break
            print(f"  Must be between 2 and {sel_max_qubits}.")
        except ValueError:
            print("  Invalid input — enter an integer.")

    n_counting = n_total - 1

    # ── Phase 4: submit ────────────────────────────────────────────────────────
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
