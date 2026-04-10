#!/usr/bin/env python3
"""
submit_QFT.py
=============
Usage: python submit_QFT.py <company>
  company: ionq | rigetti | iqm

For each device belonging to the chosen company, submits two QFT jobs:
  - n = max_qubits // 2   (half qubits)
  - n = max_qubits        (full qubits)

Appends one JSON record per submitted task to:
  qpu_benchmark/job_results/job_log.txt

Each record contains: task_id, submitted_at, device, device_arn,
circuit_type (QFT), n_qubits, n_shots.
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
S3_PREFIX = "qpu-benchmark"
SHOTS     = 256

def get_default_bucket() -> str:
    """Return the default Braket S3 bucket for this account and region."""
    region     = boto3.session.Session().region_name
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"amazon-braket-{region}-{account_id}"

JOB_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")

# ── Company → Device map ───────────────────────────────────────────────────────
# ARNs marked "← verify" are best-guess; confirm in the Braket console.
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
def build_qft(n: int) -> Circuit:
    """
    n-qubit QFT decomposed into H, CNOT, PhaseShift, and SWAP gates.

    CPhaseShift(θ) is avoided because some devices (e.g. Forte-Enterprise-1)
    do not support it. It is decomposed as:
      PhaseShift(θ/2) on control
      CNOT(control, target)
      PhaseShift(-θ/2) on target
      CNOT(control, target)
      PhaseShift(θ/2) on target
    """
    circ = Circuit()
    for i in range(n):
        circ.h(i)
        for j in range(i + 1, n):
            angle = np.pi / 2 ** (j - i)
            circ.phaseshift(j, angle / 2)
            circ.cnot(j, i)
            circ.phaseshift(i, -angle / 2)
            circ.cnot(j, i)
            circ.phaseshift(i, angle / 2)
    for i in range(n // 2):
        circ.swap(i, n - 1 - i)
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
        print("Usage: python submit_QFT.py <company>")
        print(f"  company: {' | '.join(COMPANY_DEVICES)}")
        sys.exit(1)

    company = sys.argv[1].lower()
    if company not in COMPANY_DEVICES:
        print(f"Unknown company '{company}'. Choose from: {', '.join(COMPANY_DEVICES)}")
        sys.exit(1)

    devices = COMPANY_DEVICES[company]
    print(f"Company   : {company}  ({len(devices)} device(s))")
    total_submitted = 0

    for dev_info in devices:
        name = dev_info["name"]
        arn  = dev_info["arn"]
        print(f"\n{'─'*60}")
        print(f"Device : {name}")
        print(f"ARN    : {arn}")

        # Query device capabilities
        try:
            device = AwsDevice(arn)
            caps   = device.properties
            if not hasattr(caps, "paradigm"):
                print("  Could not determine qubit count — skipping.")
                continue
            max_qubits = caps.paradigm.qubitCount
            status     = device.status
            print(f"  Status     : {status}")
            print(f"  Max qubits : {max_qubits}")
        except Exception as e:
            print(f"  ERROR [{name}] querying device: {e}")
            continue

        # Submit half-qubit and full-qubit jobs
        for n_qubits in [max_qubits // 2, max_qubits]:
            label = "half" if n_qubits == max_qubits // 2 else "full"
            print(f"\n  [{label}] Submitting QFT n={n_qubits} ...")

            circ = build_qft(n_qubits)

            submitted_at = datetime.now(timezone.utc).isoformat()
            try:
                s3_bucket = get_default_bucket()
                task    = device.run(
                    circ,
                    shots=SHOTS,
                    s3_destination_folder=(s3_bucket, f"{S3_PREFIX}/{name}"),
                )
                task_id = task.id
                print(f"  ✓ Task ID      : {task_id}")
                print(f"    Submitted at : {submitted_at}")
                print(f"    Qubits       : {n_qubits}  |  Shots: {SHOTS}  |  Type: QFT")

                record = {
                    "task_id":      task_id,
                    "submitted_at": submitted_at,
                    "device":       name,
                    "device_arn":   arn,
                    "circuit_type": "QFT",
                    "n_qubits":     n_qubits,
                    "n_shots":      SHOTS,
                }
                append_log(record)
                print(f"    Logged       → {LOG_FILE}")
                total_submitted += 1

            except Exception as e:
                print(f"  ✗ ERROR [{name}  n={n_qubits}  QFT] submission failed: {e}")

    print(f"\n{'='*60}")
    print(f"Total tasks submitted : {total_submitted}")
    print(f"Job log               : {LOG_FILE}")


if __name__ == "__main__":
    main()
