#!/usr/bin/env python3
"""
submit_GHZ.py
=============
Usage: python submit_GHZ.py <company>
  company: ionq | rigetti | iqm

For each device belonging to the chosen company, submits two GHZ jobs:
  - n = max_qubits // 2   (half qubits)
  - n = max_qubits        (full qubits)

GHZ state: (|00...0⟩ + |11...1⟩) / sqrt(2)

Verification is classical (post-measurement):
  ideal_fraction = fraction of shots that are |00...0⟩ or |11...1⟩
  (computed in checkRetieve.py when results are retrieved)

Appends one JSON record per submitted task to:
  qpu_benchmark/job_results/job_log.txt

Each record contains: task_id, submitted_at, device, device_arn,
circuit_type (GHZ), n_qubits, n_shots.
"""

import sys
import json
import os
from datetime import datetime, timezone

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
def build_ghz(n: int) -> Circuit:
    """n-qubit GHZ state: H on qubit 0, then CNOT chain."""
    circ = Circuit()
    circ.h(0)
    for i in range(n - 1):
        circ.cnot(i, i + 1)
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
        print("Usage: python submit_GHZ.py <company>")
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
            print(f"\n  [{label}] Submitting GHZ n={n_qubits} ...")

            circ = build_ghz(n_qubits)

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
                print(f"    Qubits       : {n_qubits}  |  Shots: {SHOTS}  |  Type: GHZ")
                print(f"    Ideal states : |{'0'*n_qubits}⟩ and |{'1'*n_qubits}⟩")

                record = {
                    "task_id":      task_id,
                    "submitted_at": submitted_at,
                    "device":       name,
                    "device_arn":   arn,
                    "circuit_type": "GHZ",
                    "n_qubits":     n_qubits,
                    "n_shots":      SHOTS,
                }
                append_log(record)
                print(f"    Logged       → {LOG_FILE}")
                total_submitted += 1

            except Exception as e:
                print(f"  ✗ ERROR [{name}  n={n_qubits}  GHZ] submission failed: {e}")

    print(f"\n{'='*60}")
    print(f"Total tasks submitted : {total_submitted}")
    print(f"Job log               : {LOG_FILE}")
    print(f"\nNote: GHZ verification (ideal fraction) is computed automatically")
    print(f"      by checkRetieve.py when results are retrieved.")


if __name__ == "__main__":
    main()
