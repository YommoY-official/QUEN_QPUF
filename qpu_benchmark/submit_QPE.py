#!/usr/bin/env python3
"""
submit_QPE.py
=============
Usage: python submit_QPE.py <company>
  company: ionq | aqt | rigetti | iqm

For each device belonging to the chosen company, submits two QPE jobs:
  - n_total = max_qubits // 2   (half qubits)
  - n_total = max_qubits        (full qubits)

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
  CPhaseShift(θ) decomposition (same as submit_QFT.py):
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
    Same decomposition as submit_QFT.py.
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

    The ideal measurement outcome on counting qubits is the binary
    representation of round(2^n_counting * φ) = 2^(n_counting - 3)
    for n_counting >= 3.
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
    # T^(2^k) has phase e^(i * 2^k * π/4); wraps mod 2π for large k.
    for k in range(n_counting):
        angle = (np.pi / 4) * (2 ** k) % (2 * np.pi)
        _cphase(circ, counting[k], target, angle)

    # Step 4 — Inverse QFT on counting qubits
    # Bit-reversal swaps first (mirrors the QFT swap step)
    for i in range(n_counting // 2):
        circ.swap(counting[i], counting[n_counting - 1 - i])

    # Inverse QFT rotations: work from most-significant to least-significant,
    # negate all CPhaseShift angles relative to forward QFT.
    for i in range(n_counting - 1, -1, -1):
        for j in range(n_counting - 1, i, -1):
            angle = -np.pi / 2 ** (j - i)
            _cphase(circ, counting[j], counting[i], angle)
        circ.h(counting[i])

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

    devices = COMPANY_DEVICES[company]
    print(f"Company   : {company}  ({len(devices)} device(s))")
    print(f"Unitary   : T gate  |  Phase target φ = {PHASE_TARGET} (= 1/8)")
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

        if max_qubits < 2:
            print(f"  Skipping — QPE requires at least 2 qubits.")
            continue

        # Submit half-qubit and full-qubit jobs
        for n_total in [max_qubits // 2, max_qubits]:
            if n_total < 2:
                continue
            n_counting = n_total - 1
            label = "half" if n_total == max_qubits // 2 else "full"
            print(f"\n  [{label}] Submitting QPE n_total={n_total} "
                  f"(counting={n_counting}, target=1) ...")

            circ = build_qpe(n_counting)

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
                print(f"    Qubits       : {n_total} total  |  Shots: {SHOTS}  |  Type: QPE")
                print(f"    Phase target : φ = {PHASE_TARGET}  (T gate, ideal outcome: "
                      f"{round(2**n_counting * PHASE_TARGET)})")

                record = {
                    "task_id":      task_id,
                    "submitted_at": submitted_at,
                    "device":       name,
                    "device_arn":   arn,
                    "circuit_type": "QPE",
                    "n_qubits":     n_total,
                    "n_counting":   n_counting,
                    "n_shots":      SHOTS,
                    "phase_target": PHASE_TARGET,
                }
                append_log(record)
                print(f"    Logged       → {LOG_FILE}")
                total_submitted += 1

            except Exception as e:
                print(f"  ✗ ERROR [{name}  n_total={n_total}  QPE] submission failed: {e}")

    print(f"\n{'='*60}")
    print(f"Total tasks submitted : {total_submitted}")
    print(f"Job log               : {LOG_FILE}")
    print(f"\nNote: Ideal counting-register outcome is "
          f"round(2^n_counting * {PHASE_TARGET}) = 2^(n_counting-3).")
    print(f"      Retrieve results and check peak-bin accuracy in checkRetieve.py.")


if __name__ == "__main__":
    main()