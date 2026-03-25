#!/usr/bin/env python3
"""
retrieve_jobs.py
================
Run this on the DCV (where AWS credentials are available).

Reads every job in jobs_log.jsonl, checks its AWS status, and – for each
COMPLETED job – retrieves the measurement counts and writes them to:

    jobs_results/<task-uuid>.txt

Each file is a self-contained JSON document with the job metadata AND the
counts dict, so the companion analysis notebook needs nothing else.
"""

import json
import os
import sys

from braket.aws import AwsQuantumTask

LOG_FILE    = os.path.join(os.path.dirname(__file__), "jobs_log.jsonl")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "jobs_results")


def task_uuid(job_id: str) -> str:
    """Extract the UUID from a Braket ARN (part after the last '/')."""
    return job_id.split("/")[-1]


def retrieve_counts(job_id: str, device_name: str) -> dict | None:
    """Retrieve Qiskit-format counts via qiskit_braket_provider."""
    try:
        from qiskit_braket_provider import BraketProvider
    except ImportError:
        print("  ERROR: qiskit-braket-provider not installed "
              "(pip install qiskit-braket-provider)")
        return None

    try:
        provider = BraketProvider()
        backend  = provider.get_backend(device_name)
        job_hw   = backend.retrieve_job(job_id)
        return job_hw.result().get_counts()
    except Exception as e:
        print(f"  ERROR retrieving counts: {e}")
        return None


def main():
    # ── Load log ──────────────────────────────────────────────────────────
    if not os.path.exists(LOG_FILE):
        print(f"ERROR: {LOG_FILE} not found. Run submit_1qubit_qpu.py first.")
        sys.exit(1)

    with open(LOG_FILE) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("ERROR: jobs_log.jsonl is empty.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Found {len(records)} job(s) in {LOG_FILE}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    saved = 0

    for i, rec in enumerate(records):
        job_id   = rec["job_id"]
        uuid     = task_uuid(job_id)
        out_path = os.path.join(OUTPUT_DIR, f"{uuid}.txt")

        print(f"[{i+1}/{len(records)}] {uuid}")
        print(f"  Submitted : {rec['datetime']}")
        print(f"  Device    : {rec['device']}  |  n_prec={rec['n_prec']}  "
              f"n_shots={rec['n_shots']}")

        # Skip if already retrieved
        if os.path.exists(out_path):
            print("  Already saved – skipping.\n")
            saved += 1
            continue

        # Check status
        try:
            status = AwsQuantumTask(arn=job_id).state()
        except Exception as e:
            print(f"  ERROR checking status: {e}\n")
            continue

        print(f"  Status    : {status}")

        if status != "COMPLETED":
            print(f"  Not yet completed – skipping.\n")
            continue

        # Retrieve counts
        print("  Retrieving counts …")
        counts = retrieve_counts(job_id, rec["device"])
        if counts is None:
            print()
            continue

        n_shots_actual = sum(counts.values())
        print(f"  Retrieved {n_shots_actual} shots, {len(counts)} unique outcomes.")

        # Write result file
        payload = {
            "job_id":    job_id,
            "datetime":  rec["datetime"],
            "device":    rec["device"],
            "n_prec":    rec["n_prec"],
            "n_shots":   n_shots_actual,
            "seed":      rec.get("seed"),
            "angles":    rec["angles"],
            "counts":    counts,
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Saved → {out_path}\n")
        saved += 1

    print(f"Done. {saved}/{len(records)} job result(s) available in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()