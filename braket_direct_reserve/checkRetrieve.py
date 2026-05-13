#!/usr/bin/env python3
"""
checkRetrieve.py
================
Companion to submit_QPUF_ntarg.py / submit_test.py.

Reads every job listed in job_results/job_log.txt and, for each one, queries
AWS for its task status. For COMPLETED tasks it retrieves the measurement
counts and writes a self-contained JSON to job_results/<uuid>.json.

This script only ever looks at jobs that are recorded in job_log.txt — it
never enumerates the S3 task bucket. That matters because the bucket
typically holds tasks from many unrelated runs; we only care about the
ones this project submitted.

Under a Braket Direct reservation, tasks generally sit in "RUNNING" briefly
and then move to "COMPLETED" — so a single pass at the end of the
reservation window is usually sufficient.
"""

import json
import os
import sys

from braket.aws import AwsQuantumTask

JOB_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")


def task_uuid(job_id: str) -> str:
    """Extract the UUID from a Braket task ARN (the slug after the last '/')."""
    return job_id.split("/")[-1]


def retrieve_counts(job_id: str, device_arn: str) -> dict | None:
    """Pull qiskit-format measurement counts for a completed task."""
    try:
        from qiskit_braket_provider import BraketProvider
    except ImportError:
        print("  ERROR: qiskit-braket-provider not installed "
              "(pip install qiskit-braket-provider)")
        return None

    try:
        provider = BraketProvider()
        backend  = None
        for b in provider.backends():
            dev = getattr(b, "_device", None)
            if dev and getattr(dev, "arn", None) == device_arn:
                backend = b
                break
        if backend is None:
            print(f"  ERROR: no backend found for ARN {device_arn}")
            return None
        job_hw = backend.retrieve_job(job_id)
        return job_hw.result().get_counts()
    except Exception as e:
        print(f"  ERROR retrieving counts: {e}")
        return None


def main():
    if not os.path.exists(LOG_FILE):
        print(f"ERROR: {LOG_FILE} not found. Submit a job first.")
        sys.exit(1)

    with open(LOG_FILE) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("ERROR: job_log.txt is empty.")
        sys.exit(1)

    os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
    print(f"Found {len(records)} job(s) in {LOG_FILE}")
    print(f"Output directory: {JOB_RESULTS_DIR}\n")

    saved = 0

    for i, rec in enumerate(records):
        job_id   = rec["job_id"]
        uuid     = task_uuid(job_id)
        out_path = os.path.join(JOB_RESULTS_DIR, f"{uuid}.json")

        print(f"[{i+1}/{len(records)}] {uuid}")
        print(f"  Submitted : {rec['datetime']}")
        print(f"  QPU       : {rec['qpu']}  |  n_prec={rec['n_prec']}  "
              f"n_targ={rec.get('n_targ', '?')}  n_shots={rec['n_shots']}")

        # Skip if we've already retrieved this one — idempotent.
        if os.path.exists(out_path):
            print("  Already saved — skipping.\n")
            saved += 1
            continue

        # Query AWS for the current task status.
        try:
            status = AwsQuantumTask(arn=job_id).state()
        except Exception as e:
            print(f"  ERROR checking status: {e}\n")
            continue

        print(f"  Status    : {status}")

        if status != "COMPLETED":
            # With a reservation, expect RUNNING → COMPLETED. Other states
            # (FAILED, CANCELLED) are surfaced but not retrieved.
            print(f"  Not COMPLETED — skipping.\n")
            continue

        print("  Retrieving counts ...")
        counts = retrieve_counts(job_id, rec["device_arn"])
        if counts is None:
            print()
            continue

        n_shots_actual = sum(counts.values())
        print(f"  Retrieved {n_shots_actual} shots, {len(counts)} unique outcomes.")

        # Self-contained output: everything the analysis notebook needs to
        # interpret the run, including the Haar unitary used.
        payload = {
            "job_id":           job_id,
            "datetime":         rec["datetime"],
            "qpu":              rec["qpu"],
            "device_arn":       rec["device_arn"],
            "circuit_type":     rec.get("circuit_type"),
            "n_prec":           rec["n_prec"],
            "n_targ":           rec.get("n_targ"),
            "n_shots":          n_shots_actual,
            "n_shots_requested": rec.get("n_shots_requested", rec["n_shots"]),
            "n_gates":          rec.get("n_gates"),
            "seed":             rec.get("seed"),
            "target_init_seed": rec.get("target_init_seed"),
            "unitary":          rec.get("unitary"),
            "counts":           counts,
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Saved → {out_path}\n")
        saved += 1

    print(f"Done. {saved}/{len(records)} job result(s) available in {JOB_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
