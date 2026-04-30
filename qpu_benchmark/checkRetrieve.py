#!/usr/bin/env python3
"""
checkRetrieve.py
===============
Run on the DCV where AWS credentials are configured.

Reads every job in job_results/job_log.txt, checks its AWS Braket status,
and for each COMPLETED job retrieves the measurement counts and writes them to:

    job_results/<task_uuid>.json

Timing captured per job:
  - submitted_at        : recorded at submission time (from job_log.txt)
  - completed_at        : from AWS task metadata (endedAt)
  - queue_time_seconds  : endedAt - createdAt  (AWS-side total task time)
  - wall_time_seconds   : endedAt - submitted_at  (total real-world wait)
  - qpu_time_seconds    : device-reported QPU execution time (if available)

GHZ verification (circuit_type == "GHZ"):
  ideal_fraction = fraction of shots that are |00...0⟩ or |11...1⟩
"""

import json
import os
import sys
from datetime import datetime, timezone

from braket.aws import AwsQuantumTask

JOB_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")


def task_uuid(task_id: str) -> str:
    """Extract the UUID from a Braket task ARN (part after last '/')."""
    return task_id.split("/")[-1]


def extract_qpu_time(result) -> float | None:
    """
    Try to extract device-reported QPU execution time from the result object.
    Returns seconds as float, or None if not available.

    IonQ reports executionDuration in additionalMetadata.ionqMetadata.
    Rigetti / IQM may use different fields.
    """
    addl = getattr(result, "additional_metadata", None)
    if addl is None:
        return None
    for attr in ("ionqMetadata", "rigettiMetadata", "iqmMetadata"):
        vendor_meta = getattr(addl, attr, None)
        if vendor_meta is not None:
            for field in ("executionDuration", "execution_duration",
                          "qpuExecutionDuration", "qpu_execution_duration"):
                val = getattr(vendor_meta, field, None)
                if val is not None:
                    return float(val)
    return None


def verify_ghz(counts: dict, n_qubits: int) -> dict:
    """
    Classical post-measurement GHZ verification.

    Ideal GHZ state produces only |00...0⟩ or |11...1⟩.
    Returns a dict with ideal_fraction, ideal_shots, total_shots.
    """
    ideal_states = {"0" * n_qubits, "1" * n_qubits}
    total_shots  = sum(counts.values())
    ideal_shots  = sum(cnt for outcome, cnt in counts.items() if outcome in ideal_states)
    return {
        "ideal_fraction": ideal_shots / total_shots if total_shots else 0.0,
        "ideal_shots":    ideal_shots,
        "total_shots":    total_shots,
    }


def main():
    if not os.path.exists(LOG_FILE):
        print(f"ERROR: {LOG_FILE} not found.")
        print("Run submit_QFT.py or submit_GHZ.py first.")
        sys.exit(1)

    with open(LOG_FILE) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("ERROR: job_log.txt is empty.")
        sys.exit(1)

    os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
    print(f"Found {len(records)} job(s) in {LOG_FILE}")
    print(f"Output directory : {JOB_RESULTS_DIR}\n")

    saved = skipped = pending = errors = failed = 0

    for i, rec in enumerate(records):
        task_id  = rec["task_id"]
        uuid     = task_uuid(task_id)
        out_path = os.path.join(JOB_RESULTS_DIR, f"{uuid}.json")

        # Skip silently if already retrieved (or already marked terminal-failed)
        if os.path.exists(out_path):
            skipped += 1
            try:
                with open(out_path) as fh:
                    prior_status = json.load(fh).get("status")
            except Exception:
                prior_status = None
            if prior_status in ("FAILED", "CANCELLED"):
                failed += 1
            else:
                saved += 1
            continue

        print(f"[{i+1}/{len(records)}] {uuid}")
        print(f"  Submitted : {rec['submitted_at']}")
        print(f"  Device    : {rec['device']}  |  "
              f"Circuit: {rec['circuit_type']}  |  Qubits: {rec['n_qubits']}")

        # Check status on AWS
        try:
            aws_task = AwsQuantumTask(arn=task_id)
            status   = aws_task.state()
        except Exception as e:
            print(f"  ERROR checking status: {e}\n")
            errors += 1
            continue

        print(f"  Status    : {status}")

        if status in ("FAILED", "CANCELLED"):
            marker = {
                "task_id":      task_id,
                "status":       status,
                "submitted_at": rec["submitted_at"],
                "device":       rec["device"],
                "device_arn":   rec["device_arn"],
                "circuit_type": rec["circuit_type"],
                "n_qubits":     rec["n_qubits"],
            }
            with open(out_path, "w") as f:
                json.dump(marker, f, indent=2)
            print(f"  Marked terminal {status} → {out_path}\n")
            failed += 1
            continue

        if status != "COMPLETED":
            print(f"  Not yet completed — skipping.\n")
            pending += 1
            continue

        # Retrieve measurement counts
        print("  Retrieving result ...")
        try:
            result = aws_task.result()
            counts = {str(k): int(v) for k, v in result.measurement_counts.items()}
        except Exception as e:
            print(f"  ERROR retrieving result: {e}\n")
            errors += 1
            continue

        n_shots_actual = sum(counts.values())

        # ── Timing ────────────────────────────────────────────────────────────
        completed_at       = None
        queue_time_seconds = None
        wall_time_seconds  = None
        qpu_time_seconds   = None

        try:
            meta       = aws_task.metadata()
            created_at = meta.get("createdAt")   # datetime object
            ended_at   = meta.get("endedAt")      # datetime object

            if ended_at:
                completed_at = ended_at.isoformat()

            if created_at and ended_at:
                queue_time_seconds = (ended_at - created_at).total_seconds()

            # Wall time: from our local submit timestamp to AWS completion
            submitted_dt = datetime.fromisoformat(rec["submitted_at"])
            if ended_at:
                # Make both timezone-aware for safe subtraction
                if submitted_dt.tzinfo is None:
                    submitted_dt = submitted_dt.replace(tzinfo=timezone.utc)
                wall_time_seconds = (ended_at - submitted_dt).total_seconds()

        except Exception as e:
            print(f"  WARNING: could not extract timing metadata: {e}")

        # Device-reported QPU execution time
        try:
            qpu_time_seconds = extract_qpu_time(result)
        except Exception:
            pass

        # ── GHZ verification ──────────────────────────────────────────────────
        ghz_verification = None
        if rec["circuit_type"] == "GHZ":
            ghz_verification = verify_ghz(counts, rec["n_qubits"])
            frac = ghz_verification["ideal_fraction"]
            ideal = ghz_verification["ideal_shots"]
            print(f"  GHZ verification : {ideal}/{n_shots_actual} ideal shots  "
                  f"(ideal fraction = {frac:.4f})")

        # ── Build and save result payload ─────────────────────────────────────
        payload = {
            "task_id":            task_id,
            "submitted_at":       rec["submitted_at"],
            "completed_at":       completed_at,
            "queue_time_seconds": queue_time_seconds,
            "wall_time_seconds":  wall_time_seconds,
            "qpu_time_seconds":   qpu_time_seconds,
            "device":             rec["device"],
            "device_arn":         rec["device_arn"],
            "circuit_type":       rec["circuit_type"],
            "n_qubits":           rec["n_qubits"],
            "n_shots_requested":  rec["n_shots"],
            "n_shots_actual":     n_shots_actual,
            "counts":             counts,
        }
        if ghz_verification is not None:
            payload["ghz_verification"] = ghz_verification

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Retrieved {n_shots_actual} shots, {len(counts)} unique outcomes.")
        if queue_time_seconds is not None:
            print(f"  Queue+exec time  : {queue_time_seconds:.1f} s")
        if wall_time_seconds is not None:
            print(f"  Wall time        : {wall_time_seconds:.1f} s")
        if qpu_time_seconds is not None:
            print(f"  QPU exec time    : {qpu_time_seconds:.4f} s")
        print(f"  Saved → {out_path}\n")
        saved += 1

    print(f"{'='*60}")
    print(f"Summary : {saved} saved  |  {pending} pending  |  "
          f"{failed} failed  |  {errors} errors")
    print(f"Results : {JOB_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
