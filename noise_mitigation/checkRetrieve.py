#!/usr/bin/env python3
"""
checkRetrieve.py
================
Companion to ionq_noise_mitig.py.

Reads every job listed in job_results/job_log.txt and, for each one, queries
AWS for its task status. For COMPLETED tasks it retrieves the measurement
counts and writes a self-contained JSON to job_results/<uuid>.json — exactly
the files ionq_noise_mitigation_compare.ipynb auto-discovers and classifies
into "debias" vs "no-mitig" runs.

This script only ever looks at jobs recorded in job_log.txt — it never
enumerates the S3 task bucket. That matters because the bucket typically
holds tasks from many unrelated runs; we only care about the ones this
project submitted.

These jobs are submitted ON-DEMAND (no Braket Direct reservation), so a task
may sit in QUEUED → RUNNING before reaching COMPLETED. Re-run this script
until every job has been saved; it is idempotent (already-saved jobs are
skipped).

The circuit is a single-stage QPE QPUF with ONE classical register (c[n_prec]),
so counts come back as plain n_prec-bit strings.

Timing captured per job:
  - datetime           : submission timestamp (from job_log.txt)
  - completed_at       : AWS task endedAt
  - task_time_seconds  : endedAt − createdAt (AWS-side task lifetime; on-demand
                         this includes any queue wait)
  - wall_time_seconds  : endedAt − datetime  (total real-world wait, including
                         our local submit round trip)
  - qpu_time_seconds   : device-reported QPU execution time, from
                         result.additional_metadata.ionqMetadata
                         (None if the device doesn't report it)
"""

import json
import os
import sys
from datetime import datetime, timezone

from braket.aws import AwsQuantumTask

JOB_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")


def task_uuid(job_id: str) -> str:
    """Extract the UUID from a Braket task ARN (the slug after the last '/')."""
    return job_id.split("/")[-1]


def extract_qpu_time(result) -> float | None:
    """
    Device-reported QPU execution time, in seconds.

    IonQ exposes this as additional_metadata.ionqMetadata.executionDuration
    (units: seconds). Other vendors use analogous fields; we probe the
    common ones and return the first hit, or None if nothing is reported.
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


def serialize_additional_metadata(result) -> dict | None:
    """
    Full result.additional_metadata blob as a JSON-friendly dict.

    This is where IonQ reports its server-side compilation results — native
    gate counts, the compiled program, error-mitigation settings, and (for a
    debias run) the aggregated `sharpenedProbabilities`. Round-tripping
    through pydantic's JSON serializer (rather than .dict()/.model_dump())
    guarantees the output is safe to json.dump later (datetimes etc. come out
    as ISO strings).
    """
    addl = getattr(result, "additional_metadata", None)
    if addl is None:
        return None
    for method in ("model_dump_json", "json"):
        if hasattr(addl, method):
            try:
                return json.loads(getattr(addl, method)())
            except Exception:
                continue
    return None


def retrieve_counts(job_id: str, device_arn: str) -> dict | None:
    """
    Pull qiskit-format measurement counts via the qiskit-braket-provider.

    For a debias job these are IonQ's debiased (aggregated) counts; for a
    no-mitig job they are the raw counts. Either way they come back as plain
    n_prec-bit strings keyed by the single classical register.
    """
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
              f"n_targ={rec.get('n_targ', '?')}  n_shots={rec['n_shots']}  "
              f"mitigation={rec.get('error_mitigation', '?')}")

        # Skip if we've already retrieved this one — idempotent.
        if os.path.exists(out_path):
            print("  Already saved — skipping.\n")
            saved += 1
            continue

        # Query AWS for the current task status.
        try:
            aws_task = AwsQuantumTask(arn=job_id)
            status   = aws_task.state()
        except Exception as e:
            print(f"  ERROR checking status: {e}\n")
            continue

        print(f"  Status    : {status}")

        if status != "COMPLETED":
            # On-demand: expect QUEUED → RUNNING → COMPLETED. Other states
            # (FAILED, CANCELLED) are surfaced but not retrieved.
            print(f"  Not COMPLETED — skipping (re-run later).\n")
            continue

        print("  Retrieving counts ...")
        counts = retrieve_counts(job_id, rec["device_arn"])
        if counts is None:
            print()
            continue

        n_shots_actual = sum(counts.values())
        print(f"  Retrieved {n_shots_actual} shots, {len(counts)} unique outcomes.")

        # ── Timing ───────────────────────────────────────────────────────────
        completed_at      = None
        task_time_seconds = None
        wall_time_seconds = None
        qpu_time_seconds  = None

        try:
            meta       = aws_task.metadata()
            created_at = meta.get("createdAt")   # tz-aware datetime
            ended_at   = meta.get("endedAt")      # tz-aware datetime

            if ended_at:
                completed_at = ended_at.isoformat()

            if created_at and ended_at:
                task_time_seconds = (ended_at - created_at).total_seconds()

            submitted_dt = datetime.fromisoformat(rec["datetime"])
            if submitted_dt.tzinfo is None:
                submitted_dt = submitted_dt.replace(tzinfo=timezone.utc)
            if ended_at:
                wall_time_seconds = (ended_at - submitted_dt).total_seconds()

        except Exception as e:
            print(f"  WARNING: could not extract timing metadata: {e}")

        # Device-reported QPU execution time AND the full IonQ-side metadata
        # blob (native gate counts, compiled program, error mitigation,
        # sharpenedProbabilities, ...). Second result() call is cheap — the
        # result JSON is already in S3 from the qiskit retrieval above.
        additional_metadata = None
        try:
            braket_result       = aws_task.result()
            qpu_time_seconds    = extract_qpu_time(braket_result)
            additional_metadata = serialize_additional_metadata(braket_result)
        except Exception as e:
            print(f"  WARNING: could not extract additional_metadata: {e}")

        # Self-contained output: everything the comparison notebook needs to
        # interpret the run, including the Haar unitary and the mitigation
        # mode (used to classify debias vs no-mitig).
        payload = {
            "job_id":            job_id,
            "datetime":          rec["datetime"],
            "completed_at":      completed_at,
            "task_time_seconds": task_time_seconds,
            "wall_time_seconds": wall_time_seconds,
            "qpu_time_seconds":  qpu_time_seconds,
            "qpu":               rec["qpu"],
            "device_arn":        rec["device_arn"],
            "circuit_type":      rec.get("circuit_type"),
            "error_mitigation":  rec.get("error_mitigation"),
            "n_prec":            rec["n_prec"],
            "n_targ":            rec.get("n_targ"),
            "n_shots":           n_shots_actual,
            "n_shots_requested": rec.get("n_shots_requested", rec["n_shots"]),
            "n_gates":           rec.get("n_gates"),
            "n_1q_gates":        rec.get("n_1q_gates"),
            "n_2q_gates":        rec.get("n_2q_gates"),
            "gates_per_circuit": rec.get("gates_per_circuit"),
            "total_gates":       rec.get("total_gates"),
            "est_fidelity":      rec.get("est_fidelity"),
            "seed":              rec.get("seed"),
            "target_init_seed":  rec.get("target_init_seed"),
            "unitary":           rec.get("unitary"),
            "counts":            counts,
            "additional_metadata": additional_metadata,
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        if task_time_seconds is not None:
            print(f"  Task time     : {task_time_seconds:.2f} s  (incl. any queue wait)")
        if wall_time_seconds is not None:
            print(f"  Wall time     : {wall_time_seconds:.2f} s")
        if qpu_time_seconds is not None:
            print(f"  QPU exec time : {qpu_time_seconds:.4f} s")
        if isinstance(additional_metadata, dict):
            ionq = additional_metadata.get("ionqMetadata")
            if isinstance(ionq, dict) and ionq:
                print(f"  IonQ metadata : {sorted(ionq.keys())}")
        print(f"  Saved → {out_path}\n")
        saved += 1

    print(f"Done. {saved}/{len(records)} job result(s) available in {JOB_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
