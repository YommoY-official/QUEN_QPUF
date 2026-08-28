#!/usr/bin/env python3
"""
cancelQueue.py
==============
Companion to submit_ionq_mitigation.py / checkRetrieve.py.

Lists this campaign's tasks that are still QUEUED on the device and lets you
pick, interactively in the terminal, which ones to cancel. Only QUEUED tasks
are shown, because that is the only state from which a Braket task can still
be cancelled (once it is RUNNING/COMPLETED it cannot be pulled back).

Adapted from braket_direct/ionq_20260601/cancelEditQueue.py with the
reservation lines stripped: this campaign submits ON-DEMAND (no RES_ARN, no
DirectReservation), so there is no reservation context to print or operate
within -- on-demand tasks queue behind other users' work on the shared
device instead.

Scope: like checkRetrieve.py, this script only ever looks at tasks recorded
in job_results/job_log.txt -- it never enumerates the S3 bucket or the whole
account. That keeps it limited to the jobs THIS campaign submitted.

Nothing is cancelled until you make a selection AND confirm with 'y'.
"""

import json
import os
import sys

from braket.aws import AwsQuantumTask

from ionq_mitigation_common import DEVICE_NAME, DEVICE_ARN

JOB_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")


def task_uuid(job_id: str) -> str:
    """Extract the UUID from a Braket task ARN (the slug after the last '/')."""
    return job_id.split("/")[-1]


def queue_position(aws_task: AwsQuantumTask) -> str:
    """Best-effort queue position string, or '?' if the SDK can't report it."""
    try:
        info = aws_task.queue_position()
        pos = getattr(info, "queue_position", None)
        return str(pos) if pos not in (None, "") else "?"
    except Exception:
        return "?"


def collect_queued():
    """
    Return a list of dicts (one per QUEUED task found in job_log.txt):
        {"job_id", "uuid", "rec", "task", "queue_pos"}
    """
    if not os.path.exists(LOG_FILE):
        print(f"ERROR: {LOG_FILE} not found. Submit a job first.")
        sys.exit(1)

    with open(LOG_FILE) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("ERROR: job_log.txt is empty.")
        sys.exit(1)

    print(f"Scanning {len(records)} logged job(s) for QUEUED tasks ...\n")

    queued = []
    for rec in records:
        job_id = rec["job_id"]
        uuid   = task_uuid(job_id)
        try:
            aws_task = AwsQuantumTask(arn=job_id)
            status   = aws_task.state()
        except Exception as e:
            print(f"  {uuid}: ERROR checking status: {e}")
            continue

        if status != "QUEUED":
            continue

        queued.append({
            "job_id":    job_id,
            "uuid":      uuid,
            "rec":       rec,
            "task":      aws_task,
            "queue_pos": queue_position(aws_task),
        })

    return queued


def print_table(queued):
    """Print the numbered list of QUEUED tasks the user can choose from."""
    print("=" * 78)
    print(f"QUEUED tasks on {DEVICE_NAME}  ({DEVICE_ARN})")
    print("Submission mode: ON-DEMAND (no reservation)")
    print("=" * 78)
    print(f"{'#':>3}  {'task uuid':36s}  {'qpos':>4}  {'submitted':25s}  shape")
    print("-" * 78)
    for i, q in enumerate(queued):
        rec = q["rec"]
        shape = (f"prec={rec.get('n_prec', '?')} "
                 f"targ={rec.get('n_targ', '?')} "
                 f"shots={rec.get('n_shots', '?')} "
                 f"mit={rec.get('error_mitigation', '?')} "
                 f"lambda={rec.get('zne_scale', '?')} "
                 f"ddd={rec.get('ddd_sequence', '?')}")
        print(f"{i:>3}  {q['uuid']:36s}  {q['queue_pos']:>4}  "
              f"{rec.get('datetime', '?'):25s}  {shape}")
    print("-" * 78)


def parse_selection(raw: str, n: int):
    """
    Turn a selection string into a sorted list of unique valid indices.

    Accepts: 'all', comma/space separated indices ('0,2 3'), or empty/'q'
    to mean "no selection". Returns [] for abort. Raises ValueError on a
    malformed or out-of-range entry so the caller can re-prompt.
    """
    raw = raw.strip().lower()
    if raw in ("", "q", "quit", "n", "no"):
        return []
    if raw == "all":
        return list(range(n))

    idxs = set()
    for tok in raw.replace(",", " ").split():
        idx = int(tok)              # ValueError -> caller re-prompts
        if not (0 <= idx < n):
            raise ValueError(f"index {idx} out of range 0..{n - 1}")
        idxs.add(idx)
    return sorted(idxs)


def main():
    queued = collect_queued()

    if not queued:
        print("No QUEUED tasks found -- nothing to cancel.")
        return

    print_table(queued)

    while True:
        try:
            raw = input("\nEnter task numbers to CANCEL "
                        "(comma-separated, 'all', or blank to abort): ")
        except EOFError:
            raw = ""
        try:
            selection = parse_selection(raw, len(queued))
            break
        except ValueError as e:
            print(f"  Invalid selection: {e}")

    if not selection:
        print("No tasks selected -- nothing cancelled.")
        return

    print("\nThe following tasks will be CANCELLED:")
    for idx in selection:
        print(f"  [{idx}] {queued[idx]['uuid']}")
    try:
        resp = input(f"\nCancel these {len(selection)} task(s)? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted -- nothing cancelled.")
        return

    cancelled = 0
    for idx in selection:
        q = queued[idx]
        try:
            q["task"].cancel()
            print(f"  [{idx}] {q['uuid']}: cancellation requested.")
            cancelled += 1
        except Exception as e:
            # Most likely the task already left QUEUED between scan and cancel.
            print(f"  [{idx}] {q['uuid']}: could NOT cancel -- {e}")

    print(f"\nDone. Requested cancellation for {cancelled}/{len(selection)} task(s).")


if __name__ == "__main__":
    main()
