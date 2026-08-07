#!/usr/bin/env python3
"""
cancel_tasks.py
===============
Cancel Braket quantum tasks. Three ways to find them, because during a
reservation you will not always have the nice one available.

    python cancel_tasks.py --log job_results_qpe     # ARNs from a job log
    python cancel_tasks.py --tag                     # everything tagged for this run
    python cancel_tasks.py --queued                  # everything QUEUED/RUNNING on Cepheus

    ... add --yes to actually cancel. Without it this only PRINTS what it
    would do. Cancelling is irreversible and a cancelled task returns no
    results, so the safe mode is the default.

Which mode to use
-----------------
  --log     Most precise: reads the task ARNs the submit scripts recorded.
            Use this when the run went out normally and you know which
            workload you want to kill.

  --tag     Uses the ReservationRun tag that task_tags() puts on every task.
            Use this when tasks were submitted across several scripts or
            several sessions and you want the whole reservation's worth.
            Braket has no "list everything for reservation X" API -- the tag
            is the only handle, which is why it gets attached at submit time.

  --queued  Broadest and bluntest: every task on the device still in a
            cancellable state, tagged or not. Use this when something has
            gone wrong, you are not sure what is out there, and the window
            is burning. It will also catch tasks from unrelated work in the
            same account/region, so read the list before passing --yes.

Note on what is cancellable
---------------------------
CREATED / QUEUED / RUNNING can be cancelled. COMPLETED / FAILED / CANCELLED
are terminal and are skipped. A RUNNING task on a QPU may already have
consumed device time; cancelling stops it but does not refund it.

Reservation guidance (Braket Direct best practices): if the workload looks
wrong, do NOT cancel reflexively. There is no extra charge for letting tasks
sit while you diagnose, and a cancelled task returns nothing. Understand the
problem first, then cancel.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigetti_qpuf_common import AWS_REGION, DEVICE_ARN, DEVICE_NAME, RUN_TAG

HERE = os.path.dirname(os.path.abspath(__file__))

# States a task can still be cancelled out of.
CANCELLABLE = {"CREATED", "QUEUED", "RUNNING"}


def braket_client():
    import boto3
    return boto3.client("braket", region_name=AWS_REGION)


# -- Finding tasks -------------------------------------------------------------

def arns_from_log(sub_dir: str) -> list[str]:
    """Task ARNs recorded by the submit scripts in <sub_dir>/job_log.txt."""
    from checkRetrieve import read_job_log

    results_dir = sub_dir if os.path.isabs(sub_dir) else os.path.join(HERE, sub_dir)
    log_file = os.path.join(results_dir, "job_log.txt")
    if not os.path.exists(log_file):
        print(f"ERROR: {log_file} not found.")
        sys.exit(1)
    arns = [r["job_id"] for r in read_job_log(log_file) if r.get("job_id")]
    print(f"Read {len(arns)} task ARN(s) from {log_file}")
    return arns


def arns_from_tag(tag_key: str, tag_value: str) -> list[str]:
    """
    Task ARNs carrying a given resource tag, via resourcegroupstaggingapi.

    Paginated deliberately: get_resources caps each page, and a ZNE sweep plus
    readout calibration across a few sessions passes that cap quickly.
    """
    import boto3

    client = boto3.client("resourcegroupstaggingapi", region_name=AWS_REGION)

    def query(type_filter: bool) -> list[str]:
        found, token = [], None
        while True:
            kwargs = {"TagFilters": [{"Key": tag_key, "Values": [tag_value]}]}
            if type_filter:
                kwargs["ResourceTypeFilters"] = ["braket:quantum-task"]
            if token:
                kwargs["PaginationToken"] = token
            resp = client.get_resources(**kwargs)
            found += [m["ResourceARN"] for m in resp.get("ResourceTagMappingList", [])]
            token = resp.get("PaginationToken") or ""
            if not token:
                return found

    # Ask for quantum tasks specifically, but do NOT trust an empty answer: if
    # the resource-type string is ever wrong, a typed query returns [] and that
    # is indistinguishable from "nothing to cancel" -- the worst possible
    # ambiguity in an emergency. Fall back to an untyped query and filter here.
    arns = query(type_filter=True)
    if not arns:
        arns = [a for a in query(type_filter=False) if ":quantum-task/" in a]
        if arns:
            print("  (typed tag query returned nothing; used untyped fallback)")

    print(f"Found {len(arns)} task(s) tagged {tag_key}={tag_value} in {AWS_REGION}")
    return arns


def arns_queued(device_arn: str | None) -> list[str]:
    """Every task in a cancellable state, optionally restricted to one device."""
    client = braket_client()
    arns, token = [], None
    for state in sorted(CANCELLABLE):
        token = None
        while True:
            filters = [{"name": "status", "operator": "EQUAL", "values": [state]}]
            if device_arn:
                filters.append({"name": "deviceArn", "operator": "EQUAL",
                                "values": [device_arn]})
            kwargs = {"filters": filters, "maxResults": 100}
            if token:
                kwargs["nextToken"] = token
            resp = client.search_quantum_tasks(**kwargs)
            arns += [t["quantumTaskArn"] for t in resp.get("quantumTasks", [])]
            token = resp.get("nextToken")
            if not token:
                break
    where = DEVICE_NAME if device_arn else f"all devices in {AWS_REGION}"
    print(f"Found {len(arns)} cancellable task(s) on {where}")
    return arns


# -- Cancelling ----------------------------------------------------------------

def describe(arns: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    ([(arn, status)], [(arn, error)]) -- readable tasks and unreadable ones.

    Unreadable tasks are returned rather than dropped. Silently skipping them
    would let a credentials or region failure print as "nothing to cancel",
    which is the exact opposite of the truth and the worst thing this script
    could tell you while a reservation is burning.
    """
    client = braket_client()
    ok, bad = [], []
    for arn in arns:
        try:
            ok.append((arn, client.get_quantum_task(quantumTaskArn=arn)["status"]))
        except Exception as e:
            bad.append((arn, str(e)))
    return ok, bad


def cancel(arns: list[str], really: bool) -> None:
    if not arns:
        print("Nothing to do.")
        return

    unique = sorted(set(arns))
    print(f"\nChecking status of {len(unique)} task(s) ...")
    tasks, unreadable = describe(unique)

    live = [(a, s) for a, s in tasks if s in CANCELLABLE]
    done = [(a, s) for a, s in tasks if s not in CANCELLABLE]

    print("\n" + "=" * 78)
    print(f"{len(live)} cancellable, {len(done)} already terminal, "
          f"{len(unreadable)} UNREADABLE")
    print("=" * 78)
    for arn, status in live:
        print(f"  {status:<10} {arn.split('/')[-1]}")
    if done:
        from collections import Counter
        tally = ", ".join(f"{n} {s}" for s, n in Counter(s for _, s in done).items())
        print(f"  (skipping: {tally})")

    if unreadable:
        print(f"\n*** Could not read the status of {len(unreadable)} task(s). "
              f"Their state is UNKNOWN -- they may still be queued or running.")
        for arn, err in unreadable[:5]:
            print(f"    {arn.split('/')[-1]}: {err}")
        if len(unreadable) > 5:
            print(f"    ... and {len(unreadable) - 5} more")
        print("    Check your AWS credentials and that the region is "
              f"{AWS_REGION}, then re-run. Do NOT read this as 'nothing to "
              "cancel'.")

    if not live:
        if unreadable:
            print("\nNothing was cancelled, and the tasks above were never "
                  "checked. Fix the error and re-run before concluding "
                  "anything.")
            sys.exit(2)
        print("\nNo task is in a cancellable state.")
        return

    if not really:
        print(f"\nDRY RUN -- nothing was cancelled.")
        print(f"Re-run with --yes to cancel these {len(live)} task(s).")
        return

    print(f"\nCancelling {len(live)} task(s) ...")
    client = braket_client()
    ok = 0
    for arn, _ in live:
        try:
            resp = client.cancel_quantum_task(quantumTaskArn=arn)
            print(f"  {resp.get('cancellationStatus', 'REQUESTED'):<10} "
                  f"{arn.split('/')[-1]}")
            ok += 1
        except Exception as e:
            # Losing a race with a task that just completed is normal here.
            print(f"  FAILED     {arn.split('/')[-1]}: {e}")
    print(f"\n{ok}/{len(live)} cancellation(s) accepted.")


def main():
    p = argparse.ArgumentParser(
        description="Cancel Braket quantum tasks (dry run unless --yes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python cancel_tasks.py --log job_results_qpe\n"
               "  python cancel_tasks.py --tag --yes\n"
               "  python cancel_tasks.py --queued\n")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", metavar="DIR",
                     help="cancel task ARNs recorded in DIR/job_log.txt")
    src.add_argument("--tag", action="store_true",
                     help=f"cancel tasks tagged ReservationRun={RUN_TAG}")
    src.add_argument("--queued", action="store_true",
                     help="cancel every CREATED/QUEUED/RUNNING task on the device")
    src.add_argument("--arn", metavar="ARN", nargs="+",
                     help="cancel these task ARNs explicitly")

    p.add_argument("--tag-key", default="ReservationRun", help="tag key for --tag")
    p.add_argument("--tag-value", default=RUN_TAG, help="tag value for --tag")
    p.add_argument("--all-devices", action="store_true",
                   help="with --queued, do not restrict to Cepheus")
    p.add_argument("--yes", action="store_true",
                   help="actually cancel (default is a dry run)")
    args = p.parse_args()

    print(f"Region : {AWS_REGION}")
    if args.log:
        arns = arns_from_log(args.log)
    elif args.tag:
        arns = arns_from_tag(args.tag_key, args.tag_value)
    elif args.queued:
        arns = arns_queued(None if args.all_devices else DEVICE_ARN)
    else:
        arns = args.arn

    cancel(arns, really=args.yes)


if __name__ == "__main__":
    main()
