#!/usr/bin/env python3
"""
checkRetrieve.py
================
Companion to submit_test.py and submit_qpuf_mitigation.py.

Reads every job listed in <results_dir>/job_log.txt, queries AWS for its
status, and for COMPLETED tasks writes a self-contained JSON to
<results_dir>/<uuid>.json. Idempotent -- already-saved jobs are skipped, so
re-run it as often as you like.

    python checkRetrieve.py                  # default: job_results/
    python checkRetrieve.py job_results_test # the pipeline smoke test
    python checkRetrieve.py --watch          # poll until everything lands

Only jobs recorded in job_log.txt are ever touched; the S3 task bucket is
never enumerated (it holds tasks from many unrelated runs).

Rigetti-specific value
----------------------
Rigetti returns `additional_metadata.rigettiMetadata.nativeQuilMetadata`,
which is the AUTHORITATIVE post-compilation cost of the circuit -- what
Quilc actually ran, after its own routing and optimisation:

    gateVolume            total native gates
    gateDepth             circuit depth
    multiQubitGateDepth   2-qubit depth  (the number that drives runtime)
    topologicalSwaps      SWAPs the router had to insert
    programFidelity       Rigetti's own fidelity estimate
    qpuRuntimeEstimation  Rigetti's own runtime estimate (ms)

This script prints those side by side with the qiskit-side prediction that
submit_*.py logged, so after the first job you can correct the gate-time and
fidelity constants in rigetti_qpuf_common.py with real numbers instead of
placeholders.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from braket.aws import AwsQuantumTask

HERE = os.path.dirname(os.path.abspath(__file__))


def read_job_log(path: str) -> list[dict]:
    """
    Parse job_log.txt into records. Normally one JSON object per line, but a
    record written without a trailing newline can glue two objects onto one
    line, which json.loads(line) rejects. Stream the file through a
    JSONDecoder instead so the newlines don't matter.
    """
    text = open(path).read()
    decoder = json.JSONDecoder()
    records, idx, n = [], 0, len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except ValueError as e:
            # A truncated final record (submit killed mid-write) must not cost
            # you the records before it -- those task ARNs may be the only
            # handle you have on tasks that are already running.
            line = text.count("\n", 0, idx) + 1
            print(f"WARNING: {path} is malformed at line {line} (byte {idx}): {e}")
            print(f"         Keeping the {len(records)} record(s) parsed before "
                  f"that point and ignoring the rest.")
            break
        records.append(obj)
        idx = end
    return records


def discover_logs(root: str) -> list[tuple[str, int]]:
    """
    Every job_results* directory under `root` that holds a job log, as
    (dirname, n_records). Used to turn "that log is empty" into "...but these
    other ones aren't", which is the actual question you are asking.
    """
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not name.startswith("job_results"):
            continue
        log = os.path.join(path, "job_log.txt")
        if os.path.exists(log):
            try:
                found.append((name, len(read_job_log(log))))
            except Exception:
                found.append((name, -1))
    return found


def print_log_hint(requested: str) -> None:
    """Tell the user which logs DO have records, and how to read them."""
    others = [(d, n) for d, n in discover_logs(HERE) if n > 0]
    if not others:
        print("\nNo job_results* directory here holds any records yet.")
        print("Submit something first (submit_test.py / submit_qpe.py / "
              "submit_qpuf_mitigation.py).")
        return
    print(f"\nThese logs DO have records (you asked for '{requested}'):")
    for d, n in others:
        print(f"    {d:<24} {n:>3} job(s)   ->  python checkRetrieve.py {d}")
    print("\nEach submit script writes its own directory:")
    print("    submit_test.py            -> job_results_test")
    print("    submit_qpe.py             -> job_results_qpe")
    print("    submit_qpuf_mitigation.py -> job_results   (the bare default)")


def task_uuid(job_id: str) -> str:
    """UUID slug after the last '/' of a Braket task ARN."""
    return job_id.split("/")[-1]


def serialize_additional_metadata(result) -> dict | None:
    """
    result.additional_metadata as a JSON-safe dict. Round-tripping through
    pydantic's JSON serializer (not .dict()) guarantees datetimes etc. come
    out as strings that json.dump will accept.
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


def extract_native_quil(addl: dict | None) -> dict | None:
    """Pull rigettiMetadata.nativeQuilMetadata out of the serialized blob."""
    if not isinstance(addl, dict):
        return None
    rig = addl.get("rigettiMetadata")
    if not isinstance(rig, dict):
        return None
    nqm = rig.get("nativeQuilMetadata")
    return nqm if isinstance(nqm, dict) else None


def extract_qpu_time(result, addl: dict | None) -> float | None:
    """
    Device-reported QPU execution time in seconds.

    Rigetti reports `qpuRuntimeEstimation` in MILLISECONDS inside
    nativeQuilMetadata; IonQ reports `executionDuration` in seconds on
    ionqMetadata. Probe both so this script also works on IonQ job logs.
    """
    nqm = extract_native_quil(addl)
    if nqm and isinstance(nqm.get("qpuRuntimeEstimation"), (int, float)):
        return float(nqm["qpuRuntimeEstimation"]) / 1000.0

    meta = getattr(result, "additional_metadata", None)
    if meta is None:
        return None
    for attr in ("ionqMetadata", "iqmMetadata", "rigettiMetadata"):
        vendor = getattr(meta, attr, None)
        if vendor is None:
            continue
        for field in ("executionDuration", "execution_duration",
                      "qpuExecutionDuration", "qpu_execution_duration"):
            val = getattr(vendor, field, None)
            if val is not None:
                return float(val)
    return None


def retrieve_counts(task: AwsQuantumTask):
    """
    Measurement counts, keyed by bitstring, plus the measured-qubit order.

    We read them straight off the Braket result rather than going through
    qiskit-braket-provider: our circuits use a SINGLE classical register, so
    there is no c1/c2 split to preserve, and the raw path has no dependency
    on the provider being able to re-resolve the backend.

    `measured_qubits` is saved alongside because Braket orders the bitstring
    by measured qubit index -- the analysis needs that mapping to split the
    string back into the stage-1 and stage-2 precision registers.
    """
    result = task.result()
    counts = {str(k): int(v) for k, v in result.measurement_counts.items()}
    measured = getattr(result, "measured_qubits", None)
    return result, counts, (list(measured) if measured is not None else None)


def process(records: list[dict], results_dir: str) -> tuple[int, int]:
    """Retrieve every COMPLETED job not already saved. Returns (saved, total)."""
    os.makedirs(results_dir, exist_ok=True)
    saved = 0

    for i, rec in enumerate(records):
        job_id   = rec["job_id"]
        uuid     = task_uuid(job_id)
        out_path = os.path.join(results_dir, f"{uuid}.json")

        print(f"[{i+1}/{len(records)}] {uuid}")
        print(f"  Submitted : {rec['datetime']}")
        print(f"  QPU       : {rec['qpu']}  |  n_prec={rec.get('n_prec')} "
              f"n_targ={rec.get('n_targ')} shots={rec.get('n_shots')} "
              f"role={rec.get('role', '?')} lambda={rec.get('zne_scale', '?')} "
              f"verbatim={rec.get('verbatim', '?')}")

        if os.path.exists(out_path):
            print("  Already saved -- skipping.\n")
            saved += 1
            continue

        try:
            task   = AwsQuantumTask(arn=job_id)
            status = task.state()
        except Exception as e:
            print(f"  ERROR checking status: {e}\n")
            continue

        print(f"  Status    : {status}")
        if status != "COMPLETED":
            # Under a reservation expect RUNNING -> COMPLETED almost at once.
            # On-demand you may sit in QUEUED for a long while.
            print("  Not COMPLETED -- skipping (re-run later).\n")
            continue

        print("  Retrieving counts ...")
        try:
            result, counts, measured_qubits = retrieve_counts(task)
        except Exception as e:
            print(f"  ERROR retrieving counts: {e}\n")
            continue

        n_shots_actual = sum(counts.values())
        print(f"  Retrieved {n_shots_actual} shots, {len(counts)} unique outcomes.")

        # -- Timing ------------------------------------------------------------
        completed_at = task_time_seconds = wall_time_seconds = None
        try:
            meta       = task.metadata()
            created_at = meta.get("createdAt")
            ended_at   = meta.get("endedAt")
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

        additional_metadata = serialize_additional_metadata(result)
        native_quil         = extract_native_quil(additional_metadata)
        qpu_time_seconds    = extract_qpu_time(result, additional_metadata)

        payload = dict(rec)                       # carry the submission record
        payload.update({
            "completed_at":        completed_at,
            "task_time_seconds":   task_time_seconds,
            "wall_time_seconds":   wall_time_seconds,
            "qpu_time_seconds":    qpu_time_seconds,
            "n_shots":             n_shots_actual,
            "n_shots_requested":   rec.get("n_shots"),
            "measured_qubits":     measured_qubits,
            "counts":              counts,
            "native_quil_metadata": native_quil,
            "additional_metadata": additional_metadata,
        })
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        if task_time_seconds is not None:
            print(f"  Task time     : {task_time_seconds:.2f} s")
        if wall_time_seconds is not None:
            print(f"  Wall time     : {wall_time_seconds:.2f} s")
        if qpu_time_seconds is not None:
            print(f"  QPU exec time : {qpu_time_seconds:.4f} s")

        # -- Predicted vs actual: use this to re-fit the model constants ------
        if native_quil:
            print("  --- Rigetti native-Quil metadata (authoritative) ---")
            pred_2q    = rec.get("n_2q_gates")
            pred_gates = rec.get("n_gates")
            pred_d2q   = rec.get("depth_2q")
            for key, pred in (("gateVolume", pred_gates),
                              ("gateDepth", rec.get("depth")),
                              ("multiQubitGateDepth", pred_d2q),
                              ("topologicalSwaps", None),
                              ("programFidelity", rec.get("est_fidelity")),
                              ("qpuRuntimeEstimation", None)):
                val = native_quil.get(key)
                if val is None:
                    continue
                if isinstance(pred, (int, float)) and pred:
                    print(f"    {key:<22}: {val}   (qiskit predicted {pred})")
                else:
                    print(f"    {key:<22}: {val}")
            if pred_2q:
                print(f"    (qiskit predicted 2q gate count: {pred_2q})")
            est = rec.get("est_runtime_parallel_s")
            if est and task_time_seconds:
                print(f"    est. runtime {est:.2f} s  vs  actual task time "
                      f"{task_time_seconds:.2f} s  "
                      f"(ratio {task_time_seconds/est:.2f}x)")

        print(f"  Saved -> {out_path}\n")
        saved += 1

    return saved, len(records)


def main():
    args    = [a for a in sys.argv[1:] if not a.startswith("--")]
    watch   = "--watch" in sys.argv
    sub_dir = args[0] if args else "job_results"
    results_dir = sub_dir if os.path.isabs(sub_dir) else os.path.join(HERE, sub_dir)
    log_file    = os.path.join(results_dir, "job_log.txt")

    if not os.path.exists(log_file):
        print(f"ERROR: {log_file} not found.")
        print_log_hint(sub_dir)
        sys.exit(1)

    while True:
        records = read_job_log(log_file)
        if not records:
            # Name the PATH. Without it this reads as "the log you are looking
            # at is empty", when what actually happened is almost always that
            # the default directory was read instead of the one just written.
            print(f"ERROR: {log_file} contains no records "
                  f"({os.path.getsize(log_file)} bytes).")
            print_log_hint(sub_dir)
            sys.exit(1)

        print(f"Found {len(records)} job(s) in {log_file}")
        print(f"Output directory: {results_dir}\n")
        saved, total = process(records, results_dir)
        print(f"Done. {saved}/{total} job result(s) available in {results_dir}/")

        if not watch or saved == total:
            break
        print("\n--watch: not all jobs done; re-checking in 30 s ...\n")
        time.sleep(30)


if __name__ == "__main__":
    main()
