#!/usr/bin/env python3
"""
probe_gate_limit.py
====================
Empirically probes the real gate-count / compile limit on IonQ
Forte-Enterprise-1, using the genuine-MCM two-stage QPE-QPUF circuit
(`build_full_circuit` in submit_QPUF_ntarg.py -- the endianness-fixed
builder with a real mid-circuit measurement between stages, NOT the
no-MCM two-stage circuit in noise_mitigation/).

Why this circuit, not a synthetic gate-count filler
----------------------------------------------------
AWS's documentation reportedly states a 5000-gate compile limit for
Forte-Enterprise-1. CLAUDE.md's ground-truth principle says trust the
device's own response over a doc claim, not the other way round, so this
script submits increasingly large REAL QPUF circuits and logs the
VERBATIM error the service returns (if any), rather than trusting either
the 5000 figure or a guess at what "gate limit exceeded" looks like.

THE QUBIT WALL, NOT THE GATE WALL (read this before changing the sweep)
-------------------------------------------------------------------------
build_full_circuit allocates n_total = n_targ + 2*n_prec qubits: TWO
disjoint precision registers (stage 1 and stage 2), because Forte-1 does
not expose `reset` on its OpenQASM submission path, so the precision
register can't be reused between stages (see submit_QPUF_ntarg.py's
module docstring). An earlier version of this sweep's spec assumed the
ceiling was n_prec + n_targ <= 36, which is wrong for this circuit -- it
would try e.g. n_targ=1, n_prec=35 -> 71 qubits on a 36-qubit device,
which fails on QUBIT COUNT, not gate count. Reading that rejection as
"the gate limit" would be exactly the misdiagnosis class CLAUDE.md
Section 2 warns about (a register-format error once misread in this repo
as "MCM unsupported"). The corrected ceiling used throughout this script
is

    n_targ + 2*n_prec <= 36   =>   n_prec <= (36 - n_targ) // 2

enforced by check_qubit_ceiling() below, which is called before a
candidate circuit is even built.

IMPORTANT CAVEAT ON WHAT A CLEAN SWEEP PROVES
-----------------------------------------------
Post-transpile gate counts (1q+2q, Haar U, SEED=10), measured directly
from build_and_transpile() below:
  n_targ=1: n_prec 3..17 -> 141..1,601 gates.  Hits the 36-qubit wall
            (n_prec=17) before ever reaching 5,000 gates.
  n_targ=2: n_prec 3..17 -> 438..3,266 gates.  Same story -- qubit wall
            first, never crosses 5,000.
  n_targ=3: n_prec 3..16 -> 1,915..10,907 gates.  CROSSES 5,000 at
            n_prec=8 (5,263 gates, 19 qubits) -- this is the only arm
            that can actually test a gate-count limit before its own
            qubit wall (n_prec=16, 35 qubits) is reached.

A clean run of the n_targ=1 or n_targ=2 arms therefore proves only "no
compile failure up to ~1,600 / ~3,266 gates respectively" -- it does NOT
prove "no 5000-gate limit exists" on this device. Only the n_targ=3 arm's
behavior around n_prec=8 (5,263 gates) is informative about the claimed
5000-gate ceiling. Read results accordingly; don't over-claim from a
clean n_targ=1/2 run.

Submission pattern
-------------------
On-demand (bare AwsDevice(DEVICE_ARN).run(...), no Braket Direct
reservation), following braket_direct/20260828_ionq_error_mitigation/
submit_ionq_mitigation.py's proven on-demand pattern. 100 shots per task
(device minimum per SHOTS_RANGE=[100,5000] -- this is a compile test, not
data collection).

Cost note: a REJECTED submission (failed compile/validation) may cost
nothing -- the task never runs on the QPU. The $8/task figure quoted
below ($0.08/shot x 100 shots) is therefore an UPPER BOUND on the cost of
a failure, not a guaranteed charge.

Error handling -- read this before "fixing" a log entry
-----------------------------------------------------------
On any submission failure this script logs the VERBATIM exception: type,
str(e), and -- for botocore/Braket ClientError-style exceptions --
Error.Code / Error.Message / ResponseMetadata.HTTPStatusCode /
ResponseMetadata.RequestId, if present. It does NOT normalize, truncate,
or classify the error as "gate limit exceeded" or anything else. This
project already misread one Braket error (a register-format complaint)
as "MCM unsupported" -- raw error text goes to the log for a human to
read, not a keyword match.

Safety -- no submission happens by accident
-----------------------------------------------
Dry run is the DEFAULT (no flags needed): the script only builds,
transpiles, exports QASM, and prints projections -- it never imports
AwsDevice or calls .run(). Real submission requires --execute AND a
typed "y"/"yes" interactive confirmation that reprints device, task
count, and total cost; it is refused outright if stdin is not a TTY (see
confirm_submit()). --dry-run always wins if both flags are given.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from qiskit import transpile, qasm3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Import, don't copy: submit_QPUF_ntarg.py is left untouched, and this
# script reuses its exact circuit builder / Haar sampler / QASM rewriter
# so the probed circuit is bit-for-bit what that campaign would submit.
from submit_QPUF_ntarg import (
    DEVICE_NAME, DEVICE_ARN,
    SEED, TARGET_INIT_SEED,
    haar_random_unitary, build_full_circuit,
    _rewrite_qasm_for_braket,
)

# -- CONFIG -------------------------------------------------------------------
DEVICE_QUBIT_COUNT = 36        # Forte-Enterprise-1 physical qubit count
BASIS_GATES         = ['rz', 'rx', 'rxx', 'measure', 'reset']
OPTIMIZATION_LEVEL  = 1
PROBE_SHOTS         = 100      # device minimum (SHOTS_RANGE floor) -- compile test, not data collection
COST_PER_SHOT_USD   = 0.08
N_PREC_START        = 3
MAX_CONSECUTIVE_FAILURES = 2
N_TARG_LIST_DEFAULT = [1, 2, 3]

# The claim under test, not a trusted constant -- see module docstring.
GATE_LIMIT_UNDER_TEST = 5000

CAMPAIGN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(CAMPAIGN_DIR, "gate_limit_probe_results")
LOG_FILE     = os.path.join(RESULTS_DIR, "job_log.txt")

# Same list submit_QPUF_ntarg.py declares as basis_gates for qasm3.dumps,
# so qiskit emits no `gate` definitions for these -- see that script's
# comment on why `rxx` still needs the separate rewrite below.
BRAKET_BUILTIN_GATES = ['h', 'cx', 'cnot',
                        'rx', 'ry', 'rz', 'xx',
                        'x',  'y',  'z',
                        's',  't',  'swap', 'i']


# -- Qubit ceiling --------------------------------------------------------------

def qubit_ceiling(n_targ: int) -> int:
    """
    Max n_prec for a given n_targ under build_full_circuit's TWO-disjoint-
    precision-register layout (n_total = n_targ + 2*n_prec). See module
    docstring "THE QUBIT WALL, NOT THE GATE WALL".
    """
    return (DEVICE_QUBIT_COUNT - n_targ) // 2


def check_qubit_ceiling(n_targ: int, n_prec: int) -> None:
    """
    Hard pre-submission guard, called before a candidate circuit is even
    built. build_full_circuit uses TWO disjoint precision registers
    (stage 1 + stage 2, no `reset`-based reuse -- see its docstring), so
    the qubit count is n_targ + 2*n_prec, not n_targ + n_prec. Refusing
    here keeps a qubit-count rejection from ever reaching the device and
    being misread as a gate-count rejection -- the exact misdiagnosis
    class this repo hit before with a register-format error read as "MCM
    unsupported" (CLAUDE.md Section 2).
    """
    n_total = n_targ + 2 * n_prec
    if n_total > DEVICE_QUBIT_COUNT:
        raise ValueError(
            f"n_targ={n_targ}, n_prec={n_prec} needs {n_total} qubits "
            f"(= n_targ + 2*n_prec) but {DEVICE_NAME} has "
            f"{DEVICE_QUBIT_COUNT}. This is a QUBIT-COUNT wall, distinct "
            f"from the gate-count limit being probed -- refused before "
            f"build rather than letting the device reject it and "
            f"conflating the two failure modes.")


# -- Circuit build + transpile ---------------------------------------------------

def build_and_transpile(n_prec: int, n_targ: int):
    """
    Build the genuine-MCM two-stage QPUF circuit (build_full_circuit,
    imported unmodified from submit_QPUF_ntarg.py) and transpile it
    exactly as that script does. Fixed SEED/TARGET_INIT_SEED so gate
    counts are reproducible run to run.
    """
    check_qubit_ceiling(n_targ, n_prec)
    rng = np.random.default_rng(seed=SEED)
    U = haar_random_unitary(2 ** n_targ, rng=rng)
    qc = build_full_circuit(n_prec, n_targ, U, target_init_seed=TARGET_INIT_SEED)
    qc_hw = transpile(qc, basis_gates=BASIS_GATES, optimization_level=OPTIMIZATION_LEVEL)

    ops  = qc_hw.count_ops()
    n_1q = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q = ops.get('rxx', 0)
    profile = {
        "n_qubits":      qc_hw.num_qubits,
        "n_1q_gates":    n_1q,
        "n_2q_gates":    n_2q,
        "n_total_gates": n_1q + n_2q,
        "depth":         qc_hw.depth(),
    }
    return qc_hw, U, profile


def to_braket_qasm(qc_hw) -> str:
    """
    OpenQASM 3 export Braket's parser accepts -- rxx isn't a Braket
    built-in, so _rewrite_qasm_for_braket (imported from
    submit_QPUF_ntarg.py) drops the emitted `gate rxx {...}` definition
    and renames call sites to `xx(...)`, the equivalent Braket built-in.
    """
    src = qasm3.dumps(qc_hw, includes=(), basis_gates=BRAKET_BUILTIN_GATES)
    return _rewrite_qasm_for_braket(src)


# -- Sweep enumeration -----------------------------------------------------------

def sweep_candidates(n_targ_list):
    """
    Yield (n_targ, n_prec) in sweep order: n_prec from N_PREC_START
    upward by 1 within each n_targ, stopping that n_targ's range at the
    corrected qubit ceiling. This does NOT apply the 2-consecutive-
    failure stop -- that only makes sense against real submission
    outcomes and is applied in run_execute(); the dry-run projection
    below shows the full ceiling-bounded sweep as an upper bound on task
    count.
    """
    for n_targ in n_targ_list:
        ceiling = qubit_ceiling(n_targ)
        for n_prec in range(N_PREC_START, ceiling + 1):
            yield n_targ, n_prec


# -- Logging ------------------------------------------------------------------------

def append_log(record: dict) -> None:
    """
    Append one JSON record immediately. CLAUDE.md reproducibility
    discipline: a record (including a task ARN, once one exists) is
    logged the moment it's known, not batched at the end -- a crash
    mid-sweep must not lose the record of what already ran.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_records_for(n_targ: int) -> list:
    """All previously-logged attempt records for one n_targ arm, sorted by n_prec."""
    records = []
    if not os.path.exists(LOG_FILE):
        return records
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("n_targ") == n_targ and "n_prec" in rec:
                records.append(rec)
    records.sort(key=lambda r: r["n_prec"])
    return records


def describe_error(e: Exception) -> dict:
    """
    Verbatim error capture -- do NOT normalize/classify. Log exactly what
    the service said and let a human read it; a keyword match is exactly
    what led this project to once misread a register-format error as
    "MCM unsupported".
    """
    out = {"exception_type": type(e).__name__, "exception_str": str(e)}
    response = getattr(e, "response", None)   # botocore ClientError shape
    if isinstance(response, dict):
        err  = response.get("Error", {})
        meta = response.get("ResponseMetadata", {})
        out["response_error_code"]    = err.get("Code")
        out["response_error_message"] = err.get("Message")
        out["response_http_status"]   = meta.get("HTTPStatusCode")
        out["response_request_id"]    = meta.get("RequestId")
    return out


# -- Reporting ------------------------------------------------------------------------

def print_projected_table(n_targ_list):
    """
    Build + transpile every candidate in the sweep (no AWS calls) and
    print the projected gate-count table, task count, and cost. Returns
    the list of (n_targ, n_prec, profile) rows for reuse by run_execute's
    confirmation prompt.
    """
    rows = []
    print(f"{'n_targ':>6} {'n_prec':>6} {'qubits':>6} {'1q':>6} {'2q':>6} "
          f"{'total':>7}  {'>' + str(GATE_LIMIT_UNDER_TEST) + '?':>6}")
    print("-" * 52)
    for n_targ, n_prec in sweep_candidates(n_targ_list):
        _, _, profile = build_and_transpile(n_prec, n_targ)
        crosses = profile["n_total_gates"] > GATE_LIMIT_UNDER_TEST
        rows.append((n_targ, n_prec, profile))
        print(f"{n_targ:>6} {n_prec:>6} {profile['n_qubits']:>6} "
              f"{profile['n_1q_gates']:>6} {profile['n_2q_gates']:>6} "
              f"{profile['n_total_gates']:>7}  {'YES' if crosses else '':>6}")

    n_tasks = len(rows)
    total_cost = n_tasks * PROBE_SHOTS * COST_PER_SHOT_USD
    print("-" * 52)
    print(f"Projected task count : {n_tasks}  (upper bound -- assumes every "
          f"attempt succeeds; the 2-consecutive-failure stop can end an "
          f"arm earlier)")
    print(f"Projected cost       : {n_tasks} x {PROBE_SHOTS} shots x "
          f"${COST_PER_SHOT_USD:.2f}/shot = ${total_cost:.2f}  (upper bound "
          f"-- a rejected submission may cost nothing, see module docstring)")
    return rows


def confirm_submit(question: str) -> bool:
    """
    Final go/no-go before anything is billed. Refuses outright if stdin
    is not a TTY (e.g. piped/heredoc invocation), so this can't be
    triggered non-interactively by accident.
    """
    if not sys.stdin.isatty():
        print(question)
        print("ERROR: stdin is not a terminal, so this cannot be answered "
              "interactively. Re-run directly in a terminal.")
        return False
    try:
        answer = input(question).strip().lower()
    except EOFError:
        print("\nERROR: end of input while waiting for confirmation "
              "(stdin closed).")
        return False
    if answer in ("y", "yes"):
        return True
    print(f"Aborted -- nothing submitted. (got {answer!r}, expected 'y')")
    return False


# -- Real submission (only reachable via --execute) --------------------------------

def run_execute(n_targ_list):
    # Imported here, not at module scope, so a plain `import
    # probe_gate_limit` (e.g. for its constants, in a future script) can
    # never touch boto3/Braket as a side effect.
    from braket.aws import AwsDevice
    from braket.experimental_capabilities import EnableExperimentalCapability
    from braket.ir.openqasm import Program as OpenQasmProgram

    # Full ceiling-bounded candidate set, for the confirmation prompt's
    # task-count/cost display -- matches print_projected_table's upper
    # bound; the 2-consecutive-failure stop may end an arm earlier.
    all_candidates = list(sweep_candidates(n_targ_list))
    already_logged = set()
    for n_targ in n_targ_list:
        already_logged |= {(n_targ, r["n_prec"]) for r in _load_records_for(n_targ)}
    remaining = [c for c in all_candidates if c not in already_logged]

    n_tasks    = len(remaining)
    total_cost = n_tasks * PROBE_SHOTS * COST_PER_SHOT_USD
    print(f"\n{len(already_logged)} attempt(s) already in {LOG_FILE} -- will be skipped.")
    print(f"Up to {n_tasks} new task(s) remain in this sweep "
          f"(upper bound; 2-consecutive-failure stop may end an arm early).")

    question = (f"\nType 'y' to attempt real submissions to {DEVICE_NAME} "
                f"({DEVICE_ARN}): up to {n_tasks} task(s), {PROBE_SHOTS} "
                f"shots each, ~${total_cost:.2f} total (upper bound) [y/N]: ")
    if not confirm_submit(question):
        return

    device = AwsDevice(DEVICE_ARN)
    print(f"\nDevice resolved: {device.name}  (status {device.status})")
    live_qubits = device.properties.paradigm.qubitCount
    if live_qubits != DEVICE_QUBIT_COUNT:
        print(f"NOTE: live device reports {live_qubits} qubits, this script's "
              f"DEVICE_QUBIT_COUNT constant is {DEVICE_QUBIT_COUNT} -- ground "
              f"truth is the live value; the sweep ceiling was computed from "
              f"the hardcoded constant, so re-check qubit_ceiling() if these "
              f"differ.")

    for n_targ in n_targ_list:
        records = _load_records_for(n_targ)
        done_precs = {r["n_prec"] for r in records}

        # Reconstruct the trailing consecutive-failure count from the log
        # so a resumed sweep re-applies the same stop rule a continuous
        # run would have hit.
        consecutive_failures = 0
        for r in reversed(records):
            if r.get("outcome") == "failed":
                consecutive_failures += 1
            else:
                break
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\nn_targ={n_targ}: log already shows "
                  f"{consecutive_failures} consecutive failures -- arm "
                  f"already stopped, skipping entirely.")
            continue

        ceiling = qubit_ceiling(n_targ)
        print(f"\n=== n_targ={n_targ}  (n_prec {N_PREC_START}..{ceiling}) ===")
        for n_prec in range(N_PREC_START, ceiling + 1):
            if n_prec in done_precs:
                print(f"  n_prec={n_prec}: already logged -- skipping (resumability).")
                continue

            timestamp = datetime.now(timezone.utc).isoformat()
            record = {
                "timestamp":        timestamp,
                "n_targ":           n_targ,
                "n_prec":           n_prec,
                "shots":            PROBE_SHOTS,
                "device":           DEVICE_NAME,
                "device_arn":       DEVICE_ARN,
                "seed":             SEED,
                "target_init_seed": TARGET_INIT_SEED,
            }

            try:
                qc_hw, U, profile = build_and_transpile(n_prec, n_targ)
                qasm_src = to_braket_qasm(qc_hw)
                record.update(profile)
            except Exception as e:
                record["outcome"] = "failed"
                record["stage"]   = "build_or_transpile"
                record["error"]   = describe_error(e)
                append_log(record)
                consecutive_failures += 1
                print(f"  n_prec={n_prec}: BUILD/TRANSPILE FAILED -- "
                      f"{type(e).__name__}: {e}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"  {consecutive_failures} consecutive failures -- "
                          f"stopping n_targ={n_targ} arm.")
                    break
                continue

            print(f"  n_prec={n_prec}: {profile['n_qubits']} qubits, "
                  f"{profile['n_total_gates']} gates "
                  f"(1q={profile['n_1q_gates']}, 2q={profile['n_2q_gates']}), "
                  f"depth={profile['depth']} -- submitting {PROBE_SHOTS} shots ...")
            try:
                with EnableExperimentalCapability():
                    task = device.run(OpenQasmProgram(source=qasm_src), shots=PROBE_SHOTS)
                record["outcome"] = "submitted"
                record["job_id"]  = task.id
                append_log(record)
                print(f"    Task ARN: {task.id}")
                consecutive_failures = 0
            except Exception as e:
                record["outcome"] = "failed"
                record["stage"]   = "submit"
                record["job_id"]  = None
                record["error"]   = describe_error(e)
                append_log(record)
                consecutive_failures += 1
                print(f"    SUBMIT FAILED -- {type(e).__name__}: {e}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"    {consecutive_failures} consecutive failures -- "
                          f"stopping n_targ={n_targ} arm.")
                    break

    print(f"\nDone. Log: {LOG_FILE}")


# -- Main -----------------------------------------------------------------------------

def _parse_int_list(s: str) -> list:
    return [int(x) for x in s.replace(",", " ").split()]


def main():
    ap = argparse.ArgumentParser(
        description="Probe the real gate-count/compile limit on IonQ "
                     "Forte-Enterprise-1 using the genuine-MCM two-stage "
                     "QPE-QPUF circuit. Dry run by default; see module "
                     "docstring before using --execute.")
    ap.add_argument("--n-targ-list", type=_parse_int_list, default=N_TARG_LIST_DEFAULT,
                    help="Order and set of n_targ arms to sweep, e.g. "
                         "'3,1,2' to run n_targ=3 first -- it's the only "
                         "arm that can cross the 5000-gate claim before "
                         "hitting its own qubit wall (see module docstring). "
                         f"Default: {N_TARG_LIST_DEFAULT}")
    ap.add_argument("--dry-run", action="store_true",
                    help="Explicit dry run. This is also the default with "
                         "no flags at all; passing it just makes the choice "
                         "visible in the command line. Always wins over "
                         "--execute if both are given.")
    ap.add_argument("--execute", action="store_true",
                    help="Attempt real submissions to AWS. Still requires "
                         "a typed 'y'/'yes' confirmation (refused if stdin "
                         "is not a TTY) that reprints device, task count, "
                         "and total cost before anything is sent.")
    args = ap.parse_args()

    dry_run = (not args.execute) or args.dry_run

    print(f"IonQ Forte-Enterprise-1 gate-limit probe")
    print(f"n_targ arms : {args.n_targ_list}  (qubit ceiling n_prec <= "
          f"(36 - n_targ) // 2 per arm)")
    print(f"mode        : {'DRY RUN (no AWS calls)' if dry_run else 'EXECUTE'}")
    print()

    print_projected_table(args.n_targ_list)

    if dry_run:
        print("\nDRY RUN -- nothing submitted. Pass --execute to attempt "
              "real submissions (still requires typed confirmation).")
        return

    run_execute(args.n_targ_list)


if __name__ == "__main__":
    main()
