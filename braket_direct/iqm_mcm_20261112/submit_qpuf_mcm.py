#!/usr/bin/env python3
"""
submit_qpuf_mcm.py
==================
Genuinely sequential two-stage PE-QPUF on IQM Garnet / Emerald (Braket
experimental dynamic circuits: measure_ff + cc_prx, verbatim, eu-north-1).

    python submit_qpuf_mcm.py --dry-run                 # build everything, print counts + cost, submit NOTHING
    python submit_qpuf_mcm.py --device emerald --d 2 3  # real submission (asks you to type SUBMIT)

HARDWARE GATE (CLAUDE.md s2): without --dry-run this script submits to real
hardware and costs money. It is only to be run by Yon, after an explicit yes
for this exact device / circuits / shots / cost. There is deliberately no
--yes flag: the final confirmation must be typed at a terminal.

Tasks per d: arms {A, B} (+Breuse on request) x targets {haar (U seed 100, init seed 99),
known (phase 1/2^d, target |1>)}. Each task is logged to the manifest
(job_results/job_log.txt) the moment it is submitted, with the OpenQASM
program that was sent.
"""
import argparse, os, sys, warnings
from datetime import datetime, timezone
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iqm_qpuf_common import (DEVICES, haar_random_unitary, known_phase_unitary, encode_unitary,
                             load_caps, build_program, readout_cal_programs, to_qasm, gate_counts, check_dynamic_rules,
                             estimate_cost, est_success, task_tags, append_job_log)

SEED, TARGET_INIT_SEED, N_TARG = 100, 99, 1
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_results")


def build_jobs(device, ds, arms, caps, layout="primary", cal=True):
    jobs = []
    for d in ds:
        for target in ("haar", "known"):
            if target == "haar":
                U, init, phi = haar_random_unitary(2 ** N_TARG, np.random.default_rng(SEED)), TARGET_INIT_SEED, None
            else:
                U, init, phi = known_phase_unitary(N_TARG, 1 / 2 ** d), "ones", 1 / 2 ** d
            for arm in arms:
                p = build_program(arm, d, N_TARG, U, init, caps, layout=layout)
                jobs.append(dict(role="qpuf", label=f"{arm} d={d} {target}", d=d, arm=arm, target=target, phi=phi, U=U,
                                 prog=p, qasm=to_qasm(p), counts=gate_counts(p), rules=check_dynamic_rules(p, caps)))
    if cal:   # readout calibration on exactly the qubits the batch measures
        qs = sorted({q for j in jobs for q in j["prog"]["order"]})
        for st, p in readout_cal_programs(qs).items():
            jobs.append(dict(role="readout_cal", label=f"readout cal |{st}>", d=0, arm="cal", target="cal", phi=None,
                             U=np.eye(2), prog=p, qasm=to_qasm(p), counts=gate_counts(p), rules=check_dynamic_rules(p, caps)))
    return jobs


def report(jobs, device, shots, caps):
    print("=" * 96)
    print(f"{device}  ({DEVICES[device]['arn']})   caps: {'REAL device_caps file' if caps['real'] else 'PLACEHOLDER lattice -- run query_device_caps.py on the DCV'}")
    print("=" * 96)
    print(f"{'task':<24}{'qubits':>7}{'cz':>5}{'prx':>6}{'mff':>5}{'ccprx':>6}{'depth':>7}{'P_surv':>8}  layout (T | P1 | anc | P2)")
    for j in jobs:
        c, p = j["counts"], j["prog"]
        print(f"{j['label']:<24}{c['qubits']:>7}{c['cz']:>5}{c['prx']:>6}{c['measure_ff']:>5}{c['cc_prx']:>6}{c['depth']:>7}{est_success(p, caps['fid']) if caps.get('fid') else float('nan'):>8.4f}  "
              f"{p['t_sites']} | {p['p1_stage1']} | {p['anc']} | {p['p2']}")
    print("\ncopy pairs P1_i -> ancilla_i (cc_prx source -> target), readout fidelity where known:")
    for j in jobs:
        if j["prog"]["arm"] in ("B", "Breuse"):
            adj = {tuple(sorted(e)) for e in caps["edges"]}; fro = caps.get("fro") or {}
            print(f"  {j['label']:<18}" + "  ".join(
                f"${a}->${b} {'COUPLER' if tuple(sorted((a, b))) in adj else 'non-coupler'}"
                f" (ro {fro.get(str(a), '?')}/{fro.get(str(b), '?')})" for a, b in zip(j["prog"]["p1_stage1"], j["prog"]["anc"])))
    errs = sorted({e for j in jobs for e in j["rules"][0]}); warns = sorted({w for j in jobs for w in j["rules"][1]})
    print("\nIQM dynamic-circuit rules:", "OK" if not errs else "VIOLATIONS")
    for e in errs: print("   ERROR  ", e)
    for w in warns: print("   warning", w)
    print("\nCOST (AWS Braket pricing page; not region-specific there):")
    print(f"  {len(jobs)} tasks x {shots} shots")
    for dev, D in DEVICES.items():
        print(f"  {dev:<8} ${D['task_usd']:.2f}/task + ${D['shot_usd']:.5f}/shot  ->  ${estimate_cost(dev, len(jobs), shots):7.2f}"
              f"   ({'selected' if dev == device else 'for comparison'})")
    return errs


def main():
    ap = argparse.ArgumentParser(description="Sequential two-stage PE-QPUF on IQM (dynamic circuits).")
    ap.add_argument("--device", choices=list(DEVICES), default="garnet")
    ap.add_argument("--d", type=int, nargs="+", default=[2, 3], help="precision qubits per stage")
    ap.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "B", "Breuse"])
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--layout", choices=["primary", "adjacent", "nonadj"], default="primary",
                    help="primary: ancilla anywhere in P1's qubit group; adjacent: each P1_i-ancilla_i pair is also a coupler; nonadj: no pair is a coupler, good-readout qubits only")
    ap.add_argument("--smoke", action="store_true",
                    help="single circuit: known-answer arm B, d=2, 200 shots, primary layout (the originally submitted smoke test), no calibration")
    ap.add_argument("--smoke-nonadj", action="store_true",
                    help="like --smoke but nonadj layout: copy pairs are not couplers, good-readout qubits only (program B_d2_known_nonadj.qasm)")
    ap.add_argument("--no-cal", action="store_true", help="omit the two readout-calibration circuits")
    ap.add_argument("--dry-run", action="store_true", help="build and report only; submit nothing")
    a = ap.parse_args()

    caps = load_caps(a.device)
    if a.smoke or a.smoke_nonadj:
        a.smoke = True
        a.d, a.arms, a.shots, a.layout, a.no_cal = [2], ["B"], 200, "nonadj" if a.smoke_nonadj else "primary", True
    try:
        jobs = build_jobs(a.device, a.d, a.arms, caps, a.layout, not a.no_cal)
    except RuntimeError as e:
        sys.exit(f"Could not build the {a.layout!r} layout for d={a.d}: {e}")
    if a.smoke: jobs = [j for j in jobs if j["target"] == "known"]
    errs = report(jobs, a.device, a.shots, caps)
    if errs:
        print("\nRefusing: rule violations above."); sys.exit(1)
    os.makedirs(os.path.join(RESULTS_DIR, "programs"), exist_ok=True)
    print(f"\nlayout mode: {a.layout}")
    for j in jobs:                                   # inspectable copy of every program, even on --dry-run
        open(os.path.join(RESULTS_DIR, "programs", j["label"].replace(" ", "_").replace("=", "") + ("_nonadj" if a.layout == "nonadj" else "") + ".qasm"), "w").write(j["qasm"])
    if a.dry_run:
        print("\n--dry-run: nothing submitted. Programs written to job_results/programs/."); return
    if not caps["real"]:
        print("\nRefusing: submission needs the real device_caps file (placeholder lattice would not match the chip)."); sys.exit(1)
    if not sys.stdin.isatty():
        print("\nRefusing: not a terminal; the confirmation must be typed."); sys.exit(1)
    cost = estimate_cost(a.device, len(jobs), a.shots)
    print(f"\nThis submits {len(jobs)} tasks to REAL hardware {a.device}, {a.shots} shots each, about ${cost:.2f}.")
    if input("Type SUBMIT to proceed: ").strip() != "SUBMIT":
        print("Aborted -- nothing submitted."); return

    from braket.aws import AwsDevice
    from braket.ir.openqasm import Program
    D = DEVICES[a.device]; dev = AwsDevice(D["arn"])
    print(f"Device {dev.name}: {dev.status}")
    for j in jobs:
        # experimental_capabilities="ALL" sets experimentalCapabilities on the task (installed SDK, aws_quantum_task.py)
        task = dev.run(Program(source=j["qasm"]), shots=a.shots, experimental_capabilities="ALL",
                       tags=task_tags("qpuf-mcm", {"Role": j["role"], "Arm": j["arm"], "d": j["d"], "Target": j["target"]}))
        p = j["prog"]
        rec = dict(role=j["role"], cal_state=p.get("cal_state"), layout_mode=a.layout, job_id=task.id, datetime=datetime.now(timezone.utc).isoformat(), qpu=a.device, device_arn=D["arn"],
                   region=D["region"], label=j["label"], arm=j["arm"], d=j["d"], n_targ=N_TARG, target=j["target"],
                   known_phase=j["phi"], n_shots=a.shots, verbatim=True, seed=SEED, target_init_seed=TARGET_INIT_SEED,
                   measure=[list(m) for m in p["measure"]], order=p["order"], gate_counts=j["counts"],
                   layout=dict(t=p["t_sites"], p1=p["p1_stage1"], anc=p["anc"], p2=p["p2"]),
                   caps_real=caps["real"], unitary=encode_unitary(j["U"]), qasm=j["qasm"])
        append_job_log(RESULTS_DIR, rec)             # log immediately, one record per task
        print(f"  {j['label']:<24} {task.id}")
    print(f"\n{len(jobs)} tasks submitted. Retrieve with:  python checkRetrieve.py")


if __name__ == "__main__":
    main()
