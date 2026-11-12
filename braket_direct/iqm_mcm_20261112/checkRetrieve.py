#!/usr/bin/env python3
"""
checkRetrieve.py -- companion to submit_qpuf_mcm.py.

Reads job_results/job_log.txt, queries AWS for each task, and for COMPLETED
tasks writes job_results/<uuid>.json containing: the manifest record, the
SUBMITTED OpenQASM program (proof the measurement was mid-circuit), the raw
measurement counts, the measured physical qubits, the decoded (m1,m2) joint
distribution and the ancilla/re-read-P1 agreement. Idempotent.

    python checkRetrieve.py            # job_results/
    python checkRetrieve.py --watch    # poll until all land
Only tasks in job_log.txt are touched.
"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from iqm_qpuf_common import decode


def read_log(path):
    text, dec, i, out = open(path).read(), json.JSONDecoder(), 0, []
    while i < len(text):
        while i < len(text) and text[i].isspace(): i += 1
        if i >= len(text): break
        o, i = dec.raw_decode(text, i); out.append(o)
    return out


def process(records, rdir):
    from braket.aws import AwsQuantumTask
    done = pending = 0
    for r in records:
        uid = r["job_id"].rsplit("/", 1)[-1]; out = os.path.join(rdir, f"{uid}.json")
        if os.path.exists(out): done += 1; continue
        task = AwsQuantumTask(r["job_id"]); st = task.state()
        if st != "COMPLETED":
            print(f"  {r['label']:<24} {st}"); pending += 1; continue
        res = task.result()
        counts = dict(res.measurement_counts)
        # Braket returns the bit string over the measured qubits in ascending physical order,
        # which is how the program numbers b[j] (see to_braket). Check it explicitly:
        mq = [int(q) for q in (res.measured_qubits or [])]
        if mq and mq != r["order"]: print(f"  WARNING {r['label']}: measured_qubits {mq} != logged order {r['order']}")
        if r.get("role") == "readout_cal":
            n = sum(counts.values()); one = r["cal_state"] == "1"
            flips = {str(q): sum(v for k, v in counts.items() if (k[j] == "1") != one) / n
                     for j, q in enumerate(r["order"])}
            json.dump(dict(record=r, status=st, submitted_qasm=r["qasm"], measurement_counts=counts, measured_qubits=mq,
                           flip_prob_per_qubit=flips), open(out, "w"), indent=2)
            print(f"  {r['label']:<24} saved  max flip {max(flips.values()):.3f}"); done += 1; continue
        M, agree = decode(counts, dict(arm=r["arm"], d=r["d"], measure=r["measure"], order=r["order"]))
        payload = dict(record=r, status=st, submitted_qasm=r["qasm"], measurement_counts=counts, measured_qubits=mq,
                       decoded_m1m2=M.tolist(), ancilla_vs_reread_P1=agree,
                       task_metadata=json.loads(json.dumps(res.task_metadata.__dict__, default=str)))
        json.dump(payload, open(out, "w"), indent=2)
        d = r["d"]; i = np.arange(2 ** d); dl = np.abs(i[:, None] - i[None, :]); dl = np.minimum(dl, 2 ** d - dl)
        print(f"  {r['label']:<24} saved  P(D=0)={M[dl == 0].sum():.3f}  P(D<=2)={M[dl <= 2].sum():.3f}"
              + ("" if agree is None else f"  anc==P1reread={agree:.3f}"))
        done += 1
    return done, pending


def expected_agreement(r, rdir):
    """Witness agreement predicted from readout calibration alone (independent flips of the
    ancilla and re-read-P1 bits). Compare with the measured agreement before attributing any
    shortfall to measurement back-action."""
    cal = {}
    for f in os.listdir(rdir):
        if f.endswith(".json"):
            try: j = json.load(open(os.path.join(rdir, f)))
            except Exception: continue
            if j.get("flip_prob_per_qubit"):
                st = j["record"]["cal_state"]
                for q, e in j["flip_prob_per_qubit"].items(): cal.setdefault(int(q), {})[st] = e
    meas = {(role, i): q for role, i, q in r["measure"]}
    if any(meas[("anc", i)] not in cal or meas[("P1", i)] not in cal for i in range(r["d"])): return None
    e = lambda q: 0.5 * (cal[q]["0"] + cal[q]["1"])
    p = 1.0
    for i in range(r["d"]):
        ea, ep = e(meas[("anc", i)]), e(meas[("P1", i)]); p *= (1 - ea) * (1 - ep) + ea * ep
    return p


def main():
    rdir = os.path.join(HERE, "job_results"); log = os.path.join(rdir, "job_log.txt")
    if not os.path.exists(log): sys.exit(f"no {log}")
    recs = read_log(log)
    while True:
        done, pending = process(recs, rdir); print(f"{done}/{len(recs)} retrieved")
        for rec in recs:
            f = os.path.join(rdir, rec["job_id"].rsplit("/", 1)[-1] + ".json")
            if rec.get("arm") == "B" and os.path.exists(f):
                ex = expected_agreement(rec, rdir); got = json.load(open(f))["ancilla_vs_reread_P1"]
                if ex is not None: print(f"  {rec['label']:<24} witness agreement {got:.3f}  vs readout-only prediction {ex:.3f}")
        if pending == 0 or "--watch" not in sys.argv: break
        time.sleep(30)


if __name__ == "__main__":
    main()
