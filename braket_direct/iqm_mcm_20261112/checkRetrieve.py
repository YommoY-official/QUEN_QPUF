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
        if r.get("role") in ("readout_cal", "idle_cal"):
            n = sum(counts.values()); one = r["cal_state"] == "1"
            flips = {str(q): sum(v for k, v in counts.items() if (k[j] == "1") != one) / n
                     for j, q in enumerate(r["order"])}
            json.dump(dict(record=r, status=st, submitted_qasm=r["qasm"], measurement_counts=counts, measured_qubits=mq,
                           flip_prob_per_qubit=flips), open(out, "w"), indent=2)
            shown = {q: f for q, f in flips.items() if r["role"] == "readout_cal" or int(q) in r["layout"]["p1"] + r["layout"]["anc"]}
            print(f"  {r['label']:<24} saved  flip prob (record qubits): " + "  ".join(f"${q} {f:.3f}" for q, f in shown.items())); done += 1; continue
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


def load_cal(rdir, role):
    """{qubit: {'0': P(read 1 | prepared 0), '1': P(read 0 | prepared 1)}} from saved calibration tasks of `role`."""
    cal = {}
    for f in os.listdir(rdir):
        if f.endswith(".json"):
            try: j = json.load(open(os.path.join(rdir, f)))
            except Exception: continue
            if j.get("flip_prob_per_qubit") and j["record"].get("role") == role:
                for q, e in j["flip_prob_per_qubit"].items(): cal.setdefault(int(q), {})[j["record"]["cal_state"]] = e
    return cal


def marginals(counts, r):
    """m1 from the ancilla copy, m1 from the re-read P1, and m2, each as a distribution over 0..2^d-1."""
    d = r["d"]; col = {(ro, i): r["order"].index(q) for ro, i, q in r["measure"]}
    val = lambda s, ro: sum(int(s[col[(ro, k)]]) << (d - 1 - k) for k in range(d))
    out = {k: np.zeros(2 ** d) for k in ("m1_ancilla", "m1_P1_reread", "m2")}; n = sum(counts.values())
    for s, c in counts.items():
        out["m1_ancilla"][val(s, "anc")] += c / n; out["m1_P1_reread"][val(s, "P1")] += c / n; out["m2"][val(s, "P2")] += c / n
    return out


def unconfuse(p, qubits, cal, d):
    """Invert the per-qubit readout+idle confusion (independent bits, bit k = weight 2^(d-1-k)); clip and renormalise."""
    C = np.ones((1, 1))
    for q in qubits:
        e0, e1 = cal[q]["0"], cal[q]["1"]; C = np.kron(C, np.array([[1 - e0, e1], [e0, 1 - e1]]))
    x = np.clip(np.linalg.solve(C, p), 0, None); return x / x.sum()


def expected_agreement(r, cal, m1_true):
    """Witness agreement predicted from calibration alone: the ancilla and re-read-P1 bits flip independently
    with the calibrated probabilities, given the true m1 distribution `m1_true`. Compare with the measured
    agreement before attributing any shortfall to measurement back-action."""
    d = r["d"]; meas = {(ro, i): q for ro, i, q in r["measure"]}; p = 1.0
    for i in range(d):
        P1 = sum(m1_true[m] for m in range(2 ** d) if m >> (d - 1 - i) & 1); a, b = meas[("anc", i)], meas[("P1", i)]
        # a flip probability for true bit 0 is cal[q]['0'], for true bit 1 is cal[q]['1']
        p *= sum(w * ((1 - cal[a][t]) * (1 - cal[b][t]) + cal[a][t] * cal[b][t]) for t, w in (("0", 1 - P1), ("1", P1)))
    return p


def analyse_B(rec, f, rdir):
    j = json.load(open(f)); counts = j["measurement_counts"]; d = rec["d"]; mg = marginals(counts, rec)
    np.set_printoptions(precision=3, suppress=True)
    print(f"  {rec['label']}: measured witness agreement {j['ancilla_vs_reread_P1']:.3f}")
    for k, v in mg.items(): print(f"    {k:<14} {v}")
    meas = {(ro, i): q for ro, i, q in rec["measure"]}
    for role, name in (("idle_cal", "idle calibration"), ("readout_cal", "readout calibration")):
        cal = load_cal(rdir, role); need = [meas[(ro, i)] for ro in ("anc", "P1") for i in range(d)]
        if not all(q in cal and len(cal[q]) == 2 for q in need): continue
        anc_q, p1_q = [meas[("anc", i)] for i in range(d)], [meas[("P1", i)] for i in range(d)]
        ca, cp = unconfuse(mg["m1_ancilla"], anc_q, cal, d), unconfuse(mg["m1_P1_reread"], p1_q, cal, d)
        print(f"    corrected with {name}: m1_ancilla {ca}   m1_P1_reread {cp}")
        ex = expected_agreement(rec, cal, ca)
        print(f"    witness agreement: measured {j['ancilla_vs_reread_P1']:.3f}  predicted from {name} alone {ex:.3f}"
              f"  (shortfall = {ex - j['ancilla_vs_reread_P1']:+.3f} beyond what {name} explains)")
        break
    else: print("    no calibration tasks retrieved yet; nothing to correct with")


def main():
    rdir = os.path.join(HERE, "job_results"); log = os.path.join(rdir, "job_log.txt")
    if not os.path.exists(log): sys.exit(f"no {log}")
    recs = read_log(log)
    while True:
        done, pending = process(recs, rdir); print(f"{done}/{len(recs)} retrieved")
        for rec in recs:
            f = os.path.join(rdir, rec["job_id"].rsplit("/", 1)[-1] + ".json")
            if rec.get("arm") == "B" and os.path.exists(f): analyse_B(rec, f, rdir)
        if pending == 0 or "--watch" not in sys.argv: break
        time.sleep(30)


if __name__ == "__main__":
    main()
