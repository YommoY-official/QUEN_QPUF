#!/usr/bin/env python3
"""
run_all_families.py
====================
Runs the qpe_mitigation_analysis.ipynb pipeline (REM, ZNE, REM+ZNE, table,
export) over EVERY run family in job_results_qpe/, instead of the single
SELECT_RUN'd family the notebook analyses interactively.

The grouping/loading helpers below (load_results, family_key, assign_batches,
slot_tasks, pool_counts, resolve_clbit_to_phys, _when) are mirrored 1:1 from
the notebook's sections 2-3 -- that logic is small, notebook-cell-local
infrastructure with no importable home, not the thing this file is meant to
avoid duplicating. `family_key` in particular MUST stay byte-for-byte
identical to the notebook's version (7-tuple incl. `measured_qubits`) or a
family here and a family there will disagree about what counts as one
experiment.

What IS reused rather than re-implemented is the actual estimator: every
REM/ZNE/bootstrap number below comes from a call into mitig_analysis.py,
exactly as the notebook cells do. This file only owns the per-family loop
and the cross-family rollup.

Run with the project's braket-arm environment (its Qiskit/Braket/Mitiq
pins), not the interpreter on PATH -- see README.md / the notebook's kernel.
"""

import csv
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MODDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _MODDIR)

import mitig_analysis as ma
from rigetti_qpuf_common import (
    load_device_caps, transpile_for_rigetti, build_qpe_circuit,
    measured_physical_qubits,
)
try:
    from rigetti_qpuf_common import place_and_route
except ImportError:
    place_and_route = None

RESULTS_DIR = os.path.join(_MODDIR, "job_results_qpe")
PLOTS_DIR = os.path.join(_MODDIR, "plots")

BATCH_GAP_S = 30          # see qpe_mitigation_analysis.ipynb section 0 for the derivation
N_BOOT = 500
REM_METHOD = "simplex"
ZNE_MODELS = ("richardson", "linear", "exponential")
ZNE_HEADLINE = "linear"
SEED = 0

# REM inverts a dense 2^n_prec x 2^n_prec confusion matrix via constrained
# NNLS (mitig_analysis.rem_correct, method="simplex"). Measured cost on this
# machine: n_prec=8 (dim=256) 0.01s/call, n_prec=10 (dim=1024) 0.7s/call,
# n_prec=12 (dim=4096) 55s/call -- consistent with the ~O(dim^3) NNLS active-
# set cost. Extrapolating that scaling to n_prec=15 (dim=32768, the largest
# n_prec actually present in job_results_qpe/) puts a SINGLE inversion at
# several hours, before even the bootstrap loop's few hundred repeats, and
# the dense matrix itself is ~8.6 GB. That is not a "slow but correct"
# situation, it is a different computational regime the current tensored,
# non-structure-exploiting `rem_correct` was not built for. So REM (and
# anything downstream of it: ZNE+REM) is skipped above this cutoff rather
# than attempted -- flagged in the export as `rem_skip_reason`, not silently
# dropped. ZNE alone (no matrix inversion) is unaffected and still runs.
REM_MAX_NPREC = 12


# =========================================================================
# Grouping / loading -- mirrors qpe_mitigation_analysis.ipynb sections 2-3
# =========================================================================

def _when(r):
    """Submission time, always timezone-aware. See notebook section 2 for why."""
    try:
        t = datetime.fromisoformat(r["datetime"])
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def load_results(results_dir):
    paths = sorted(p for p in glob.glob(os.path.join(results_dir, "*.json"))
                   if not os.path.basename(p).startswith("analysis_summary"))
    if not paths:
        raise FileNotFoundError(f"No *.json in {results_dir}.")
    out, skipped = [], []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        r["_path"] = os.path.basename(p)
        if not r.get("counts") or not r.get("role"):
            skipped.append(r["_path"])
            continue
        out.append(r)
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with no counts/role: "
              f"{', '.join(skipped[:4])}{' ...' if len(skipped) > 4 else ''}")
    return out


def family_key(r):
    """
    Identical to the notebook's family_key (section 2) -- do not let this
    drift out of sync with it. `measured_qubits` is last so SELECT_RUN-style
    prefix matching still works; placement is in the key because pooling two
    different placements' counts would silently corrupt both the merged
    distribution and the REM correction (see the notebook docstring for the
    full argument).
    """
    return (r.get("n_prec"), r.get("n_targ"), r.get("state_mode"),
            r.get("phi"), tuple(r.get("zne_scales_all") or [r.get("zne_scale", 1)]),
            r.get("ddd_sequence") or "none",
            tuple(int(q) for q in (r.get("measured_qubits") or [])))


def assign_batches(records, gap_s):
    if records and all(r.get("batch_id") for r in records):
        out = {}
        for r in sorted(records, key=_when):
            out.setdefault(r["batch_id"], []).append(r)
        return out, "logged batch_id"

    out, cur, prev, idx = {}, [], None, 1
    for r in sorted(records, key=_when):
        t = _when(r)
        if prev is not None and (t - prev).total_seconds() > gap_s:
            out[f"inferred-{idx}"] = cur
            cur, idx = [], idx + 1
        cur.append(r)
        prev = t
    if cur:
        out[f"inferred-{idx}"] = cur
    return out, f"inferred from timestamps (gap > {gap_s}s)"


def slot_tasks(records):
    qpe, cal = {}, {}
    for r in sorted(records, key=_when):
        if r.get("role") == "qpe":
            qpe[float(r["zne_scale"])] = r
        elif r.get("role") == "readout_cal":
            cal[r["cal_state"]] = r
    return qpe, cal


def pool_counts(records):
    merged = dict(records[0])
    counts = {}
    for r in records:
        for k, v in r["counts"].items():
            counts[k] = counts.get(k, 0) + v
    merged["counts"] = counts
    merged["n_shots"] = sum(counts.values())
    merged["_pooled_from"] = [r.get("batch_id") or r["job_id"] for r in records]
    return merged


def resolve_clbit_to_phys(rec, n_meas):
    """Return (mapping, provenance). See notebook section 3 for the three tiers."""
    ctp = rec.get("clbit_to_phys")
    if ctp:
        return [int(q) for q in ctp], "logged"

    logged = sorted(int(q) for q in rec.get("physical_qubits", []))
    try:
        u = rec["unitary"]
        U = np.array(u["real"], dtype=float) + 1j * np.array(u["imag"], dtype=float)
        caps, caps_real = load_device_caps()
        if not caps_real:
            raise RuntimeError("device_caps.json absent -- routing would differ")
        qc = build_qpe_circuit(
            rec["n_prec"], rec["n_targ"], U,
            target_init_seed=rec.get("target_init_seed"),
            eigenstate=(rec.get("state_mode") == "known"),
        )
        routers = []
        if place_and_route is not None:
            routers.append(("place_and_route", lambda c: place_and_route(
                c, caps, hub_qubits=list(range(rec["n_targ"])), verbose=False)))
        routers.append(("transpile_for_rigetti",
                        lambda c: transpile_for_rigetti(c, caps)))
        for name, route in routers:
            derived = measured_physical_qubits(route(qc))
            if logged and sorted(int(q) for q in derived) == logged:
                return [int(q) for q in derived], f"re-derived ({name})"
        raise RuntimeError(f"no router reproduced the logged qubit set {logged}")
    except Exception:
        pass

    phys = [int(q) for q in rec.get("physical_qubits", [])][:n_meas]
    if len(phys) < n_meas:
        raise RuntimeError("cannot establish a clbit -> physical map for this record")
    return phys, "ASSUMED-SORTED"


def pooled_lambda1(records):
    batches, _ = assign_batches(records, BATCH_GAP_S)
    per = [slot_tasks(r)[0] for r in batches.values()]
    base = min({s for q in per for s in q}, default=None)
    if base is None:
        return None
    return pool_counts([q[base] for q in per if base in q])


def bootstrap_zne_rem(counts_by_scale, cal0, cal1, n_prec, ideal_bin, n_meas,
                      n_boot, seed, models, method, **conv):
    """Same as notebook section 7 -- calls into ma for every actual statistic."""
    rng = np.random.default_rng(seed)
    scales = sorted(counts_by_scale)
    per_model = {m: [] for m in models}
    for _ in range(n_boot):
        a = ma.tensored_confusion(
            ma.confusion_matrices(ma.resample_counts(cal0, rng),
                                  ma.resample_counts(cal1, rng),
                                  n_meas, **conv)[:n_prec])
        vals = []
        for s in scales:
            p = ma.to_probabilities(
                ma.counts_to_m(ma.resample_counts(counts_by_scale[s], rng),
                               n_prec, **conv), n_prec)
            vals.append(ma.p_correct(ma.rem_correct(p, a, method=method), ideal_bin))
        fit = ma.extrapolate(scales, vals, models=models)
        for m in models:
            per_model[m].append(fit.get(m))
    return {m: ma.bootstrap_ci(v) for m, v in per_model.items()}


# =========================================================================
# Collision-safe export filename
# =========================================================================

def family_slug(key):
    """
    analysis_summary_nprec{N}_ntarg{T}_{mode}_phi{P}_scales{S}_ddd{D}_q{H}.json

    n_prec/n_targ/state_mode/phi/scales/ddd are spelled out because they're
    what a human scans for. `measured_qubits` (up to 15+ physical qubit
    indices) is not practical to put in a filename, so it collapses to a
    10-hex-char sha1 of the FULL 7-tuple key -- not just the qubit tuple, the
    whole key -- so the filename is collision-safe against every field the
    family grouping actually uses, not merely the one that's easiest to
    forget (which is exactly the bug being fixed here: the old naming scheme
    dropped zne_scales and placement and let two real families overwrite
    each other). The full qubit list is still written into the JSON body
    (`qubits.measured_qubits`) so nothing is actually lost, only shortened
    for the filesystem.
    """
    n_prec, n_targ, mode, phi, scales, ddd, qubits = key
    scales_s = "-".join(
        (str(int(s)) if float(s).is_integer() else f"{s:g}".replace(".", "p"))
        for s in scales
    )
    phi_s = "none" if phi is None else f"{phi:g}".replace(".", "p").replace("-", "m")
    khash = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
    return (f"nprec{n_prec}_ntarg{n_targ}_{mode}_phi{phi_s}"
            f"_scales{scales_s}_ddd{ddd}_q{khash}")


# =========================================================================
# Per-family analysis -- mirrors notebook sections 3, 4, 5, 6, 7, 7b, 8, 11
# =========================================================================

def analyze_family(key, fam, families):
    n_prec, n_targ, state_mode, phi, scales_logged, ddd_seq, fam_qubits = key

    BATCHES, BATCH_PROV = assign_batches(fam, BATCH_GAP_S)
    slots = [slot_tasks(recs) for recs in BATCHES.values()]
    qpe_by_scale, cal_by_state = {}, {}
    for s in sorted({s for q, _ in slots for s in q}):
        group = [q[s] for q, _ in slots if s in q]
        qpe_by_scale[s] = pool_counts(group)
    for cs in sorted({c for _, cal in slots for c in cal}):
        group = [cal[cs] for _, cal in slots if cs in cal]
        cal_by_state[cs] = pool_counts(group)

    SCALES = sorted(qpe_by_scale)
    if not SCALES:
        return None, "no QPE (role='qpe') task in this family -- only calibration data"

    HAS_ZNE = len(SCALES) > 1
    HAS_REM_DATA = {"cal0", "cal1"} <= set(cal_by_state)
    DIM = 2 ** n_prec
    rem_intractable = HAS_REM_DATA and n_prec > REM_MAX_NPREC
    HAS_REM = HAS_REM_DATA and not rem_intractable
    rem_skip_reason = (
        None if HAS_REM else
        "no cal0/cal1 in this family" if not HAS_REM_DATA else
        f"n_prec={n_prec} (dim={DIM}) exceeds REM_MAX_NPREC={REM_MAX_NPREC}: "
        f"tensored-confusion NNLS inversion is ~O(dim^3) and measured "
        f"~55s/call at dim=4096, so a single call here is estimated at "
        f"several hours -- skipped rather than attempted."
    )

    ref = qpe_by_scale[SCALES[0]]
    MEASURED_QUBITS = [int(q) for q in ref["measured_qubits"]]
    N_MEAS = len(MEASURED_QUBITS)
    try:
        CLBIT_TO_PHYS, PROVENANCE = resolve_clbit_to_phys(ref, N_MEAS)
    except Exception as e:
        return None, f"cannot resolve clbit->physical map: {e}"

    for r in fam:
        mq = [int(q) for q in r["measured_qubits"]]
        if mq != MEASURED_QUBITS:
            return None, (f"task {r['job_id']} measured {mq}, expected "
                          f"{MEASURED_QUBITS} -- inconsistent placement within "
                          f"one family (should be impossible given family_key)")

    CONV = dict(convention="braket", clbit_to_phys=CLBIT_TO_PHYS,
                measured_qubits=MEASURED_QUBITS)

    IDEAL_BIN = (int(round(ref["ideal_bin"])) if ref.get("ideal_bin") is not None
                 else int(max((ref.get("ideal_distribution") or {"0": 1.0}).items(),
                              key=lambda kv: kv[1])[0]))

    counts_by_scale = {s: qpe_by_scale[s]["counts"] for s in SCALES}
    p_by_scale = {s: ma.to_probabilities(ma.counts_to_m(c, n_prec, **CONV), n_prec)
                  for s, c in counts_by_scale.items()}

    ideal = np.zeros(DIM)
    for m, p in (ref.get("ideal_distribution") or {}).items():
        if int(m) < DIM:
            ideal[int(m)] = p
    if ideal.sum() == 0:
        ideal[IDEAL_BIN] = 1.0

    # -- batch drift anchors (section 4b) ------------------------------------
    def batch_anchor(recs):
        qpe, cal = slot_tasks(recs)
        base = min(qpe) if qpe else None
        if base is None:
            return None
        p = ma.to_probabilities(ma.counts_to_m(qpe[base]["counts"], n_prec, **CONV), n_prec)
        row = {"scale": base, "shots": sum(qpe[base]["counts"].values()),
               "raw": ma.p_correct(p, IDEAL_BIN), "rem": None}
        if HAS_REM and {"cal0", "cal1"} <= set(cal):
            a = ma.tensored_confusion(
                ma.confusion_matrices(cal["cal0"]["counts"], cal["cal1"]["counts"],
                                      N_MEAS, **CONV)[:n_prec])
            row["rem"] = ma.p_correct(ma.rem_correct(p, a, method=REM_METHOD), IDEAL_BIN)
        return row

    anchors = {b: a for b, a in
               ((b, batch_anchor(r)) for b, r in BATCHES.items()) if a}

    # -- REM (section 5) -----------------------------------------------------
    p_rem_by_scale = A = mats = COND = rem_boot = None
    if HAS_REM:
        cal0_counts = cal_by_state["cal0"]["counts"]
        cal1_counts = cal_by_state["cal1"]["counts"]
        mats = ma.confusion_matrices(cal0_counts, cal1_counts, N_MEAS, **CONV)
        A = ma.tensored_confusion(mats[:n_prec])
        COND = ma.condition_number(A)
        p_rem_by_scale = {s: ma.rem_correct(p, A, method=REM_METHOD)
                          for s, p in p_by_scale.items()}
        rem_boot = ma.bootstrap_rem_paired(
            counts_by_scale[SCALES[0]], cal0_counts, cal1_counts,
            n_prec, IDEAL_BIN, N_MEAS, n_boot=N_BOOT, method=REM_METHOD,
            seed=SEED, **CONV)

    # -- ZNE / ZNE+REM (section 6-7) -----------------------------------------
    zne_fit = zne_rem_fit = zne_boot = zne_rem_boot = None
    if HAS_ZNE:
        vals_raw = [ma.p_correct(p_by_scale[s], IDEAL_BIN) for s in SCALES]
        zne_fit = ma.extrapolate(SCALES, vals_raw, models=ZNE_MODELS)
        zne_boot = ma.bootstrap_zne(counts_by_scale, n_prec, IDEAL_BIN,
                                    n_boot=N_BOOT, seed=SEED, models=ZNE_MODELS, **CONV)
        if HAS_REM:
            vals_rem = [ma.p_correct(p_rem_by_scale[s], IDEAL_BIN) for s in SCALES]
            zne_rem_fit = ma.extrapolate(SCALES, vals_rem, models=ZNE_MODELS)
            zne_rem_boot = bootstrap_zne_rem(
                counts_by_scale, cal_by_state["cal0"]["counts"],
                cal_by_state["cal1"]["counts"], n_prec, IDEAL_BIN, N_MEAS,
                N_BOOT, SEED, ZNE_MODELS, REM_METHOD, **CONV)

    # -- DDD siblings (section 7b) -- every family this round is ddd=none, so
    # this will always find exactly one arm and report nothing to compare.
    siblings = {k[5]: v for k, v in families.items()
               if k[:5] == key[:5] and k[6] == key[6]}
    ddd_deltas = None
    if len(siblings) >= 2:
        baseline_seq = "none" if "none" in siblings else sorted(siblings)[0]
        base_rec = pooled_lambda1(siblings[baseline_seq])
        base_mq = [int(q) for q in base_rec["measured_qubits"]]
        ddd_deltas = {}
        for seq in sorted(siblings):
            if seq == baseline_seq:
                continue
            rec = pooled_lambda1(siblings[seq])
            if [int(q) for q in rec["measured_qubits"]] != base_mq:
                continue
            ddd_deltas[seq] = ma.bootstrap_delta_unpaired(
                base_rec["counts"], rec["counts"], n_prec, IDEAL_BIN,
                n_boot=N_BOOT, seed=SEED, **CONV)

    # -- table (section 8) ---------------------------------------------------
    rows = [{
        "condition": "raw (lambda=1)",
        "p_correct": ma.p_correct(p_by_scale[SCALES[0]], IDEAL_BIN),
        "tvd": ma.tvd(p_by_scale[SCALES[0]], ideal),
        "phase_err": ma.phase_error(p_by_scale[SCALES[0]], phi, n_prec) if phi is not None else None,
        "ci": rem_boot["raw"] if rem_boot else None,
        "valid": True,
    }]
    if HAS_REM:
        rows.append({
            "condition": f"REM ({REM_METHOD})",
            "p_correct": ma.p_correct(p_rem_by_scale[SCALES[0]], IDEAL_BIN),
            "tvd": ma.tvd(p_rem_by_scale[SCALES[0]], ideal),
            "phase_err": ma.phase_error(p_rem_by_scale[SCALES[0]], phi, n_prec) if phi is not None else None,
            "ci": rem_boot["rem"], "valid": True,
        })
    if HAS_ZNE:
        v = zne_fit.get(ZNE_HEADLINE)
        # "valid" gates on extrapolate()'s own `physical` flag, which checks
        # EVERY fitted model (richardson/linear/poly2/exponential), not just
        # the headline one -- a linear fit can land inside [0,1] by luck
        # while richardson or exponential blow up on the same points, and
        # that disagreement (mitig_analysis.extrapolate's `unphysical` list)
        # is itself evidence the extrapolation isn't trustworthy here.
        rows.append({
            "condition": f"ZNE ({ZNE_HEADLINE})", "p_correct": v,
            "tvd": None, "phase_err": None, "ci": zne_boot["ci"][ZNE_HEADLINE],
            "valid": v is not None and bool(zne_fit.get("physical")),
        })
    if HAS_ZNE and HAS_REM:
        v = zne_rem_fit.get(ZNE_HEADLINE)
        rows.append({
            "condition": f"ZNE+REM ({ZNE_HEADLINE})", "p_correct": v,
            "tvd": None, "phase_err": None, "ci": zne_rem_boot[ZNE_HEADLINE],
            "valid": v is not None and bool(zne_rem_fit.get("physical")),
        })

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "synthetic": False,
        "run": {"n_prec": n_prec, "n_targ": n_targ, "state_mode": state_mode,
                "phi": phi, "ideal_bin": IDEAL_BIN, "scales": SCALES,
                "shots_per_task": sum(counts_by_scale[SCALES[0]].values())},
        "qubits": {"measured_qubits": MEASURED_QUBITS,
                   "clbit_to_phys": CLBIT_TO_PHYS, "provenance": PROVENANCE},
        "batches": {"mode": "pooled", "provenance": BATCH_PROV,
                    "ids": list(BATCHES), "anchors": anchors},
        "ddd": {"sequence": ddd_seq, "arms_present": sorted(siblings),
                "deltas": ddd_deltas},
        "rem": None if not HAS_REM else {
            "method": REM_METHOD, "condition_number": COND,
            "p_1_given_0": [float(m[1, 0]) for m in mats[:n_prec]],
            "p_0_given_1": [float(m[0, 1]) for m in mats[:n_prec]],
            "bootstrap": rem_boot},
        "rem_skip_reason": rem_skip_reason,
        "zne": None if not HAS_ZNE else {"fit": zne_fit, "bootstrap": zne_boot},
        "zne_rem": None if not (HAS_ZNE and HAS_REM) else {
            "fit": zne_rem_fit, "bootstrap": zne_rem_boot},
        "table": rows,
        "task_arns": {r.get("label", r["job_id"]): r["job_id"] for r in fam},
    }
    return summary, None


# =========================================================================
# Cross-family rollup: table + figure
# =========================================================================

def rollup_row(key, summary, slug, filename):
    n_prec, n_targ, state_mode, phi, scales, ddd, qubits = key
    rows_by_cond = {r["condition"]: r for r in summary["table"]}
    raw = rows_by_cond.get("raw (lambda=1)", {})
    rem = next((r for c, r in rows_by_cond.items() if c.startswith("REM")), None)
    zne = next((r for c, r in rows_by_cond.items() if c.startswith("ZNE (")), None)
    zne_rem = next((r for c, r in rows_by_cond.items() if c.startswith("ZNE+REM")), None)

    def delta(r):
        return None if (r is None or r["p_correct"] is None or raw.get("p_correct") is None) \
            else r["p_correct"] - raw["p_correct"]

    return {
        "n_prec": n_prec, "n_targ": n_targ, "state_mode": state_mode, "phi": phi,
        "scales": list(scales), "ddd_sequence": ddd, "n_qubits_measured": len(qubits),
        "export_file": filename,
        "p_raw": raw.get("p_correct"),
        "rem_available": rem is not None, "rem_skip_reason": summary.get("rem_skip_reason"),
        "p_rem": None if rem is None else rem["p_correct"],
        "delta_rem": delta(rem), "delta_rem_valid": None if rem is None else rem["valid"],
        "zne_available": zne is not None,
        "p_zne": None if zne is None else zne["p_correct"],
        "delta_zne": delta(zne), "delta_zne_valid": None if zne is None else zne["valid"],
        "zne_rem_available": zne_rem is not None,
        "p_zne_rem": None if zne_rem is None else zne_rem["p_correct"],
        "delta_zne_rem": delta(zne_rem),
        "delta_zne_rem_valid": None if zne_rem is None else zne_rem["valid"],
    }


def write_table(rows, path_csv, path_json):
    with open(path_json, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    fields = list(rows[0].keys()) if rows else []
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path_json}\nwrote {path_csv}")


def plot_rollup(rows, path_png):
    """
    delta P_correct (mitigated - raw) vs n_prec, one marker shape per n_targ,
    one colour per technique. Hollow/hatched markers flag `valid=False`
    (extrapolation landed outside [0,1] -- see mitig_analysis.extrapolate)
    so an unphysical point is visibly different from a real result, not
    silently plotted as if it were one.
    """
    C = {"rem": "#1baf7a", "zne": "#eda100", "zne_rem": "#e87ba4"}
    MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}
    techniques = [("delta_rem", "delta_rem_valid", "rem_available", "rem", "REM"),
                  ("delta_zne", "delta_zne_valid", "zne_available", "zne", "ZNE (linear)"),
                  ("delta_zne_rem", "delta_zne_rem_valid", "zne_rem_available", "zne_rem",
                   "ZNE+REM (linear)")]
    offsets = {"rem": -0.15, "zne": 0.0, "zne_rem": 0.15}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    seen_targ, seen_tech = set(), set()
    for r in rows:
        n_targ = r["n_targ"]
        marker = MARKERS.get(n_targ, "x")
        for dkey, vkey, akey, colkey, label in techniques:
            if not r[akey] or r[dkey] is None:
                continue
            valid = r[vkey]
            x = r["n_prec"] + offsets[colkey]
            y = r[dkey]
            ax.scatter(x, y, marker=marker, s=70,
                       facecolor=C[colkey] if valid else "none",
                       edgecolor=C[colkey], linewidth=1.6,
                       alpha=0.9 if valid else 0.6, zorder=3)
            seen_targ.add(n_targ)
            seen_tech.add(colkey)

    ax.axhline(0.0, color="#8a8984", linewidth=0.8, zorder=1)
    ax.set_xlabel("n_prec (precision qubits)")
    ax.set_ylabel(r"$\Delta P_{correct}$ = mitigated $-$ raw")
    ax.set_title("REM / ZNE / ZNE+REM benefit vs n_prec, across all analysed families\n"
                 "(hollow marker = extrapolation landed outside [0,1]: unphysical, not a result)")
    ax.grid(axis="x", visible=False)

    from matplotlib.lines import Line2D
    tech_handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=C[c],
                           markeredgecolor=C[c], label=lbl, markersize=8)
                    for _, _, _, c, lbl in techniques if c in seen_tech]
    targ_handles = [Line2D([0], [0], marker=MARKERS[t], linestyle="", markerfacecolor="none",
                           markeredgecolor="#52514e", label=f"n_targ={t}", markersize=8)
                    for t in sorted(seen_targ)]
    leg1 = ax.legend(handles=tech_handles, loc="upper right", title="technique", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=targ_handles, loc="lower right", title="n_targ", fontsize=9)

    fig.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path_png}")


# =========================================================================
# Main
# =========================================================================

def main():
    results = load_results(RESULTS_DIR)
    print(f"Loaded {len(results)} task result(s).")

    families = {}
    for r in results:
        families.setdefault(family_key(r), []).append(r)
    print(f"{len(families)} run famil{'y' if len(families) == 1 else 'ies'} found.\n")

    rollup, skipped = [], []
    for i, (key, fam) in enumerate(sorted(families.items()), 1):
        slug = family_slug(key)
        n_prec, n_targ, mode, phi, scales, ddd, qubits = key
        t0 = time.time()
        print(f"[{i}/{len(families)}] n_prec={n_prec} n_targ={n_targ} mode={mode} "
              f"phi={phi} scales={list(scales)} ddd={ddd} "
              f"qubits({len(qubits)})={list(qubits)[:4]}{'...' if len(qubits) > 4 else ''}",
              flush=True)
        try:
            summary, skip_reason = analyze_family(key, fam, families)
        except Exception as e:
            summary, skip_reason = None, f"unhandled exception: {e!r}"

        if summary is None:
            print(f"    SKIPPED: {skip_reason}")
            skipped.append({"key": list(key), "reason": skip_reason})
            continue

        filename = f"analysis_summary_{slug}.json"
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        rollup.append(rollup_row(key, summary, slug, filename))
        print(f"    wrote {filename}  ({time.time() - t0:.1f}s)"
              + ("" if summary["rem"] else f"  [REM: {summary['rem_skip_reason']}]"))

    print(f"\n{len(rollup)} family/families analysed, {len(skipped)} skipped.")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s['key']}: {s['reason']}")

    if rollup:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        write_table(rollup,
                   os.path.join(PLOTS_DIR, "mitigation_benefit_table.csv"),
                   os.path.join(PLOTS_DIR, "mitigation_benefit_table.json"))
        plot_rollup(rollup, os.path.join(PLOTS_DIR, "mitigation_benefit_all_families.png"))


if __name__ == "__main__":
    main()
