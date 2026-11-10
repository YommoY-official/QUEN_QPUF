#!/usr/bin/env python3
"""
run_all_families.py
====================
IonQ Forte-Enterprise-1 counterpart to
braket_direct/rigetti_cepheus_20260807/run_all_families.py: runs the
REM/ZNE/REM+ZNE/table/export pipeline over EVERY run family in job_results/
(this campaign has no per-run notebook to mirror -- job_results_qpe/'s
notebook doesn't exist here, so this file IS the analysis, not a port of a
notebook's cells).

Ported from the Cepheus driver near-verbatim for the pieces that have
nothing device-specific in them: family definition/grouping (load_results,
family_key, assign_batches, slot_tasks, pool_counts, pooled_lambda1), the
drift-anchor machinery (batch_anchor), analyze_family's per-family table
rows, rollup_row, and plot_rollup. What's DELIBERATELY DROPPED and what's
NEW is below.

DROPPED: all placement/routing/chiplet logic.
------------------------------------------------------------------------
Cepheus's driver carries a `resolve_clbit_to_phys` with a 3-tier fallback
(logged clbit_to_phys -> re-derive via place_and_route/transpile_for_rigetti
-> ASSUMED-SORTED) because Cepheus-1-108Q is a fixed lattice where SWAP
routing means the physical qubits a circuit lands on is not deterministic
from the logical circuit alone. Forte-Enterprise-1 is all-to-all: this
campaign's transpile_for_ionq passes no coupling_map, so qiskit's transpiler
uses a trivial (unpermuted) layout and `physical_qubits` in every logged
record IS the deterministic answer -- there is nothing to re-derive. So this
file reads `physical_qubits` straight off the record and only checks
consistency within a family, instead of resolving it.

NEW: a bit-ordering finding that changes which `mitig_analysis` convention
this file uses -- READ BEFORE TRUSTING ANY NUMBER OUT OF THIS FILE.
------------------------------------------------------------------------
Cepheus's checkRetrieve.py pulls RAW `AwsQuantumTask.result().measurement_
counts` + `result.measured_qubits` (ascending physical-qubit order), so its
driver MUST pass convention="braket" with clbit_to_phys/measured_qubits to
mitig_analysis.counts_to_m -- that machinery exists specifically for that
retrieval path.

This campaign's checkRetrieve.py retrieves counts differently: via
qiskit_braket_provider's `backend.retrieve_job(job_id).result().get_counts()`
(see checkRetrieve.py's retrieve_counts()). That code path
(BraketQuantumTask._result_from_circuit_task in qiskit_braket_provider,
installed version 0.11.1) takes Braket's raw measurement_counts string
(ascending physical-qubit order) and REVERSES it character-for-character --
with NO knowledge of which qiskit clbit each qubit was originally attached
to, because a task retrieved by ARN alone carries no source circuit.

That naive reversal happens to equal exact qiskit bitstring convention
(leftmost char = highest clbit index) FOR THIS CAMPAIGN'S CIRCUITS, because
build_qpe_circuit measures qubit (n_targ+k) into clbit k for every k --
ascending qubit index in lockstep with ascending clbit index -- and neither
transpile_for_ionq (no coupling_map -> TrivialLayout, verified empirically:
measured_physical_qubits() returns a strictly ascending list) nor fold()
(reattaches the ORIGINAL measurement instructions unchanged, confirmed by
reading error_mitig_ionq.fold/_reattach_measurements) breaks that ascending
correspondence. DDD is not part of this campaign (ddd_sequence is always
"none" for these families), so its idle-window pulse insertion -- which
does NOT touch measurement instructions either -- is not a threat to this
either, but has not been exercised.

CONSEQUENCE: this file uses mitig_analysis's DEFAULT convention="qiskit"
(CONV = {} below), not "braket", and does not need clbit_to_phys at all.

CAUTION: this reasoning is analytic, not an empirical check against a real
completed IonQ task result (none exists in this repo yet -- job_results/ is
still empty at the time this file was written). This is exactly the bit-
ordering bug class CLAUDE.md's known-bug list flags (item 4, the IonQ
endianness bug) and mitig_analysis.py's own module docstring warns about --
getting it wrong yields a plausible, wrong distribution, not an error. THE
VERIFIER SHOULD RE-CONFIRM THIS the moment the first real job_results/*.json
exists, e.g. by comparing a deterministic-phase family's raw p_correct
against the logged ideal_bin / est_fidelity, before trusting any number this
file produces on real hardware data.

NEW: validity gates on ZNE numbers (MITIGATION_BRIEF_IONQ.md Section 9.1's
folds-executed concern, made concrete).
------------------------------------------------------------------------
Cepheus's driver only gates a ZNE row's `valid` flag on extrapolate()'s own
`physical` in-[0,1] check. This campaign adds two more gates, computed in
analyze_family and reported as first-class `gates` fields in each family's
JSON summary (not just folded silently into a boolean):

  (a) SIGNAL ABOVE FLOOR: P(lambda=1) - 2^-n_prec > 5 * sqrt(p(1-p)/N).
      If the unmitigated point doesn't clear the incoherent floor with
      5-sigma confidence, there is no room for an extrapolation to be
      meaningful about, independent of how the curve looks.
  (b) MONOTONIC DECAY, >5-SIGMA SLOPE: bootstrapped linear-fit slope of raw
      P_correct(lambda) must be negative (decaying toward the floor as
      lambda increases the noise) with |mean/std| > 5, and the raw points
      must be monotonically non-increasing.

Gate (b) is the FOLDS-EXECUTED DIAGNOSTIC, not a "did ZNE help" verdict: a
flat lambda curve (slope not significant) means IonQ's server-side compiler
may have cancelled the unitary folds before they ever created a noise-
scaled circuit (brief Section 9.1) -- it does NOT mean "ZNE provided no
benefit". The `gate_b_monotonic_decay.note` field in each family's JSON
summary says which of these it is; do not collapse that distinction when
reporting results.

The exponential model is the HEADLINE extrapolation (ZNE_HEADLINE below),
not linear (Cepheus's choice): its free offset parameter can find the
2^-n_prec floor that a 2-parameter linear fit cannot. Richardson/linear are
still computed and reported as honest model-disagreement spread
(extrapolate()'s `model_spread`), but per this campaign's brief,
`model_spread` is NOT used to gate viability -- it goes to ~0 on a fully
dead (flat) circuit because every model agrees on "no change", which looks
like agreement but is actually the folds-not-executed failure mode above.

Run with the project's braket-arm environment (its Qiskit/Braket pins), not
the interpreter on PATH.
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

RESULTS_DIR = os.path.join(_MODDIR, "job_results")
PLOTS_DIR = os.path.join(_MODDIR, "plots")

BATCH_GAP_S = 30          # same batching heuristic as the Cepheus notebook/driver
N_BOOT = 500
REM_METHOD = "simplex"
ZNE_MODELS = ("richardson", "linear", "exponential")
ZNE_HEADLINE = "exponential"    # see module docstring: NOT Cepheus's "linear" choice
SEED = 0
SIGMA_GATE = 5.0                # significance threshold for both validity gates

# Bit-ordering convention for every mitig_analysis call in this file. See the
# module docstring's "NEW: a bit-ordering finding" section for why this is
# the default "qiskit" convention (CONV = {}) rather than Cepheus's
# convention="braket" + clbit_to_phys/measured_qubits.
CONV: dict = {}

# REM inverts a dense 2^n_prec x 2^n_prec confusion matrix via constrained
# NNLS (mitig_analysis.rem_correct, method="simplex"); Cepheus's driver
# measured this at ~O(dim^3) and set this cutoff from actual timings on its
# largest families (dim up to 32768). The largest n_prec in THIS campaign's
# approved family plan is 9 (B1, dim=512) -- several orders of magnitude
# below where that cost model bites -- so this cutoff is inherited for
# parity with Cepheus's driver (and as a guard against a future family
# exceeding the approved plan), not because it has been re-benchmarked here.
REM_MAX_NPREC = 12


# =========================================================================
# Grouping / loading -- adapted from Cepheus's notebook sections 2-3
# =========================================================================

def _when(r):
    """Submission time, always timezone-aware."""
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
    Same shape as Cepheus's family_key, with `physical_qubits` standing in
    for `measured_qubits`: this campaign's checkRetrieve.py/submit_ionq_
    mitigation.py never log a field named `measured_qubits` (Cepheus's
    field, populated from Braket's raw `result.measured_qubits`); the
    closest analog actually on the IonQ record is `physical_qubits`
    (sorted ascending, logged at submission time by measured_physical_
    qubits()). Included in the key for the same reason Cepheus includes
    placement: pooling counts from two different physical-qubit sets would
    silently corrupt the merged distribution.
    """
    return (r.get("n_prec"), r.get("n_targ"), r.get("state_mode"),
            r.get("phi"), tuple(r.get("zne_scales_all") or [r.get("zne_scale", 1)]),
            r.get("ddd_sequence") or "none",
            tuple(int(q) for q in (r.get("physical_qubits") or [])))


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
    """
    NOTE this differs from Cepheus's slot_tasks in one deliberate way: keyed
    by float(zne_scale), not int(zne_scale). Confirmed via this campaign's
    local smoke test (qiskit-aer, no hardware) that the int() cast silently
    COLLIDES lambda=1.5 into the lambda=1 bucket -- int(1.5) == 1 ==
    int(1.0) -- which corrupted every downstream number for a synthetic
    tier-1-style family (the 1.5 circuit's counts got mislabeled as the raw
    lambda=1 baseline). Cepheus's ZNE ladder folds with method="global"
    (odd integers only), where this collision can't occur; this campaign's
    tier-1 ladder is explicitly fractional (1, 1.5, 2, 3, per
    MITIGATION_BRIEF_IONQ.md / submit_ionq_mitigation.py's ZNE_SCALES), so
    the bug is live here. float() matches how every other scale-keyed dict
    in this file and in mitig_analysis.bootstrap_zne (which requires
    `1.0 in scales`) already treats it.
    """
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


def pooled_lambda1(records):
    batches, _ = assign_batches(records, BATCH_GAP_S)
    per = [slot_tasks(r)[0] for r in batches.values()]
    base = min({s for q in per for s in q}, default=None)
    if base is None:
        return None
    return pool_counts([q[base] for q in per if base in q])


def bootstrap_linear_slope(counts_by_scale, n_prec, ideal_bin, n_boot, seed, **conv):
    """
    Bootstrap CI on the RAW (no-REM) linear-fit slope of P_correct vs
    lambda -- gate (b)'s statistic (module docstring). Deliberately a
    separate, minimal bootstrap rather than an extension of
    mitig_analysis.bootstrap_zne's contract: this quantity (the slope
    itself, not an extrapolated value) is specific to this campaign's
    folds-executed diagnostic, not a general estimator.
    """
    rng = np.random.default_rng(seed)
    scales = sorted(counts_by_scale)
    slopes = []
    for _ in range(n_boot):
        vals = []
        for s in scales:
            r = ma.resample_counts(counts_by_scale[s], rng)
            p = ma.to_probabilities(ma.counts_to_m(r, n_prec, **conv), n_prec)
            vals.append(ma.p_correct(p, ideal_bin))
        coef = np.polyfit(scales, vals, 1)
        slopes.append(float(coef[0]))
    return ma.bootstrap_ci(slopes)


def bootstrap_zne_rem(counts_by_scale, cal0, cal1, n_prec, ideal_bin, n_meas,
                      n_boot, seed, models, method, **conv):
    """Thin wrapper so this driver's call sites match Cepheus's shape --
    the actual estimator now lives in mitig_analysis.bootstrap_zne_rem
    (promoted there per this campaign's piece 2, since this driver needs it
    too and a local copy would just drift)."""
    return ma.bootstrap_zne_rem(counts_by_scale, cal0, cal1, n_prec, ideal_bin,
                                n_meas, n_boot=n_boot, seed=seed, models=models,
                                method=method, **conv)


# =========================================================================
# Collision-safe export filename
# =========================================================================

def family_slug(key):
    """See Cepheus's family_slug for the full rationale -- ported unchanged
    except the qubit tuple's meaning (physical_qubits, see family_key)."""
    n_prec, n_targ, mode, phi, scales, ddd, qubits = key
    scales_s = "-".join(str(x).rstrip("0").rstrip(".") if "." in str(x) else str(int(x))
                        for x in scales)
    phi_s = "none" if phi is None else f"{phi:g}".replace(".", "p").replace("-", "m")
    khash = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
    return (f"nprec{n_prec}_ntarg{n_targ}_{mode}_phi{phi_s}"
            f"_scales{scales_s}_ddd{ddd}_q{khash}")


# =========================================================================
# Per-family analysis
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
        f"n_prec={n_prec} (dim={DIM}) exceeds REM_MAX_NPREC={REM_MAX_NPREC}"
    )

    ref = qpe_by_scale[SCALES[0]]
    MEASURED_QUBITS = sorted(int(q) for q in (ref.get("physical_qubits") or []))
    N_MEAS = len(MEASURED_QUBITS) or (n_targ + n_prec)

    for r in fam:
        mq = sorted(int(q) for q in (r.get("physical_qubits") or []))
        if mq != MEASURED_QUBITS:
            return None, (f"task {r['job_id']} touched physical qubits {mq}, "
                          f"expected {MEASURED_QUBITS} -- inconsistent within "
                          f"one family (should be impossible given family_key)")

    # See module docstring's bit-ordering finding: default "qiskit"
    # convention, no clbit_to_phys/measured_qubits needed for this campaign.
    counts_by_scale = {s: qpe_by_scale[s]["counts"] for s in SCALES}
    p_by_scale = {s: ma.to_probabilities(ma.counts_to_m(c, n_prec, **CONV), n_prec)
                  for s, c in counts_by_scale.items()}

    # IDEAL_BIN: known mode has a logged exact answer; haar mode has no
    # single ideal bin, so fall back to the argmax of the noiseless
    # statevector-simulated distribution (ideal_qpe_distribution, logged at
    # submission) -- same fallback Cepheus's driver uses. Also record the
    # ideal top-bin probability itself, since for haar instances that is
    # NOT 1.0 the way it is for an exact eigenstate, and reporting P_correct
    # without that context is misleading (a haar target state's QPE output
    # is inherently spread over several bins even with zero noise).
    if ref.get("ideal_bin") is not None:
        IDEAL_BIN = int(round(ref["ideal_bin"]))
        ideal_top_p = float((ref.get("ideal_distribution") or {}).get(str(IDEAL_BIN), 1.0))
    else:
        idist = ref.get("ideal_distribution") or {"0": 1.0}
        top_m, top_p = max(idist.items(), key=lambda kv: kv[1])
        IDEAL_BIN, ideal_top_p = int(top_m), float(top_p)

    ideal = np.zeros(DIM)
    for m, p in (ref.get("ideal_distribution") or {}).items():
        if int(m) < DIM:
            ideal[int(m)] = p
    if ideal.sum() == 0:
        ideal[IDEAL_BIN] = 1.0

    # -- batch drift anchors --------------------------------------------------
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

    # -- REM --------------------------------------------------------------
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

    # -- ZNE / ZNE+REM, with validity gates ------------------------------------
    zne_fit = zne_rem_fit = zne_boot = zne_rem_boot = gates = None
    if HAS_ZNE:
        vals_raw = [ma.p_correct(p_by_scale[s], IDEAL_BIN) for s in SCALES]
        zne_fit = ma.extrapolate(SCALES, vals_raw, models=ZNE_MODELS)
        zne_boot = ma.bootstrap_zne(counts_by_scale, n_prec, IDEAL_BIN,
                                    n_boot=N_BOOT, seed=SEED, models=ZNE_MODELS, **CONV)

        # -- gate (a): signal above the incoherent floor at lambda=1 -------
        floor = 2.0 ** (-n_prec)
        p1 = vals_raw[0]
        N1 = sum(counts_by_scale[SCALES[0]].values())
        margin = p1 - floor
        se5 = SIGMA_GATE * float(np.sqrt(max(p1 * (1 - p1), 0.0) / N1)) if N1 > 0 else float("inf")
        gate_a_pass = margin > se5

        # -- gate (b): monotonic decay toward the floor, >5-sigma slope ----
        slope_boot = bootstrap_linear_slope(counts_by_scale, n_prec, IDEAL_BIN,
                                            N_BOOT, SEED, **CONV)
        slope_mean, slope_std = slope_boot["mean"], slope_boot["std"]
        monotonic = all(vals_raw[i] >= vals_raw[i + 1] - 1e-12
                        for i in range(len(vals_raw) - 1))
        slope_sigma = (abs(slope_mean) / slope_std
                       if slope_mean is not None and slope_std else 0.0)
        gate_b_pass = bool(monotonic and slope_mean is not None
                           and slope_mean < 0 and slope_sigma > SIGMA_GATE)
        if gate_b_pass:
            gate_b_note = ("monotonic decay toward the floor, slope significant "
                          f"at {slope_sigma:.1f} sigma")
        elif slope_sigma <= SIGMA_GATE:
            gate_b_note = (
                "FLAT/insignificant lambda-curve slope -- this is the "
                "FOLDS-NOT-EXECUTED diagnostic (IonQ's server-side compiler may "
                "have cancelled the unitary folds before they scaled anything, "
                "MITIGATION_BRIEF_IONQ.md Section 9.1), NOT evidence that ZNE "
                "provided no benefit. Do not report a null ZNE finding from "
                "this alone.")
        else:
            gate_b_note = (
                "slope is significant but WRONG SIGN or the raw points are not "
                "monotonic -- P_correct is not decaying toward the floor as "
                "lambda increases; investigate before trusting either a "
                "positive or negative ZNE finding here.")

        gates = {
            "floor": floor, "p_lambda1": p1, "n_shots_lambda1": N1,
            "gate_a_signal_above_floor": {
                "margin": margin, "threshold_5sigma": se5, "pass": gate_a_pass},
            "gate_b_monotonic_decay": {
                "linear_slope_ci": slope_boot, "slope_sigma": slope_sigma,
                "monotonic_points": monotonic, "pass": gate_b_pass,
                "note": gate_b_note},
        }
        zne_gates_pass = bool(gate_a_pass and gate_b_pass)

        if HAS_REM:
            vals_rem = [ma.p_correct(p_rem_by_scale[s], IDEAL_BIN) for s in SCALES]
            zne_rem_fit = ma.extrapolate(SCALES, vals_rem, models=ZNE_MODELS)
            zne_rem_boot = bootstrap_zne_rem(
                counts_by_scale, cal_by_state["cal0"]["counts"],
                cal_by_state["cal1"]["counts"], n_prec, IDEAL_BIN, N_MEAS,
                N_BOOT, SEED, ZNE_MODELS, REM_METHOD, **CONV)
    else:
        zne_gates_pass = False

    # -- DDD siblings -- every family in this campaign is ddd=none so far, so
    # this will always find exactly one arm and report nothing to compare
    # (no DDD arm in this campaign yet -- see error_mitig_ionq.py's `circ.
    # id()` / IONQ_SUPPORTED_OPS gap note for why that stays true for now).
    siblings = {k[5]: v for k, v in families.items()
               if k[:5] == key[:5] and k[6] == key[6]}
    ddd_deltas = None
    if len(siblings) >= 2:
        baseline_seq = "none" if "none" in siblings else sorted(siblings)[0]
        base_rec = pooled_lambda1(siblings[baseline_seq])
        base_mq = sorted(int(q) for q in (base_rec.get("physical_qubits") or []))
        ddd_deltas = {}
        for seq in sorted(siblings):
            if seq == baseline_seq:
                continue
            rec = pooled_lambda1(siblings[seq])
            if sorted(int(q) for q in (rec.get("physical_qubits") or [])) != base_mq:
                continue
            ddd_deltas[seq] = ma.bootstrap_delta_unpaired(
                base_rec["counts"], rec["counts"], n_prec, IDEAL_BIN,
                n_boot=N_BOOT, seed=SEED, **CONV)

    # -- table ---------------------------------------------------------------
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
        rows.append({
            "condition": f"ZNE ({ZNE_HEADLINE})", "p_correct": v,
            "tvd": None, "phase_err": None, "ci": zne_boot["ci"][ZNE_HEADLINE],
            "delta_ci": zne_boot["delta_ci"][ZNE_HEADLINE],
            # gated on extrapolate()'s physicality AND both campaign-specific
            # gates (a)/(b) above -- see module docstring. A physical-looking
            # number from a flat, gate-(b)-failing curve is NOT reported valid.
            "valid": v is not None and bool(zne_fit.get("physical")) and zne_gates_pass,
        })
    if HAS_ZNE and HAS_REM:
        v = zne_rem_fit.get(ZNE_HEADLINE)
        rows.append({
            "condition": f"ZNE+REM ({ZNE_HEADLINE})", "p_correct": v,
            "tvd": None, "phase_err": None, "ci": zne_rem_boot["ci"][ZNE_HEADLINE],
            "delta_ci": zne_rem_boot["delta_ci"][ZNE_HEADLINE],
            "valid": v is not None and bool(zne_rem_fit.get("physical")) and zne_gates_pass,
        })

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "synthetic": False,
        "run": {"n_prec": n_prec, "n_targ": n_targ, "state_mode": state_mode,
                "phi": phi, "ideal_bin": IDEAL_BIN, "ideal_top_bin_p": ideal_top_p,
                "scales": SCALES, "shots_per_task": sum(counts_by_scale[SCALES[0]].values())},
        "qubits": {"physical_qubits": MEASURED_QUBITS,
                   "convention": "qiskit (see module docstring)"},
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
        "zne": None if not HAS_ZNE else {"fit": zne_fit, "bootstrap": zne_boot,
                                         "gates": gates},
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
    """
    Column set matches braket_direct/rigetti_cepheus_20260807/run_all_
    families.py's rollup_row EXACTLY (same keys, same order) so the two
    campaigns' CSVs can be concatenated/compared directly -- do not add or
    rename columns here even though this campaign's per-family `table` rows
    carry an extra `delta_ci` field; that field is available in the JSON
    summary for anyone who wants it, deliberately not promoted to this row.
    """
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
    """Identical to Cepheus's plot_rollup -- see that docstring for the
    hollow-marker-means-unphysical convention."""
    C = {"rem": "#1baf7a", "zne": "#eda100", "zne_rem": "#e87ba4"}
    MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}
    techniques = [("delta_rem", "delta_rem_valid", "rem_available", "rem", "REM"),
                  ("delta_zne", "delta_zne_valid", "zne_available", "zne",
                   f"ZNE ({ZNE_HEADLINE})"),
                  ("delta_zne_rem", "delta_zne_rem_valid", "zne_rem_available", "zne_rem",
                   f"ZNE+REM ({ZNE_HEADLINE})")]
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
    ax.set_title("REM / ZNE / ZNE+REM benefit vs n_prec, IonQ Forte-Enterprise-1\n"
                 "(hollow marker = extrapolation unphysical OR failed a validity gate)")
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
