#!/usr/bin/env python3
"""
mitig_analysis.py
=================
The estimator half of the IonQ mitigation study: everything that turns
returned COUNTS into a number with an error bar. Near-verbatim copy of
braket_direct/rigetti_cepheus_20260807/mitig_analysis.py -- this half of the
pipeline (bitstring decoding, REM inversion, ZNE extrapolation, bootstrap
CIs) has nothing device-specific in it. What changed for IonQ is confined to
the pre-flight `validate()` harness at the bottom: the noise model's basis
gates (rz/rx/rxx instead of rz/rx/cz) and its fidelity/readout defaults
(F1Q/F2Q/p_readout, calibration-derived from device_properties_raw.json,
fetched live 2026-08-28 -- see validate()'s docstring -- not Rigetti's
placeholders).

`error_mitig_ionq.py` builds circuits. This file consumes results. The split
is along a real seam -- REM and ZNE each have a circuit-building half (there)
and a reduce half (here), and only together do they produce a mitigated
value.

Contents
--------
  bitstrings      counts -> distribution over the measured integer m
  observables     P_correct (primary), TVD and phase error (secondary)
  REM             confusion matrices, condition number, two inversions
  ZNE             Richardson / linear / exponential extrapolation to lambda=0
  bootstrap       CIs, including the PAIRED case that REM requires
  validation      noisy simulation of every technique, before spending shots

Run the pre-flight validation with:
    python mitig_analysis.py --validate


Bit ordering -- the trap this project keeps stepping on
-------------------------------------------------------
Our QPE writes precision qubit k into classical bit k, and precision qubit 0
is the MSB, so the estimate is

    m = sum_k c[k] * 2^(n_prec-1-k)

But nothing hands you c[k] directly:

  - qiskit `get_counts` returns the string c[n-1] ... c[0], leftmost is the
    HIGHEST clbit index.
  - Braket `measurement_counts` returns one character per MEASURED QUBIT,
    ordered by ascending physical qubit index -- which after folding/DDD
    rebuilding a circuit has no fixed relation to clbit order at all.

So the Braket path needs both the clbit->physical map (from
`measured_physical_qubits(circuit)`, logged at submission) and the result's
`measured_qubits` list. `counts_to_m` takes them explicitly rather than
guessing. Getting this wrong yields a plausible, wrong distribution -- the
same failure mode as the inverse-QFT endianness bug (CLAUDE.md known-bug
item 4; fixed in ionq_mitigation_common.build_qpe_stage, confirmed via
docs/LOG.md 2026-08-28).
"""

import argparse
import math

import numpy as np
from scipy.optimize import nnls, curve_fit


# =========================================================================
# Counts -> distribution over the measured integer
# =========================================================================

def counts_to_m(counts: dict[str, int], n_prec: int,
                convention: str = "qiskit",
                clbit_to_phys: list[int] | None = None,
                measured_qubits: list[int] | None = None) -> dict[int, int]:
    """
    Re-key measurement counts by the QPE integer m.

    convention:
      "qiskit" -- bitstring is c[n-1]...c[0] (leftmost = highest clbit).
      "braket" -- one char per measured qubit, ascending physical index.
                  Requires clbit_to_phys (from measured_physical_qubits at
                  submission time) and measured_qubits (from the result).

    Only the first n_prec classical bits are used, so this also works on the
    two-stage QPUF's 2*n_prec register (pass n_prec to read stage 1).
    """
    if convention == "qiskit":
        char_of_clbit = None
    elif convention == "braket":
        if clbit_to_phys is None or measured_qubits is None:
            raise ValueError("braket convention needs clbit_to_phys and "
                             "measured_qubits")
        order = list(measured_qubits)
        char_of_clbit = [order.index(clbit_to_phys[k]) for k in range(n_prec)]
    else:
        raise ValueError(f"unknown convention {convention!r}")

    out: dict[int, int] = {}
    for bits, n in counts.items():
        s = bits.replace(" ", "")
        m = 0
        for k in range(n_prec):
            pos = (len(s) - 1 - k) if char_of_clbit is None else char_of_clbit[k]
            if s[pos] == "1":
                m += 1 << (n_prec - 1 - k)
        out[m] = out.get(m, 0) + n
    return out


def to_probabilities(m_counts: dict[int, int], n_prec: int) -> np.ndarray:
    """Dense probability vector over m in [0, 2^n_prec)."""
    p = np.zeros(2 ** n_prec, dtype=float)
    total = sum(m_counts.values())
    if total == 0:
        return p
    for m, n in m_counts.items():
        p[m] = n / total
    return p


# =========================================================================
# Observables
# =========================================================================

def p_correct(p: np.ndarray, ideal_bin: int) -> float:
    """
    PRIMARY OBSERVABLE: probability of the correct bitstring.

    This is a genuine expectation value -- of the projector |j><j| -- which
    is why ZNE may legitimately be extrapolated over it: expectation values
    are linear in the state, so P_correct(lambda) is the smooth function of
    noise strength that extrapolation assumes.
    """
    return float(p[ideal_bin])


def tvd(p: np.ndarray, ideal: np.ndarray) -> float:
    """
    Total variation distance to the ideal distribution.

    SECONDARY / consistency check only. TVD is NOT linear in the state, so
    extrapolating TVD to zero noise is not theoretically justified. Report
    it; do not build the argument on it.
    """
    n = max(p.size, ideal.size)
    a, b = np.zeros(n), np.zeros(n)
    a[:p.size], b[:ideal.size] = p, ideal
    return float(0.5 * np.abs(a - b).sum())


def phase_error(p: np.ndarray, phi_true: float, n_prec: int) -> float:
    """|phi_hat - phi| from the argmax bin, wrapped onto the circle."""
    m = int(np.argmax(p))
    d = abs(m / 2 ** n_prec - phi_true)
    return float(min(d, 1.0 - d))


# =========================================================================
# REM -- readout error mitigation (the reduce half)
# =========================================================================

def confusion_matrices(cal0_counts: dict[str, int], cal1_counts: dict[str, int],
                       n_qubits: int, convention: str = "qiskit",
                       clbit_to_phys: list[int] | None = None,
                       measured_qubits: list[int] | None = None
                       ) -> list[np.ndarray]:
    """
    Per-qubit 2x2 confusion matrices A_j from the |0...0> and |1...1>
    calibration circuits.

        A_j = [[p(0|0), p(0|1)],
               [p(1|0), p(1|1)]]

    Column j of A_j is the readout distribution given the qubit was actually
    in |j>, so A_j is column-stochastic and p_noisy = A p_true.
    """
    def marginals(counts):
        ones = np.zeros(n_qubits)
        total = sum(counts.values())
        if total == 0:
            raise ValueError("empty calibration counts")
        for bits, n in counts.items():
            s = bits.replace(" ", "")
            for k in range(n_qubits):
                if convention == "qiskit":
                    pos = len(s) - 1 - k
                else:
                    pos = list(measured_qubits).index(clbit_to_phys[k])
                if s[pos] == "1":
                    ones[k] += n
        return ones / total

    p1_given_0 = marginals(cal0_counts)     # false excitation
    p1_given_1 = marginals(cal1_counts)     # correct readout of |1>

    mats = []
    for k in range(n_qubits):
        a = np.array([[1 - p1_given_0[k], 1 - p1_given_1[k]],
                      [p1_given_0[k],     p1_given_1[k]]], dtype=float)
        mats.append(a)
    return mats


def tensored_confusion(mats: list[np.ndarray]) -> np.ndarray:
    """
    Full A = A_0 (x) A_1 (x) ... (x) A_{n-1}.

    Kronecker order matches our integer convention: index
    m = sum_k b_k 2^(n-1-k) puts clbit 0 in the most significant position,
    and np.kron(A, B) likewise makes A's index the more significant one. So
    the matrices must be passed in clbit order, MSB first.
    """
    full = np.array([[1.0]])
    for a in mats:
        full = np.kron(full, a)
    return full


def condition_number(a: np.ndarray) -> float:
    """
    Condition number of the confusion matrix.

    A^-1 amplifies shot noise by roughly this factor -- it is the single
    number that says how much variance REM is buying you in exchange for
    removing readout bias. Log it every run.
    """
    return float(np.linalg.cond(a))


def rem_correct(p_noisy: np.ndarray, a: np.ndarray,
                method: str = "simplex") -> np.ndarray:
    """
    Invert the readout channel.

    method:
      "naive"   -- p = A^-1 p_noisy, clip negatives, renormalize. BIASED:
                   clipping is a nonlinear operation that systematically
                   pushes weight into the surviving bins. Included as the
                   baseline everyone reports.
      "simplex" -- constrained least squares: minimise ||A p - p_noisy||^2
                   subject to p >= 0 and sum(p) = 1. DEFAULT: stays a
                   probability distribution instead of producing negative
                   quasi-probabilities.
    """
    if method == "naive":
        p = np.linalg.solve(a, p_noisy) if a.shape[0] == a.shape[1] \
            else np.linalg.lstsq(a, p_noisy, rcond=None)[0]
        p = np.clip(p, 0.0, None)
        s = p.sum()
        return p / s if s > 0 else p

    if method != "simplex":
        raise ValueError(f"unknown REM method {method!r}")

    n = a.shape[1]
    w = 1e3 * max(1.0, float(np.abs(a).max()))     # weight on sum(p) = 1
    a_aug = np.vstack([a, w * np.ones((1, n))])
    b_aug = np.concatenate([p_noisy, [w]])
    p, _ = nnls(a_aug, b_aug)
    s = p.sum()
    return p / s if s > 0 else p


# =========================================================================
# ZNE -- extrapolation to zero noise (the reduce half)
# =========================================================================

def _richardson(scales, values):
    """
    Exact polynomial interpolation through every point, evaluated at 0.
    Equivalent to Lagrange weights. Uses all the information but is a
    degree-(N-1) interpolant, so it AMPLIFIES VARIANCE badly -- always
    report it next to linear and exponential rather than alone.
    """
    x, y = np.asarray(scales, float), np.asarray(values, float)
    total = 0.0
    for i in range(len(x)):
        term = y[i]
        for j in range(len(x)):
            if i != j:
                term *= (0.0 - x[j]) / (x[i] - x[j])
        total += term
    return float(total)


def _exp_model(x, a, b, c):
    return a + b * np.exp(-c * x)


def extrapolate(scales, values, models=("richardson", "linear", "exponential")
                ) -> dict:
    """
    Extrapolate P_correct(lambda) to lambda = 0 under several models.

    Reporting all of them is the point: the SPREAD across models is an
    honest part of the uncertainty, and if Richardson disagrees wildly with
    linear that is itself the finding (the noise is not in the regime where
    extrapolation is valid).
    """
    x, y = np.asarray(scales, float), np.asarray(values, float)
    out: dict = {}

    if "richardson" in models and len(x) >= 2:
        try:
            out["richardson"] = _richardson(x, y)
        except Exception as e:
            out["richardson"] = None
            out["richardson_error"] = str(e)

    if "linear" in models and len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        out["linear"] = float(np.polyval(coef, 0.0))
        out["linear_slope"] = float(coef[0])

    if "poly2" in models and len(x) >= 3:
        out["poly2"] = float(np.polyval(np.polyfit(x, y, 2), 0.0))

    if "exponential" in models and len(x) >= 3:
        try:
            span = max(y.max() - y.min(), 1e-9)
            p0 = (max(y.min() - 0.05, 0.0), span, 0.5)
            popt, _ = curve_fit(_exp_model, x, y, p0=p0, maxfev=20000)
            out["exponential"] = float(_exp_model(0.0, *popt))
            out["exponential_params"] = [float(v) for v in popt]
        except Exception as e:
            out["exponential"] = None
            out["exponential_error"] = str(e)

    vals = [v for k, v in out.items()
            if k in ("richardson", "linear", "poly2", "exponential")
            and isinstance(v, float)]
    if vals:
        out["model_spread"] = float(max(vals) - min(vals))

    # PHYSICALITY. P_correct is a probability: an extrapolation landing
    # outside [0, 1] is not a slightly-off estimate, it is proof that the
    # extrapolation is invalid. Surface it loudly rather than reporting an
    # impossible number as a result.
    #
    # IonQ-specific reminder (MITIGATION_BRIEF_IONQ.md Section 9.1): a
    # zero-asymptote read of an out-of-range fit is not the only failure
    # mode to watch for here. If IonQ's compiler cancels the folds, every
    # lambda executes as lambda=1 and the curve is FLAT -- which passes the
    # physicality check trivially (all points equal, extrapolation lands
    # exactly on the lambda=1 value) while telling you nothing was
    # mitigated. `physical=True` and `unphysical=[]` do NOT mean the run is
    # trustworthy; check the ZNE curve is monotonically decaying toward the
    # incoherent floor 2^-n_prec before reporting a ZNE number from real
    # IonQ hardware under this campaign.
    out["unphysical"] = sorted(
        k for k in ("richardson", "linear", "poly2", "exponential")
        if isinstance(out.get(k), float) and not (0.0 <= out[k] <= 1.0))
    out["physical"] = not out["unphysical"]
    return out


# =========================================================================
# Bootstrap
# =========================================================================

def resample_counts(counts: dict, rng: np.random.Generator) -> dict:
    """One multinomial resample of a counts dict, preserving total shots."""
    keys = list(counts)
    n = np.array([counts[k] for k in keys], dtype=float)
    total = int(n.sum())
    draw = rng.multinomial(total, n / total)
    return {k: int(v) for k, v in zip(keys, draw) if v > 0}


def bootstrap_ci(samples, alpha: float = 0.05) -> dict:
    s = np.asarray([v for v in samples if v is not None and np.isfinite(v)])
    if s.size == 0:
        return {"mean": None, "lo": None, "hi": None, "std": None, "n": 0}
    return {"mean": float(s.mean()), "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "lo": float(np.quantile(s, alpha / 2)),
            "hi": float(np.quantile(s, 1 - alpha / 2)), "n": int(s.size)}


def bootstrap_delta_unpaired(counts_a: dict, counts_b: dict, n_prec: int,
                             ideal_bin: int, n_boot: int = 500,
                             seed: int = 0, **conv) -> dict:
    """
    CI on P_correct(b) - P_correct(a) for two SEPARATE tasks -- e.g. DDD-on
    vs DDD-off, which are different circuits run at different times.

    Resampled independently: no shared randomness to exploit, so this
    interval is genuinely wider than the paired REM case. Captures shot
    noise ONLY -- on hardware add a drift component from a repeated anchor
    condition; two on-demand tasks minutes (or, queue-dependent, hours)
    apart are not identically distributed.
    """
    rng = np.random.default_rng(seed)
    deltas, a_s, b_s = [], [], []
    for _ in range(n_boot):
        pa = to_probabilities(counts_to_m(resample_counts(counts_a, rng),
                                          n_prec, **conv), n_prec)
        pb = to_probabilities(counts_to_m(resample_counts(counts_b, rng),
                                          n_prec, **conv), n_prec)
        va, vb = p_correct(pa, ideal_bin), p_correct(pb, ideal_bin)
        a_s.append(va)
        b_s.append(vb)
        deltas.append(vb - va)
    ci = bootstrap_ci(deltas)
    ci["significant"] = bool(ci["lo"] is not None
                             and (ci["lo"] > 0 or ci["hi"] < 0))
    return {"a": bootstrap_ci(a_s), "b": bootstrap_ci(b_s), "delta": ci}


def bootstrap_rem_paired(raw_counts: dict, cal0: dict, cal1: dict,
                         n_prec: int, ideal_bin: int, n_qubits: int,
                         n_boot: int = 500, method: str = "simplex",
                         seed: int = 0, **conv) -> dict:
    """
    PAIRED bootstrap for the REM effect.

    REM-on and REM-off are computed from the SAME counts -- REM is a
    deterministic map, not a second experiment. So the difference must be
    resampled jointly: draw one resample, apply BOTH estimators to it, take
    the difference. Combining two independently-computed error bars in
    quadrature would badly overstate the uncertainty on Delta.

    The calibration counts are resampled too, so confusion-matrix
    uncertainty propagates into the interval -- that is the variance half of
    REM's bias/variance trade and it belongs in the reported number.
    """
    rng = np.random.default_rng(seed)
    raw_s, rem_s, delta_s, conds = [], [], [], []

    for _ in range(n_boot):
        r  = resample_counts(raw_counts, rng)
        c0 = resample_counts(cal0, rng)
        c1 = resample_counts(cal1, rng)

        p_raw = to_probabilities(counts_to_m(r, n_prec, **conv), n_prec)
        mats  = confusion_matrices(c0, c1, n_qubits, **conv)
        a     = tensored_confusion(mats[:n_prec])
        p_rem = rem_correct(p_raw, a, method=method)

        v_raw, v_rem = p_correct(p_raw, ideal_bin), p_correct(p_rem, ideal_bin)
        raw_s.append(v_raw)
        rem_s.append(v_rem)
        delta_s.append(v_rem - v_raw)
        conds.append(condition_number(a))

    return {"raw": bootstrap_ci(raw_s), "rem": bootstrap_ci(rem_s),
            "delta": bootstrap_ci(delta_s),
            "condition_number": bootstrap_ci(conds)}


def bootstrap_zne(counts_by_scale: dict[float, dict], n_prec: int,
                  ideal_bin: int, n_boot: int = 500, seed: int = 0,
                  models=("richardson", "linear", "exponential"),
                  **conv) -> dict:
    """
    Bootstrap the WHOLE ZNE fit, not each lambda separately.

    The extrapolation's lever arm inflates variance far beyond any single
    point's standard error, so a CI built from per-lambda errors understates
    the real uncertainty. Resample every lambda's counts, refit, extrapolate,
    and take the spread of the extrapolated values.

    Also returns a CI on Delta = P_extrap - P_correct(lambda=1) -- the number
    the campaign actually reports ("did ZNE help"), not just a CI on the
    extrapolant by itself.

    CRITICAL: lambda=1's counts are simultaneously the ZNE ladder's first
    rung AND the unmitigated baseline -- they are literally the same counts.
    So each draw resamples lambda=1 EXACTLY ONCE and uses that one resample
    on both sides of the subtraction (as `vals[base_idx]`, already inside the
    ladder loop below). Resampling lambda=1 a second time, independently, for
    the "baseline" side (what bootstrap_delta_unpaired does for genuinely
    separate tasks) would treat it as an independent second experiment and
    inflate the Delta interval -- see bootstrap_delta_unpaired's docstring
    for that case; this is deliberately NOT that case.
    """
    rng = np.random.default_rng(seed)
    scales = sorted(counts_by_scale)
    if 1.0 not in scales:
        raise ValueError("bootstrap_zne needs a lambda=1.0 entry to define "
                         "the Delta baseline")
    base_idx = scales.index(1.0)
    per_model: dict[str, list] = {m: [] for m in models}
    delta_per_model: dict[str, list] = {m: [] for m in models}

    for _ in range(n_boot):
        vals = []
        for s in scales:
            r = resample_counts(counts_by_scale[s], rng)
            p = to_probabilities(counts_to_m(r, n_prec, **conv), n_prec)
            vals.append(p_correct(p, ideal_bin))
        fit = extrapolate(scales, vals, models=models)
        baseline = vals[base_idx]           # SAME resample used below, not a fresh one
        for m in models:
            per_model[m].append(fit.get(m))
            v = fit.get(m)
            delta_per_model[m].append(None if v is None else v - baseline)

    baseline_point = p_correct(
        to_probabilities(counts_to_m(counts_by_scale[1.0], n_prec, **conv), n_prec),
        ideal_bin)
    point = extrapolate(
        scales,
        [p_correct(to_probabilities(counts_to_m(counts_by_scale[s], n_prec, **conv),
                                    n_prec), ideal_bin) for s in scales],
        models=models)
    delta_point = {m: (None if point.get(m) is None else point[m] - baseline_point)
                   for m in models}
    return {"point_estimate": point,
            "ci": {m: bootstrap_ci(v) for m, v in per_model.items()},
            "delta_point_estimate": delta_point,
            "delta_ci": {m: bootstrap_ci(v) for m, v in delta_per_model.items()}}


def bootstrap_zne_rem(counts_by_scale: dict[float, dict], cal0: dict, cal1: dict,
                      n_prec: int, ideal_bin: int, n_meas: int,
                      n_boot: int = 500, seed: int = 0,
                      models=("richardson", "linear", "exponential"),
                      method: str = "simplex", **conv) -> dict:
    """
    bootstrap_zne, but with REM applied to every lambda before the fit.

    Ported from braket_direct/rigetti_cepheus_20260807/run_all_families.py's
    local `bootstrap_zne_rem` helper (~line 217) -- promoted to a proper
    module-level function here since this campaign's driver needs it too and
    there is no reason for two copies to drift.

    Per draw: resample cal0/cal1 ONCE and build one confusion matrix A from
    that resample; resample every lambda's counts ONCE; REM-correct each
    lambda with that same A; fit; extrapolate. The lambda=1 resample is
    reused for both the ladder's first rung and the Delta baseline below --
    same single-resample discipline as bootstrap_zne above, and for the same
    reason: it is the same counts, not a second experiment.

    Delta baseline is the RAW (no REM) p_correct at lambda=1, matching
    run_all_families.py's rollup_row -- `delta_zne_rem` there is defined
    against the fully unmitigated "raw (lambda=1)" row, not the REM-only
    row, so ZNE+REM's reported benefit is against the same reference every
    other technique in the table is compared to.
    """
    rng = np.random.default_rng(seed)
    scales = sorted(counts_by_scale)
    if 1.0 not in scales:
        raise ValueError("bootstrap_zne_rem needs a lambda=1.0 entry to "
                         "define the Delta baseline")
    base_idx = scales.index(1.0)
    per_model: dict[str, list] = {m: [] for m in models}
    delta_per_model: dict[str, list] = {m: [] for m in models}

    for _ in range(n_boot):
        a = tensored_confusion(
            confusion_matrices(resample_counts(cal0, rng), resample_counts(cal1, rng),
                               n_meas, **conv)[:n_prec])
        vals, baseline_raw = [], None
        for i, s in enumerate(scales):
            r = resample_counts(counts_by_scale[s], rng)
            p_raw = to_probabilities(counts_to_m(r, n_prec, **conv), n_prec)
            vals.append(p_correct(rem_correct(p_raw, a, method=method), ideal_bin))
            if i == base_idx:
                baseline_raw = p_correct(p_raw, ideal_bin)     # same resample, no REM
        fit = extrapolate(scales, vals, models=models)
        for m in models:
            per_model[m].append(fit.get(m))
            v = fit.get(m)
            delta_per_model[m].append(None if v is None else v - baseline_raw)

    a_point = tensored_confusion(confusion_matrices(cal0, cal1, n_meas, **conv)[:n_prec])
    vals_point, baseline_point_raw = [], None
    for i, s in enumerate(scales):
        p_raw = to_probabilities(counts_to_m(counts_by_scale[s], n_prec, **conv), n_prec)
        vals_point.append(p_correct(rem_correct(p_raw, a_point, method=method), ideal_bin))
        if i == base_idx:
            baseline_point_raw = p_correct(p_raw, ideal_bin)
    point = extrapolate(scales, vals_point, models=models)
    delta_point = {m: (None if point.get(m) is None else point[m] - baseline_point_raw)
                   for m in models}

    return {"point_estimate": point,
            "ci": {m: bootstrap_ci(v) for m, v in per_model.items()},
            "delta_point_estimate": delta_point,
            "delta_ci": {m: bootstrap_ci(v) for m, v in delta_per_model.items()}}


# =========================================================================
# Pre-flight validation in noisy simulation
# =========================================================================
#
# The rule this enforces: if a technique does not help in simulation with a
# realistic noise model, it will not help on hardware, and you have just
# saved shots and queue time.
#
# NOISE MODEL DESIGN -- this part matters more than it looks.
#
# A plain Markovian noise model (depolarizing + thermal relaxation) CANNOT
# show any DDD benefit, and running one would produce a false negative.
# Markovian noise is memoryless: splitting an idle window into two halves
# gives exactly the same total decay, so no pulse sequence can refocus it.
# Dynamical decoupling works against SLOW, CORRELATED noise -- a static
# frequency offset that accumulates a coherent phase, which the pulses
# invert and cancel. On a trapped ion the physically plausible slow-drift
# sources are magnetic-field and laser-phase drift (see
# MITIGATION_BRIEF_IONQ.md Section 9.2) -- the coherent-detuning proxy below
# is the same MODEL Rigetti's validation uses, not a measurement of IonQ's
# actual noise spectrum, which is unmeasured by this project.
#
# So the model has two parts:
#   (1) depolarizing on rx/rxx + readout error   -- what REM and ZNE address
#   (2) a COHERENT static detuning on idle qubits, injected as an rz whose
#       angle is proportional to the idle duration -- what DDD addresses
# Only with (2) present is a DDD validation meaningful.

def build_noise_model(f_1q: float, f_2q: float, p_readout: float):
    from qiskit_aer.noise import (NoiseModel, depolarizing_error, ReadoutError)
    nm = NoiseModel(basis_gates=["rz", "rx", "rxx", "id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(1 - f_1q, 1), ["rx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(1 - f_2q, 2), ["rxx"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]))
    return nm


def inject_coherent_idle_error(circ, em, rate_rad_per_s: float):
    """
    Insert rz(rate * idle_duration) into every idle window: a static
    frequency offset accumulating coherent phase.

    Inserted AFTER any DD pulses, so a decoupled window is split into
    sub-windows whose phases the pulses invert and cancel, while an
    undecoupled window accumulates the whole phase at once. That asymmetry
    is exactly what DDD is supposed to exploit, and it is why this term has
    to be here for the DDD arm to mean anything.
    """
    from qiskit import QuantumCircuit
    layer_ops, layer_dur, busy = em._schedule(circ)
    plan: dict[int, list[int]] = {}
    for q, mask in busy.items():
        active = [i for i, b in enumerate(mask) if b]
        if len(active) < 2:
            continue
        for start, stop in em._idle_runs(mask, active[0], active[-1]):
            window = sum(layer_dur[start:stop])
            if window <= 0:
                continue
            mid = start + (stop - start) // 2
            plan.setdefault(mid, []).append((q, rate_rad_per_s * window))

    out = QuantumCircuit(*circ.qregs, *circ.cregs)
    out.global_phase = circ.global_phase
    for i, ops in enumerate(layer_ops):
        for q, theta in plan.get(i, []):
            out.rz(theta, q)
        for op, qargs, cargs in ops:
            out.append(op, qargs, cargs)
    return out


def _run(circ, nm, shots, seed):
    from qiskit_aer import AerSimulator
    # optimization_level=0: any higher and the transpiler cancels the ZNE
    # folds and the DD pulses before the simulator ever sees them, which
    # would silently reproduce the exact failure this whole validation step
    # exists to catch.
    from qiskit import transpile as _tr
    sim = AerSimulator(noise_model=nm, basis_gates=["rz", "rx", "rxx", "id"])
    tc = _tr(circ, sim, optimization_level=0)
    return sim.run(tc, shots=shots, seed_simulator=seed).result().get_counts()


def validate(n_prec: int = 4, phi: float = 1 / 8, shots: int = 20000,
             f_1q: float = 0.9998, f_2q: float = 0.9882, p_readout: float = 0.006,
             detuning_rad_per_s: float = 2.0e5, n_boot: int = 200,
             seed: int = 7) -> dict:
    """
    Run every technique in noisy simulation and report whether it helps.

    f_1q/f_2q/p_readout are now CALIBRATION-DERIVED, not placeholders:
    device_properties_raw.json (fetched live from Forte-Enterprise-1,
    2026-08-28) gives provider.fidelity.{1Q,2Q,spam}.mean = 0.9998 / 0.9882 /
    0.994 -- matching ionq_mitigation_common.F1Q/F2Q/SPAM_FIDELITY.
    p_readout = 1 - SPAM_FIDELITY = 0.006 is this validation harness's first
    use of a real per-qubit readout-error rate for this device -- SPAM is
    exactly the channel REM's confusion matrix corrects, so this number
    matters more here than F1Q/F2Q do. detuning_rad_per_s remains a
    placeholder: no IonQ drift spectral density is known to this project
    (MITIGATION_BRIEF_IONQ.md Section 9.2).
    """
    from qiskit import transpile
    from ionq_mitigation_common import (known_phase_unitary, build_qpe_circuit,
                                        IONQ_BASIS)
    from error_mitig_ionq import ErrorMitigation

    em = ErrorMitigation()
    u  = known_phase_unitary(1, phi)
    qc = transpile(build_qpe_circuit(n_prec, 1, u, eigenstate=True),
                   basis_gates=IONQ_BASIS, optimization_level=1)
    ideal_bin = int(round(phi * 2 ** n_prec))
    nm = build_noise_model(f_1q, f_2q, p_readout)
    n_meas = qc.num_clbits

    print("=" * 76)
    print("PRE-FLIGHT VALIDATION IN NOISY SIMULATION (IonQ Forte-Enterprise-1)")
    print("=" * 76)
    print(f"  QPE n_prec={n_prec}, phi={phi:g} -> ideal bin {ideal_bin} "
          f"(noiseless p=1.0)")
    print(f"  noise: F1Q={f_1q} F2Q={f_2q} readout_err={p_readout} "
          f"detuning={detuning_rad_per_s:.1e} rad/s")
    print(f"  shots/circuit={shots:,}   bootstrap={n_boot}")
    print()

    def pc(counts):
        return p_correct(to_probabilities(counts_to_m(counts, n_prec), n_prec),
                         ideal_bin)

    results: dict = {}

    # ---- baseline + DDD arms ------------------------------------------------
    base_noisy = inject_coherent_idle_error(qc, em, detuning_rad_per_s)
    counts_base = _run(base_noisy, nm, shots, seed)
    results["none"] = pc(counts_base)

    print(f"{'condition':<24}{'P_correct':>10}{'delta vs base':>15}")
    print("-" * 76)
    print(f"{'no mitigation':<24}{results['none']:>10.4f}{'--':>15}")

    ddd_counts = {}
    for sequence in ("XX", "YY", "XY4", "II"):
        dd = em.apply_ddd(qc, sequence=sequence)
        dd_noisy = inject_coherent_idle_error(dd, em, detuning_rad_per_s)
        c = _run(dd_noisy, nm, shots, seed)
        ddd_counts[sequence] = c
        v = pc(c)
        results[f"ddd_{sequence}"] = v
        b = bootstrap_delta_unpaired(counts_base, c, n_prec, ideal_bin,
                                     n_boot=n_boot, seed=seed)
        d = b["delta"]
        sig = "sig" if d["significant"] else "n.s."
        tag = "  <- NULL CONTROL" if sequence == "II" else ""
        print(f"{'DDD ' + sequence:<24}{v:>10.4f}{v - results['none']:>+15.4f}"
              f"   [{d['lo']:+.4f},{d['hi']:+.4f}] {sig}{tag}")
        results[f"ddd_{sequence}_ci"] = d

    best = max(("XX", "YY", "XY4"), key=lambda s: results[f"ddd_{s}"])
    vs_null = bootstrap_delta_unpaired(ddd_counts["II"], ddd_counts[best],
                                       n_prec, ideal_bin, n_boot=n_boot, seed=seed)
    dn = vs_null["delta"]
    print(f"{'  best (' + best + ') vs II':<24}{'':>10}{dn['mean']:>+15.4f}"
          f"   [{dn['lo']:+.4f},{dn['hi']:+.4f}] "
          f"{'sig' if dn['significant'] else 'n.s. <- DDD NOT DEMONSTRATED'}")
    results["ddd_vs_null"] = dn

    # ---- REM ----------------------------------------------------------------
    cal = em.rem_calibration_circuits(qc, model="tensored")
    cal_counts = {}
    for label, circ in cal:
        cal_counts[label] = _run(circ, nm, shots, seed)
    mats = confusion_matrices(cal_counts["cal0"], cal_counts["cal1"], n_meas)
    a = tensored_confusion(mats[:n_prec])
    cond = condition_number(a)

    p_raw = to_probabilities(counts_to_m(counts_base, n_prec), n_prec)
    for method in ("naive", "simplex"):
        v = p_correct(rem_correct(p_raw, a, method=method), ideal_bin)
        results[f"rem_{method}"] = v
        print(f"{'REM (' + method + ')':<24}{v:>10.4f}{v - results['none']:>+15.4f}")
    print(f"{'  confusion cond(A)':<24}{cond:>10.2f}")

    boot = bootstrap_rem_paired(counts_base, cal_counts["cal0"],
                                cal_counts["cal1"], n_prec, ideal_bin, n_meas,
                                n_boot=n_boot, seed=seed)
    d = boot["delta"]
    print(f"{'  REM delta 95% CI':<24}[{d['lo']:+.4f}, {d['hi']:+.4f}]  "
          f"(paired bootstrap)")
    results["rem_delta_ci"] = d

    # ---- ZNE ----------------------------------------------------------------
    scales = (1, 1.5, 2, 2.5, 3)
    cset = em.zne_circuit_set(qc, scales=scales)
    ver = em.verify_zne(cset)
    counts_by_scale, curve = {}, []
    for entry in cset:
        noisy = inject_coherent_idle_error(entry["circuit"], em, detuning_rad_per_s)
        c = _run(noisy, nm, shots, seed)
        counts_by_scale[float(entry["scale"])] = c
        curve.append((entry["scale"], pc(c)))

    print()
    print(f"  ZNE noise curve (2q-scaling verified: {ver['ok']}):")
    for s, v in curve:
        print(f"    lambda={s:<5} P_correct={v:.4f}")
    fit = extrapolate([s for s, _ in curve], [v for _, v in curve])
    for model in ("richardson", "linear", "exponential"):
        v = fit.get(model)
        if isinstance(v, float):
            results[f"zne_{model}"] = v
            print(f"{'ZNE ' + model:<24}{v:>10.4f}{v - results['none']:>+15.4f}")
    print(f"  model spread: {fit.get('model_spread', float('nan')):.4f}"
          f"{'   <- models disagree wildly' if fit.get('model_spread', 0) > 0.2 else ''}")
    if fit.get("unphysical"):
        print(f"  UNPHYSICAL (outside [0,1]): {fit['unphysical']}"
              f"  -> extrapolation is INVALID at this depth, not merely noisy")

    zb = bootstrap_zne(counts_by_scale, n_prec, ideal_bin, n_boot=n_boot, seed=seed)
    for model, ci in zb["ci"].items():
        if ci["mean"] is not None:
            print(f"    {model:<14} 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")

    # ---- combined -----------------------------------------------------------
    best_dd = best
    dd = em.apply_ddd(qc, sequence=best_dd)
    cset_c = em.zne_circuit_set(dd, scales=scales)
    curve_c = []
    for entry in cset_c:
        noisy = inject_coherent_idle_error(entry["circuit"], em, detuning_rad_per_s)
        p = to_probabilities(counts_to_m(_run(noisy, nm, shots, seed), n_prec), n_prec)
        curve_c.append((entry["scale"],
                        p_correct(rem_correct(p, a, method="simplex"), ideal_bin)))
    fit_c = extrapolate([s for s, _ in curve_c], [v for _, v in curve_c])

    print()
    print(f"  COMBINED (DDD {best_dd} + REM + ZNE):")
    for model in ("richardson", "linear", "exponential"):
        v = fit_c.get(model)
        if isinstance(v, float):
            results[f"combined_{model}"] = v
            flag = "  UNPHYSICAL" if not (0.0 <= v <= 1.0) else ""
            print(f"{'    ' + model:<24}{v:>10.4f}{v - results['none']:>+15.4f}{flag}")
    if fit_c.get("unphysical"):
        print(f"    -> {fit_c['unphysical']} exceed [0,1]. Stacking REM under ZNE")
        print(f"       compounds it: REM raises every point on the curve, and the")
        print(f"       extrapolation then multiplies that shift by its lever arm.")

    print()
    print("-" * 76)
    print("Reminder: DDD is tested against the COHERENT detuning term only.")
    print("A purely Markovian model cannot show a DDD benefit at any strength,")
    print("so a null result here means 'not under this noise', not 'never'.")
    print("This simulation says NOTHING about whether IonQ's server-side")
    print("compiler preserves the folds/pulses on real hardware -- that is")
    print("MITIGATION_BRIEF_IONQ.md Section 9.1, entirely separate and unresolved.")
    print("=" * 76)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true",
                    help="run every technique in noisy simulation")
    ap.add_argument("--n-prec", type=int, default=4)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--f1q", type=float, default=0.9998)
    ap.add_argument("--f2q", type=float, default=0.9882)
    ap.add_argument("--readout-err", type=float, default=0.006)
    ap.add_argument("--detuning", type=float, default=2.0e5)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    if args.validate:
        validate(n_prec=args.n_prec, shots=args.shots, f_1q=args.f1q, f_2q=args.f2q,
                 p_readout=args.readout_err, detuning_rad_per_s=args.detuning,
                 n_boot=args.n_boot)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
