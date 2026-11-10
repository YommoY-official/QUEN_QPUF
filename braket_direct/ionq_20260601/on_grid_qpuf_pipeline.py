#!/usr/bin/env python3
"""
on_grid_qpuf_pipeline.py
==========================
NOISELESS-ONLY (this stage). No AWS/Braket imports anywhere in this file,
no device.run(), no hardware submission of any kind.

Yon's redirected physics direction (2026-09-08, see
/Users/yonnoylee/.claude/plans/there-is-a-thread-logical-clarke.md):
eigenbasis = computational basis, but with a FULLY GENERIC on-grid
diagonal U -- one independent random integer m_j per computational basis
state j (`random_diagonal_on_grid_unitary` in submit_QPUF_ntarg.py), not
the earlier per-qubit/tensor-product `on_grid_unitary`. Target is the
existing seeded RY/RZ product state (not an eigenstate), so the response
is a superposition over eigenvectors and the m1-vs-m2 scatter shows
several points along the m1=m2 diagonal, one per distinct m_j value
populated by the target -- a genuine "circle plot" rather than the single
delta spike an eigenstate target would give.

`--stage {noiseless,noisy,submit}`: `noiseless` and `noisy` are local
simulation only (`noisy` means a simulated IonQ-like noise model, NOT real
hardware). `--stage submit` builds/exports/guards the real submission
(QASM export, gate-count guard, manifest) but the actual `AwsDevice(...)`
construction and `.run()` call are import-scoped inside a function that is
UNREACHABLE unless the caller passes BOTH `--i-have-yons-go-ahead` and
answers an interactive y/N confirmation naming device/shots/cost -- see
`_submit_execute`'s docstring. `--stage retrieve --arn <arn>` fetches a
completed task's raw counts (also gated the same way: the `AwsQuantumTask`
import/instantiation lives inside `run_retrieve`, only reached if `--arn`
is given). No line in this file that could reach `device.run()`,
`AwsDevice(...)`, or `AwsQuantumTask(...)` executes as a side effect of
constructing the CLI or of any other `--stage`.
"""

import argparse
import json
import os
import sys

import numpy as np
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from submit_QPUF_ntarg import (
    random_diagonal_on_grid_unitary, on_grid_unitary, build_full_circuit,
    build_qpe_stage,
)
from decode_mcm_qpuf_result import (
    ideal_joint, clopper_pearson, circular_distance_pmf, verification_rate,
)
from on_grid_qpuf_plots import decode_bits, plot_m1_vs_m2

# mitig_analysis.py lives in a sibling campaign directory, not this one --
# reuse its build_noise_model/_run rather than reimplementing the noise
# model (same pattern this project already uses for same-directory cross-
# imports, e.g. probe_gate_limit.py importing from submit_QPUF_ntarg.py,
# just pointed at a different directory). Its module-level imports are
# lightweight (argparse/math/numpy/scipy only, no AWS/Braket) -- checked
# before importing.
MITIG_DIR = os.path.normpath(
    os.path.join(HERE, "..", "20260828_ionq_error_mitigation"))
sys.path.insert(0, MITIG_DIR)
from mitig_analysis import (
    build_noise_model, _run, resample_counts, bootstrap_ci, counts_to_m,
    to_probabilities, confusion_matrices, tensored_confusion, rem_correct,
    condition_number, extrapolate, inject_coherent_idle_error,
)
from error_mitig_ionq import ErrorMitigation
# ionq_mitigation_common's own module-level imports are json/os/sys/numpy/
# qiskit only -- no braket import at module scope (checked by reading the
# file), so this import is safe at module load for every --stage,
# including noiseless/noisy which never touch AWS.
import ionq_mitigation_common as imc

# Project convention for the RY/RZ product-state seed (submit_QPUF_ntarg.py
# module default, reused verbatim by two_stage_on_grid_baseline.py).
TARGET_INIT_SEED = 99

RESULTS_ROOT = os.path.join(HERE, "on_grid_results")
BASIS_GATES = ['rz', 'rx', 'rxx', 'measure', 'reset']


def _verify_decode_known_answer(n_prec: int = 4, n_targ: int = 3,
                                shots: int = 2000) -> None:
    """
    Known-answer test, run BEFORE any counts-derived number from this
    pipeline is trusted (per the explicit instruction not to assume
    decode_bits/parse_outcome-style conventions are correct without
    checking -- this repo's whole convention-A-vs-B saga in
    decode_mcm_qpuf_result.py exists because that assumption was wrong
    once already). Uses on_grid_unitary(3,4,ks=(1,2,4)) with
    eigenstate=True: |1...1> is an exact eigenvector with eigenphase
    (1+2+4) mod 16 = 7, so a correct decode must read (m1,m2)=(7,7) on
    EVERY shot, via the real genuine-MCM circuit on AerSimulator (not a
    statevector shortcut).
    """
    from qiskit_aer import AerSimulator

    U = on_grid_unitary(n_targ, n_prec, ks=(1, 2, 4))
    qc = build_full_circuit(n_prec, n_targ, U, target_init_seed=TARGET_INIT_SEED,
                            eigenstate=True)
    qc_t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
    counts = AerSimulator().run(qc_t, shots=shots).result().get_counts()

    expected = (7, 7)
    total = sum(counts.values())
    bad = sum(n for b, n in counts.items() if decode_bits(b, n_prec) != expected)
    print(f"[decode known-answer test] on_grid_unitary(ks=(1,2,4)) eigenstate, "
          f"{shots} shots, {len(counts)} unique bitstring(s): "
          f"{total - bad}/{total} decode to {expected}")
    if bad:
        raise RuntimeError(
            f"decode_bits FAILED the known-answer test: {bad}/{total} shots "
            f"did not decode to {expected} -- DO NOT trust downstream numbers "
            f"until this is fixed."
        )
    print("[decode known-answer test] PASSED.")


def _verify_decode_known_answer_noisy_backend(n_prec: int = 4, n_targ: int = 3,
                                              shots: int = 2000,
                                              seed_simulator: int = 1) -> None:
    """
    Same known-answer test as _verify_decode_known_answer, but run through
    the NOISY-simulation code path (mitig_analysis.py's `_run`:
    AerSimulator(noise_model=...), transpile(..., optimization_level=0))
    with a ZERO-noise noise model (build_noise_model(1.0, 1.0, 0.0)).

    Why this is a separate test rather than assuming Step 1's pass carries
    over: only the noise_model/basis_gates and optimization_level differ
    from the noiseless path, not the circuit's register structure -- but
    per this project's "no hallucinated APIs" rule, that assumption gets
    checked, not skipped, before any noisy-arm number is trusted.
    """
    U = on_grid_unitary(n_targ, n_prec, ks=(1, 2, 4))
    qc = build_full_circuit(n_prec, n_targ, U, target_init_seed=TARGET_INIT_SEED,
                            eigenstate=True)
    nm_zero = build_noise_model(1.0, 1.0, 0.0)
    counts = _run(qc, nm_zero, shots, seed_simulator)

    expected = (7, 7)
    total = sum(counts.values())
    bad = sum(n for b, n in counts.items() if decode_bits(b, n_prec) != expected)
    print(f"[decode known-answer test, NOISY-BACKEND format check, zero noise] "
          f"{total - bad}/{total} shots decode to {expected}  "
          f"({len(counts)} unique bitstring(s))")
    if bad:
        raise RuntimeError(
            f"decode_bits FAILED on the noisy-backend output format: "
            f"{bad}/{total} shots did not decode to {expected}."
        )
    print("[decode known-answer test, noisy backend] PASSED -- bitstring "
          "format is unchanged from the noiseless path.")


def _verify_noiseless_limit_regression(U: np.ndarray, n_prec: int, n_targ: int,
                                       shots: int = 2000,
                                       seed_simulator: int = 2) -> None:
    """
    Regression check: the noisy-simulation code path (build_noise_model +
    _run, optimization_level=0), asked to simulate ZERO noise
    (f_1q=f_2q=1.0, p_readout=0.0) on the ACTUAL seed-9 product-target
    circuit (not the eigenstate probe above), must still give
    P(m1==m2)=1.0 exactly -- confirms the new noisy-path plumbing didn't
    silently break the noiseless result Step 1 already established.
    """
    qc = build_full_circuit(n_prec, n_targ, U, target_init_seed=TARGET_INIT_SEED,
                            eigenstate=False)
    nm_zero = build_noise_model(1.0, 1.0, 0.0)
    counts = _run(qc, nm_zero, shots, seed_simulator)

    total = sum(counts.values())
    agree = sum(n for b, n in counts.items()
               if decode_bits(b, n_prec)[0] == decode_bits(b, n_prec)[1])
    p = agree / total
    print(f"[noiseless-limit regression] noisy code path, zero noise, "
          f"{shots} shots: P(m1==m2) = {p:.6f}  (expect 1.0 exactly)")
    if p != 1.0:
        raise RuntimeError(f"noiseless-limit regression FAILED: P(m1==m2)={p} != 1.0")
    print("[noiseless-limit regression] PASSED.")


def _joint_from_counts(counts: dict, n_prec: int) -> np.ndarray:
    """Decode a counts dict into a (2^n_prec)x(2^n_prec) integer joint via
    the verified decode_bits -- single source of truth for this decode."""
    d = 2 ** n_prec
    joint = np.zeros((d, d), dtype=int)
    for b, n in counts.items():
        m1, m2 = decode_bits(b, n_prec)
        joint[m1, m2] += n
    return joint


def _arm_stats(counts: dict, n_prec: int, ideal_bins) -> dict:
    """
    P(m1==m2), P(m1 in ideal_bins), P(m2 in ideal_bins), P(dist<=1 mod
    2^n_prec), each with a 95% Clopper-Pearson CI, reusing
    decode_mcm_qpuf_result.py's clopper_pearson/circular_distance_pmf/
    verification_rate (not reimplemented).
    """
    joint = _joint_from_counts(counts, n_prec)
    n_shots = int(joint.sum())
    ideal_set = set(int(m) for m in ideal_bins)

    k_diag = int(np.trace(joint))
    k_m1 = int(sum(joint[m, :].sum() for m in ideal_set))
    k_m2 = int(sum(joint[:, m].sum() for m in ideal_set))
    dist_counts = circular_distance_pmf(joint, d=2 ** n_prec)
    k_dist1 = int(dist_counts[:2].sum())

    def _stat(k):
        lo, hi = clopper_pearson(k, n_shots)
        return {"p": k / n_shots, "ci": [lo, hi], "k": k}

    return {
        "n_shots": n_shots,
        "p_m1_eq_m2": _stat(k_diag),
        "p_m1_in_ideal": _stat(k_m1),
        "p_m2_in_ideal": _stat(k_m2),
        "p_dist_le1": _stat(k_dist1),
    }


def _numpy_target_populations(n_targ: int, target_init_seed: int) -> np.ndarray:
    """
    Independent (non-qiskit) numpy reconstruction of the RY/RZ product
    target state's computational-basis populations |amp_j|^2, for the
    eigen-population cross-check against the sampled m1 histogram. Mirrors
    build_full_circuit's/ideal_joint's target-prep loop exactly (same
    init_rng draw order: theta ~ U(0,pi) then phi ~ U(0,2pi), per qubit,
    in order 0..n_targ-1) but computed directly as a statevector, not
    through Qiskit -- a genuinely independent check, not a re-run of the
    same code path.

    RY(theta)|0> = [cos(theta/2), sin(theta/2)];
    RZ(phi) = diag(exp(-i phi/2), exp(i phi/2)).
    Kronecker order matches on_grid_unitary's established convention:
    qubit 0 is the RIGHTMOST kron operand (little-endian: bit i of the
    basis index is qubit i).
    """
    init_rng = np.random.default_rng(seed=target_init_seed)
    state = np.array([1.0 + 0j])
    for _ in range(n_targ):
        theta = init_rng.uniform(0, np.pi)
        phi = init_rng.uniform(0, 2 * np.pi)
        psi_i = np.array([np.cos(theta / 2) * np.exp(-1j * phi / 2),
                          np.sin(theta / 2) * np.exp(1j * phi / 2)])
        state = np.kron(psi_i, state)
    return np.abs(state) ** 2


def run_noiseless(args: argparse.Namespace) -> None:
    from qiskit_aer import AerSimulator

    _verify_decode_known_answer()

    n_targ, n_prec = 3, 4
    rng = np.random.default_rng(seed=args.seed)
    U, m_j = random_diagonal_on_grid_unitary(n_targ, n_prec, rng, return_m_j=True)
    print(f"\nseed={args.seed}  m_j={m_j}")

    qc = build_full_circuit(n_prec, n_targ, U, target_init_seed=TARGET_INIT_SEED,
                            eigenstate=False)

    # Exact ideal joint. ideal_joint builds its own copy of the target-prep
    # circuit rather than taking build_full_circuit's qc directly -- but
    # its non-eigenstate branch is textually identical to build_full_circuit's
    # (same init_rng = default_rng(seed=target_init_seed), same ry-then-rz
    # loop over targ_idx in order), so passing the same target_init_seed
    # here reproduces the exact same target state deterministically. Cross-
    # checked below against the 10,000-shot sampled marginal as well.
    joint = ideal_joint(n_prec=n_prec, n_targ=n_targ,
                        target_init_seed=TARGET_INIT_SEED,
                        U=U, eigenstate=False)
    n_joint = joint.sum()
    p_diag_exact = float(np.trace(joint)) / n_joint
    marg1_exact = (joint.sum(axis=1) / n_joint)
    print(f"exact ideal_joint: P(m1==m2) = {p_diag_exact:.10f}  "
          f"(expect 1.0 to machine precision)")

    # Transpiled gate count, braket-arm (qiskit 2.3.0 / Aer 0.17.2) -- the
    # project's trusted gate-counter as of today, confirmed to match AWS
    # hardware exactly.
    qc_t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
    ops = qc_t.count_ops()
    n_1q = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q = ops.get('rxx', 0)
    n_gates = n_1q + n_2q
    print(f"braket-arm transpiled gate count: 1q={n_1q}  2q={n_2q}  "
          f"total={n_gates}  ({'UNDER' if n_gates <= 2000 else 'OVER'} the "
          f"2,000-gate IonQ limit)")

    sim = AerSimulator()
    sampled = {}
    for shots in (1000, 10000):
        counts = sim.run(qc_t, shots=shots).result().get_counts()
        m1m2 = [decode_bits(b, n_prec) for b in counts]
        p_diag_sampled = sum(n for b, n in counts.items()
                             if decode_bits(b, n_prec)[0] == decode_bits(b, n_prec)[1]) / shots
        print(f"shots={shots}: sampled P(m1==m2) = {p_diag_sampled:.4f}  "
              f"({len(counts)} unique bitstrings)")
        sampled[shots] = {
            "counts": counts,
            "p_diag_sampled": p_diag_sampled,
        }

    # Independent numpy eigen-population cross-check: merge target
    # populations by shared m_j, compare to the observed m1 histogram at
    # 10,000 shots (statistical agreement expected, not bit-exact).
    pops = _numpy_target_populations(n_targ, TARGET_INIT_SEED)
    predicted_marg = np.zeros(2 ** n_prec)
    for j, m in enumerate(m_j):
        predicted_marg[m] += pops[j]

    counts_10k = sampled[10000]["counts"]
    observed_m1 = np.zeros(2 ** n_prec)
    for b, n in counts_10k.items():
        m1, _ = decode_bits(b, n_prec)
        observed_m1[m1] += n
    observed_m1 /= 10000

    print("\nm-bin  |  predicted (numpy pops merged by m_j)  |  exact ideal_joint marg1  |  observed @10k shots")
    for m in range(2 ** n_prec):
        if predicted_marg[m] > 0 or observed_m1[m] > 0:
            print(f"  {m:3d}  |  {predicted_marg[m]:.4f}  |  {marg1_exact[m]:.4f}  |  {observed_m1[m]:.4f}")
    max_abs_diff_exact_vs_numpy = float(np.max(np.abs(predicted_marg - marg1_exact)))
    print(f"max|numpy predicted - exact ideal_joint marg1| = {max_abs_diff_exact_vs_numpy:.2e} "
          f"(expect ~0 -- both are exact, independently derived)")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = os.path.join(RESULTS_ROOT, args.date)
    os.makedirs(out_dir, exist_ok=True)

    record = {
        "stage": "noiseless",
        "n_targ": n_targ,
        "n_prec": n_prec,
        "seed": args.seed,
        "m_j": m_j,
        "target_init_seed": TARGET_INIT_SEED,
        "gate_count": {"n_1q": n_1q, "n_2q": n_2q, "total": n_gates},
        "ideal_joint": {
            "p_m1_eq_m2": p_diag_exact,
            "marg1": marg1_exact.tolist(),
        },
        "numpy_predicted_marg1": predicted_marg.tolist(),
        "sampled": {
            str(shots): {
                "n_shots": shots,
                "p_m1_eq_m2": sampled[shots]["p_diag_sampled"],
                "counts": sampled[shots]["counts"],
            }
            for shots in (1000, 10000)
        },
    }
    out_json = os.path.join(out_dir, "noiseless_results.json")
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_m1_vs_m2(sampled[1000]["counts"], n_prec, m_j, 1000,
                 title_suffix="noiseless AerSimulator", ax=axes[0])
    plot_m1_vs_m2(sampled[10000]["counts"], n_prec, m_j, 10000,
                 title_suffix="noiseless AerSimulator", ax=axes[1])
    fig.suptitle(
        f"Random per-basis-state on-grid U (computational eigenbasis), "
        f"n_targ={n_targ}, n_prec={n_prec}, seed={args.seed}\n"
        f"m_j={m_j}  --  P(m1==m2): exact={p_diag_exact:.4f}",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(out_dir, "on_grid_computational_eigenbasis_circle_plot.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


# Nominal IonQ Forte-1 calibration-derived fidelities (mitig_analysis.py's
# `validate()` defaults) and the seed_simulator values for each noisy arm --
# fixed and recorded here so every arm is independently reproducible.
NOM_F1Q, NOM_F2Q, NOM_PREAD = 0.9998, 0.9882, 0.006
ARM_SEEDS = {
    "nominal_1000": 101,
    "nominal_10000": 102,
    "f2q_0.985": 103,
    "f2q_0.995": 104,
    "readout_only": 105,
}


def run_noisy(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    n_targ, n_prec = 3, 4
    rng = np.random.default_rng(seed=args.seed)
    U, m_j = random_diagonal_on_grid_unitary(n_targ, n_prec, rng, return_m_j=True)
    ideal_bins = sorted(set(m_j))
    print(f"seed={args.seed}  m_j={m_j}  ideal_bins={ideal_bins}")

    _verify_decode_known_answer_noisy_backend()
    _verify_noiseless_limit_regression(U, n_prec, n_targ)

    qc = build_full_circuit(n_prec, n_targ, U, target_init_seed=TARGET_INIT_SEED,
                            eigenstate=False)

    # Gate count under the NOISE-MODEL transpile path (optimization_level=0,
    # target=noisy AerSimulator backend) -- NOT directly comparable to Step
    # 1's 710-gate braket-arm reference (optimization_level=1, plain
    # basis_gates list, for hardware-submission sizing). Different purpose,
    # different optimization level -> different gate count is expected, not
    # a discrepancy to chase.
    from qiskit_aer import AerSimulator
    nm_probe = build_noise_model(NOM_F1Q, NOM_F2Q, NOM_PREAD)
    sim_probe = AerSimulator(noise_model=nm_probe, basis_gates=["rz", "rx", "rxx", "id"])
    qc_t_noisy = transpile(qc, sim_probe, optimization_level=0)
    ops = qc_t_noisy.count_ops()
    n_1q_noisy = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q_noisy = ops.get('rxx', 0)
    n_gates_noisy = n_1q_noisy + n_2q_noisy
    print(f"\nnoisy-transpile (opt_level=0) gate count: 1q={n_1q_noisy}  "
          f"2q={n_2q_noisy}  total={n_gates_noisy}")
    print("Step 1 braket-arm reference (opt_level=1, hardware-submission "
          "sizing): total=710  -- NOT directly comparable (different "
          "optimization level / purpose).")

    # ── Run all arms ─────────────────────────────────────────────────────
    arm_counts = {}

    nm_nominal = build_noise_model(NOM_F1Q, NOM_F2Q, NOM_PREAD)
    arm_counts["nominal_1000"] = _run(qc, nm_nominal, 1000,
                                      ARM_SEEDS["nominal_1000"])
    arm_counts["nominal_10000"] = _run(qc, nm_nominal, 10000,
                                       ARM_SEEDS["nominal_10000"])

    nm_985 = build_noise_model(NOM_F1Q, 0.985, NOM_PREAD)
    arm_counts["f2q_0.985"] = _run(qc, nm_985, 10000, ARM_SEEDS["f2q_0.985"])

    # F2Q=0.9882 is the same noise model as the nominal_10000 arm -- reuse
    # its counts rather than resimulate (explicit, not silent).
    arm_counts["f2q_0.9882"] = arm_counts["nominal_10000"]
    print("\n(f2q_0.9882 sensitivity point reuses the nominal_10000 arm's "
          "counts -- identical noise model, no resimulation.)")

    nm_995 = build_noise_model(NOM_F1Q, 0.995, NOM_PREAD)
    arm_counts["f2q_0.995"] = _run(qc, nm_995, 10000, ARM_SEEDS["f2q_0.995"])

    nm_readout_only = build_noise_model(1.0, 1.0, NOM_PREAD)
    arm_counts["readout_only"] = _run(qc, nm_readout_only, 10000,
                                      ARM_SEEDS["readout_only"])

    # ── Sanity: the noise model must actually bite ──────────────────────
    nominal_stats_check = _arm_stats(arm_counts["nominal_10000"], n_prec, ideal_bins)
    p_nom = nominal_stats_check["p_m1_eq_m2"]["p"]
    print(f"\n[noise-model sanity] nominal (F1Q={NOM_F1Q}, F2Q={NOM_F2Q}, "
          f"p_readout={NOM_PREAD}) P(m1==m2) @10,000 shots = {p_nom:.4f}")
    if p_nom > 0.999:
        raise RuntimeError(
            f"noise-model sanity check FAILED: nominal P(m1==m2)={p_nom:.4f} "
            "is suspiciously close to the noiseless 1.0 -- the noise model "
            "is likely attaching to nothing (e.g. transpile cancelled the "
            "gates it was meant to attach to). Investigate before trusting "
            "any noisy-arm number."
        )
    print("[noise-model sanity] PASSED -- nominal P(m1==m2) is measurably "
          "below 1.0, confirming the noise model is biting.")

    # ── Stats table ──────────────────────────────────────────────────────
    arm_order = ["nominal_1000", "nominal_10000", "f2q_0.985", "f2q_0.9882",
                "f2q_0.995", "readout_only"]
    stats = {name: _arm_stats(arm_counts[name], n_prec, ideal_bins)
             for name in arm_order}

    print(f"\nideal bins: {ideal_bins}")
    print(f"{'arm':16s} {'n':>7s}  {'P(m1==m2)':>20s}  {'P(m1 in ideal)':>20s}  "
          f"{'P(m2 in ideal)':>20s}  {'P(dist<=1)':>20s}")
    for name in arm_order:
        s = stats[name]
        def _fmt(d):
            return f"{d['p']:.4f} [{d['ci'][0]:.4f},{d['ci'][1]:.4f}]"
        print(f"{name:16s} {s['n_shots']:7d}  {_fmt(s['p_m1_eq_m2']):>20s}  "
              f"{_fmt(s['p_m1_in_ideal']):>20s}  {_fmt(s['p_m2_in_ideal']):>20s}  "
              f"{_fmt(s['p_dist_le1']):>20s}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = os.path.join(RESULTS_ROOT, args.date)
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "stage": "noisy",
        "n_targ": n_targ,
        "n_prec": n_prec,
        "seed": args.seed,
        "m_j": m_j,
        "ideal_bins": ideal_bins,
        "target_init_seed": TARGET_INIT_SEED,
        "nominal_fidelities": {"f_1q": NOM_F1Q, "f_2q": NOM_F2Q, "p_readout": NOM_PREAD},
        "arm_seeds": ARM_SEEDS,
        "gate_count_noisy_transpile": {"n_1q": n_1q_noisy, "n_2q": n_2q_noisy,
                                       "total": n_gates_noisy},
        "gate_count_braket_arm_reference_opt1": 710,
        "arms": {name: {**stats[name], "counts": arm_counts[name]}
                for name in arm_order},
    }
    out_json = os.path.join(out_dir, "noisy_results.json")
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {out_json}")

    # ── Plots ────────────────────────────────────────────────────────────
    noiseless_json = os.path.join(out_dir, "noiseless_results.json")
    with open(noiseless_json) as f:
        step1 = json.load(f)
    noiseless_1000 = step1["sampled"]["1000"]["counts"]
    noiseless_10000 = step1["sampled"]["10000"]["counts"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    plot_m1_vs_m2(noiseless_1000, n_prec, m_j, 1000,
                 title_suffix="noiseless (Step 1)", ax=axes[0, 0])
    plot_m1_vs_m2(noiseless_10000, n_prec, m_j, 10000,
                 title_suffix="noiseless (Step 1)", ax=axes[0, 1])
    plot_m1_vs_m2(arm_counts["nominal_1000"], n_prec, m_j, 1000,
                 title_suffix=f"noisy nominal (F2Q={NOM_F2Q})", ax=axes[1, 0])
    plot_m1_vs_m2(arm_counts["nominal_10000"], n_prec, m_j, 10000,
                 title_suffix=f"noisy nominal (F2Q={NOM_F2Q})", ax=axes[1, 1])
    fig.suptitle(
        f"Noiseless vs. noisy-nominal, n_targ={n_targ}, n_prec={n_prec}, "
        f"seed={args.seed}, m_j={m_j}",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png1 = os.path.join(out_dir, "on_grid_noisy_nominal_circle_plot.png")
    fig.savefig(out_png1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png1}")

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 12))
    plot_m1_vs_m2(arm_counts["f2q_0.985"], n_prec, m_j, 10000,
                 title_suffix="F2Q=0.985", ax=axes2[0, 0])
    plot_m1_vs_m2(arm_counts["f2q_0.9882"], n_prec, m_j, 10000,
                 title_suffix="F2Q=0.9882 (nominal)", ax=axes2[0, 1])
    plot_m1_vs_m2(arm_counts["f2q_0.995"], n_prec, m_j, 10000,
                 title_suffix="F2Q=0.995", ax=axes2[1, 0])
    plot_m1_vs_m2(arm_counts["readout_only"], n_prec, m_j, 10000,
                 title_suffix="readout-only (F1Q=F2Q=1.0)", ax=axes2[1, 1])
    fig2.suptitle(
        f"F2Q sensitivity + readout-only arm, n_targ={n_targ}, n_prec={n_prec}, "
        f"seed={args.seed}, 10,000 shots each",
        fontsize=10.5,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    out_png2 = os.path.join(out_dir, "on_grid_noisy_sensitivity_circle_plot.png")
    fig2.savefig(out_png2, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out_png2}")


SUBMIT_SHOTS = 1000


def _build_submission_pieces(n_prec: int, n_targ: int, U: np.ndarray,
                             target_init_seed: int, eigenstate: bool = False):
    """
    Build the two program-order PIECES of build_full_circuit's circuit
    (target prep + stage-1 QPE + stage-1 measure; stage-2 QPE + stage-2
    measure), each over the SAME flat q/c registers build_full_circuit
    uses, so no qubit/clbit remapping is needed when the transpiled pieces
    are concatenated back together.

    Why pieces, not one circuit: a single transpile() call on the whole
    circuit reorders instructions by DAG DEPENDENCY, not program order --
    the stage-1 measure only touches prec_a qubits/clbits, so nothing in
    the dependency graph forces it before stage-2's gates on prec_b/target,
    and Qiskit's basis-translation/optimization passes pushed every gate
    (including stage-2's) before any measurement, destroying the MCM's
    textual position in the exported QASM. Transpiling each piece alone
    can't do this: piece 1's measure operations are its own terminal
    instructions (nothing follows them in piece 1 to get reordered ahead
    of), and likewise for piece 2.
    """
    n_total = n_targ + 2 * n_prec
    q = QuantumRegister(n_total, "q")
    c = ClassicalRegister(2 * n_prec, "c")

    targ_idx = list(range(0, n_targ))
    prec_a_idx = list(range(n_targ, n_targ + n_prec))
    prec_b_idx = list(range(n_targ + n_prec, n_targ + 2 * n_prec))

    # Piece 1: target prep + stage-1 QPE + stage-1 measure. Idle on prec_b
    # (no instructions there at all -- untouched qubits, not identity gates).
    qc1 = QuantumCircuit(q, c)
    if eigenstate:
        for i in targ_idx:
            qc1.x(q[i])
    else:
        init_rng = np.random.default_rng(seed=target_init_seed)
        for i in targ_idx:
            qc1.ry(init_rng.uniform(0, np.pi), q[i])
            qc1.rz(init_rng.uniform(0, 2 * np.pi), q[i])
    qpe1 = build_qpe_stage(n_prec, U)
    qc1.append(qpe1, prec_a_idx + targ_idx)
    qc1.measure([q[i] for i in prec_a_idx], [c[k] for k in range(n_prec)])

    # Piece 2: stage-2 QPE + stage-2 measure ONLY -- no target re-prep. The
    # target's state at this point is whatever piece 1 (run first, on the
    # real device) physically collapsed it to; piece 2 must not touch it
    # except via stage-2's own QPE gates, exactly as build_full_circuit does.
    qc2 = QuantumCircuit(q, c)
    qpe2 = build_qpe_stage(n_prec, U)
    qc2.append(qpe2, prec_b_idx + targ_idx)
    qc2.measure([q[i] for i in prec_b_idx], [c[n_prec + k] for k in range(n_prec)])

    return qc1, qc2


def _assemble_in_program_order(qc1_t: QuantumCircuit, qc2_t: QuantumCircuit,
                               n_total: int, n_clbits: int) -> QuantumCircuit:
    """
    Concatenate two independently-transpiled pieces' instructions in
    PROGRAM ORDER (piece 1 entirely, then piece 2 entirely) into a fresh
    flat-register circuit -- no further transpile()/optimization pass
    afterward, since that is exactly what would re-invite DAG reordering.
    Copies each instruction's exact qubit/clbit objects rather than using
    compose()'s positional remapping, to avoid any ambiguity about which
    qubit is which.
    """
    q = QuantumRegister(n_total, "q")
    c = ClassicalRegister(n_clbits, "c")
    assembled = QuantumCircuit(q, c)
    for piece in (qc1_t, qc2_t):
        for instr in piece.data:
            qubit_idx = [piece.find_bit(qb).index for qb in instr.qubits]
            clbit_idx = [piece.find_bit(cb).index for cb in instr.clbits]
            assembled.append(instr.operation,
                            [q[i] for i in qubit_idx],
                            [c[i] for i in clbit_idx])
    return assembled


def _build_submission(args) -> dict:
    """
    Build/transpile/export/guard the seed-9 submission circuit -- no AWS
    import or call anywhere in this function. Shared by --dry-run and the
    (gated, unreached-by-default) real-submission path so both submit
    exactly the same QASM the dry-run already showed.
    """
    n_targ, n_prec = 3, 4
    rng = np.random.default_rng(seed=args.seed)
    U, m_j = random_diagonal_on_grid_unitary(n_targ, n_prec, rng, return_m_j=True)
    ideal_bins = sorted(set(m_j))

    # Two-piece transpile + program-order reassembly (see
    # _build_submission_pieces/_assemble_in_program_order docstrings) --
    # NOT a single transpile() on build_full_circuit's output, which
    # reorders the mid-circuit measurement out of its correct textual
    # position (real bug, found by the Verifier). SAME basis/optimization
    # settings per piece as the original 710-gate reference, so gate-level
    # correctness within each piece is unchanged from what was already
    # verified -- only the assembly step changed.
    qc1, qc2 = _build_submission_pieces(n_prec, n_targ, U, TARGET_INIT_SEED,
                                        eigenstate=False)
    qc1_t = transpile(qc1, basis_gates=BASIS_GATES, optimization_level=1)
    qc2_t = transpile(qc2, basis_gates=BASIS_GATES, optimization_level=1)
    qc_t = _assemble_in_program_order(qc1_t, qc2_t, n_targ + 2 * n_prec, 2 * n_prec)

    ops = qc_t.count_ops()
    n_1q = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q = ops.get('rxx', 0)
    n_gates = n_1q + n_2q

    # Hard guard: real, empirically-confirmed AWS/IonQ per-circuit limit
    # (ionq_mitigation_common.MAX_GATES_PER_CIRCUIT), not reimplemented.
    imc.assert_gate_count_ok(qc_t, label=f"on_grid_qpuf seed={args.seed} submission")

    # Non-verbatim, non-physical export (to_braket_qasm's defaults) --
    # matches THIS project's own prior real IonQ submissions
    # (submit_QPUF_ntarg.py / two_stage_on_grid_baseline.py both export
    # plain q[]-indexed QASM with the rxx->xx rewrite, no verbatim box),
    # not the verbatim/physical-addressing mode ionq_mitigation_common's
    # OWN campaign uses for its different (ZNE/DDD-fold-preserving) needs.
    qasm_src = imc.to_braket_qasm(qc_t)
    imc.assert_supported_gates(qasm_src)

    # Clbit -> physical qubit map, BEFORE any AWS-side compilation --
    # needed to write the decode rule explicitly (see module docstring's
    # convention-B discussion in decode_mcm_qpuf_result.py).
    phys_qubits = imc.measured_physical_qubits(qc_t)

    ideal = ideal_joint(n_prec=n_prec, n_targ=n_targ, target_init_seed=TARGET_INIT_SEED,
                        U=U, eigenstate=False)
    n_joint = ideal.sum()
    p_diag = float(np.trace(ideal)) / n_joint
    marg1 = (ideal.sum(axis=1) / n_joint).tolist()

    # Explicit char -> (m1, m2) rule for THIS circuit's qubit/clbit layout,
    # written out at the same level of detail as decode_mcm_qpuf_result.py's
    # module docstring (not just the label "convention B"). Raw Braket
    # measurement_counts keys are one char per MEASURED qubit, ordered by
    # ASCENDING physical qubit index -- phys_qubits above is already in
    # clbit order [c0's qubit, c1's qubit, ...], so if phys_qubits is
    # already sorted ascending (checked below, true for this circuit: no
    # ZNE-fold/DDD circuit rebuild ever ran, and Forte-Enterprise-1 is
    # all-to-all so no routing permutes qubits either), clbit i's value is
    # exactly the ascending-order character at position i.
    is_ascending = (phys_qubits == sorted(phys_qubits))
    if not is_ascending:
        raise RuntimeError(
            f"measured_physical_qubits={phys_qubits} is NOT ascending in "
            "clbit order -- the char-position <-> clbit shortcut this "
            "decode rule assumes does not hold; do not write the manifest "
            "rule below without re-deriving it for this actual mapping."
        )
    decode_rule = (
        f"Raw Braket measurement_counts keys are one character per MEASURED "
        f"qubit, ordered by ASCENDING physical qubit index. With "
        f"measured_qubits={phys_qubits} (q0-q2 = target, unmeasured): "
        + "; ".join(f"char {i} -> q{q}" for i, q in enumerate(phys_qubits))
        + f". q{phys_qubits[0]}-q{phys_qubits[n_prec - 1]} are the stage-1 "
        f"precision register (clbits c[0..{n_prec - 1}]) and "
        f"q{phys_qubits[n_prec]}-q{phys_qubits[-1]} are stage-2 "
        f"(clbits c[{n_prec}..{2 * n_prec - 1}]); build_qpe_stage's "
        f"MSB-first convention makes q{phys_qubits[0]} the MSB of m1 and "
        f"q{phys_qubits[n_prec]} the MSB of m2. So, with bits = the raw "
        f"counts key string read left to right: "
        f"m1 = int(bits[0:{n_prec}], 2), "
        f"m2 = int(bits[{n_prec}:{2 * n_prec}], 2)."
    )

    return {
        "n_targ": n_targ, "n_prec": n_prec, "seed": args.seed, "m_j": m_j,
        "ideal_bins": ideal_bins, "U": U, "qc_t": qc_t, "qasm_src": qasm_src,
        "n_1q": n_1q, "n_2q": n_2q, "n_gates": n_gates,
        "phys_qubits": phys_qubits, "decode_rule": decode_rule,
        "p_m1_eq_m2": p_diag, "marg1": marg1, "ideal_joint": ideal,
    }


def run_submit_dry_run(args) -> None:
    """
    Build/export/guard the submission (no AWS import or call anywhere in
    this function or in _build_submission). Writes the .qasm file, the
    manifest stub, and the full ideal-joint matrix to
    on_grid_results/<date>/submit/.
    """
    b = _build_submission(args)
    print(f"seed={b['seed']}  m_j={b['m_j']}  ideal_bins={b['ideal_bins']}")
    print(f"gate count (opt_level=1, matches Step 1's reference): "
          f"1q={b['n_1q']}  2q={b['n_2q']}  total={b['n_gates']}  "
          f"({'UNDER' if b['n_gates'] <= imc.MAX_GATES_PER_CIRCUIT else 'OVER'} "
          f"the {imc.MAX_GATES_PER_CIRCUIT}-gate limit)")
    print(f"measured physical qubits (clbit order): {b['phys_qubits']}")
    print(f"ideal P(m1==m2) = {b['p_m1_eq_m2']:.10f}")
    print(f"\ndecode rule:\n  {b['decode_rule']}")

    out_dir = os.path.join(RESULTS_ROOT, args.date, "submit")
    os.makedirs(out_dir, exist_ok=True)

    qasm_path = os.path.join(out_dir, "on_grid_seed9_submission.qasm")
    with open(qasm_path, "w") as f:
        f.write(b["qasm_src"])
    print(f"\nwrote {qasm_path}")

    ideal_joint_path = os.path.join(out_dir, "on_grid_seed9_ideal_joint.json")
    with open(ideal_joint_path, "w") as f:
        json.dump({
            "n_prec": b["n_prec"], "n_targ": b["n_targ"], "seed": b["seed"],
            "m_j": b["m_j"], "joint": b["ideal_joint"].tolist(),
        }, f, indent=2)
    print(f"wrote {ideal_joint_path}")

    manifest = {
        "device_name": imc.DEVICE_NAME,
        "device_arn": imc.DEVICE_ARN,
        "shots": SUBMIT_SHOTS,
        "seed": b["seed"],
        "m_j": b["m_j"],
        "ideal_bins": b["ideal_bins"],
        "n_targ": b["n_targ"],
        "n_prec": b["n_prec"],
        "target_init_seed": TARGET_INIT_SEED,
        "gate_counts": {"n_1q": b["n_1q"], "n_2q": b["n_2q"], "total": b["n_gates"]},
        "gate_limit": imc.MAX_GATES_PER_CIRCUIT,
        "measured_physical_qubits_clbit_order": b["phys_qubits"],
        "decode_convention": {
            "label": "raw Braket convention B (see decode_mcm_qpuf_result.py "
                     "module docstring, 'THE CONVENTION CONFLICT')",
            "source_call": "AwsQuantumTask(arn).result().measurement_counts "
                           "-- NOT via qiskit_braket_provider, which applies "
                           "a {k[::-1]: v} bit-reversal producing the "
                           "REJECTED convention A instead.",
            "rule": b["decode_rule"],
        },
        "ideal_joint_summary": {
            "p_m1_eq_m2": b["p_m1_eq_m2"],
            "marg1": b["marg1"],
            "full_matrix_file": "on_grid_seed9_ideal_joint.json",
        },
        "qasm_file": "on_grid_seed9_submission.qasm",
        "program_order_fix": {
            "bug_fixed": "A single global transpile() call on build_full_circuit's "
                        "output reorders instructions by DAG DEPENDENCY, not "
                        "program order: the stage-1 measure only touches "
                        "prec_a qubits/clbits, so nothing forces it before "
                        "stage-2's gates on prec_b/target, and Qiskit pushed "
                        "every gate (including stage-2's) before any "
                        "measurement -- destroying the MCM's textual position "
                        "in the exported QASM.",
            "fix": "Each piece (target-prep+stage-1+stage-1-measure; "
                  "stage-2+stage-2-measure) is transpiled SEPARATELY at the "
                  "same basis/optimization settings, then concatenated in "
                  "program order into a fresh circuit with NO further "
                  "transpile() call -- confirmed in the exported QASM TEXT "
                  "(not just the Python object): stage-1 measures appear "
                  "textually after the last stage-1 gate and before the "
                  "first stage-2 gate.",
            "noiseless_equivalence_caveat": "LocalSimulator (AerSimulator) "
                  "reproduces the ideal joint under this manifest's decode "
                  "rule: 100% on-diagonal, correct per-bin marginal. This "
                  "confirms program-order preservation is NECESSARY for a "
                  "correct submission but it is NOT SUFFICIENT to prove the "
                  "physical device executes the measurement in that "
                  "position -- noiseless statistics are IDENTICAL whether "
                  "the mid-circuit measurement is genuinely performed "
                  "mid-circuit or the measurement is effectively deferred "
                  "by the server-side compiler, since there is no classical "
                  "feedforward in this circuit for either ordering to "
                  "affect. Whether IonQ/Braket's server-side compiler "
                  "preserves genuine mid-circuit execution (vs. an internal "
                  "deferred-measurement rewrite) is a property of that "
                  "compiler, UNVERIFIED here, and only a real hardware run "
                  "(or documented compiler behavior) could settle it.",
        },
    }
    manifest_path = os.path.join(out_dir, "on_grid_seed9_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


def _submit_execute(args) -> None:
    """
    REAL submission path. Every AWS/Braket name (AwsDevice, OpenQasmProgram,
    boto session) is imported INSIDE this function, which is only ever
    called from run_submit after BOTH --i-have-yons-go-ahead was passed AND
    an interactive y/N confirmation (imc.confirm_submit, which itself
    refuses non-TTY input) was answered 'y'. Nothing above this function
    call in run_submit constructs or touches any AWS object.

    On-demand (bare AwsDevice(DEVICE_ARN).run(...), no DirectReservation --
    matches probe_gate_limit.py's pattern, per the Director's "on-demand"
    instruction, not submit_QPUF_ntarg.py's reservation pattern).
    """
    from braket.aws import AwsDevice
    from braket.ir.openqasm import Program as OpenQasmProgram

    b = _build_submission(args)
    price_estimate = SUBMIT_SHOTS * 0.08  # $0.08/shot, IonQ on-demand list price (no task fee counted)
    question = (f"\nType 'y' to submit to {imc.DEVICE_NAME} ({imc.DEVICE_ARN}): "
               f"{SUBMIT_SHOTS} shots, {b['n_gates']} gates, "
               f"~${price_estimate:.2f} (shot cost only, excludes any "
               f"per-task fee) [y/N]: ")
    if not imc.confirm_submit(question):
        return

    device = AwsDevice(imc.DEVICE_ARN)
    print(f"Device resolved: {device.name}")
    task = device.run(OpenQasmProgram(source=b["qasm_src"]), shots=SUBMIT_SHOTS)

    from datetime import datetime, timezone
    record = {
        "job_id": task.id,
        "datetime": datetime.now(timezone.utc).isoformat(),
        "device_name": imc.DEVICE_NAME,
        "device_arn": imc.DEVICE_ARN,
        "shots": SUBMIT_SHOTS,
        "seed": b["seed"],
        "m_j": b["m_j"],
        "gate_counts": {"n_1q": b["n_1q"], "n_2q": b["n_2q"], "total": b["n_gates"]},
        "tags": imc.task_tags("on-grid-computational-eigenbasis"),
    }
    out_dir = os.path.join(RESULTS_ROOT, args.date, "submit")
    log_path = imc.append_job_log(out_dir, record)
    print(f"Task ARN: {task.id}")
    print(f"Logged to: {log_path}")


def run_submit(args) -> None:
    # Mirrors probe_gate_limit.py's proven-safe gating: dry_run is True
    # unless BOTH --i-have-yons-go-ahead is given AND --dry-run is absent.
    # _submit_execute (the only place an AWS name is even imported) is
    # therefore textually unreachable from this branch whenever either
    # condition fails -- not just logically unlikely to run.
    dry_run = (not args.i_have_yons_go_ahead) or args.dry_run
    if dry_run:
        run_submit_dry_run(args)
        if not args.i_have_yons_go_ahead:
            print("\n(--i-have-yons-go-ahead not given -> refusing any real "
                 "submission attempt; this is the default. Real submission "
                 "additionally requires an interactive y/N confirmation "
                 "naming device/shots/cost.)")
        return
    _submit_execute(args)


def _build_retrieved_record(arn: str, result) -> dict:
    """
    Pure logic, no AWS call: turns an already-fetched Braket `result` object
    into the JSON record. Factored out so it can be unit-tested against a
    mocked `result` without ever importing/instantiating AwsQuantumTask
    (see this module's own mocked self-test, run during self-verification,
    not shipped here).

    Uses `result.measurement_counts` DIRECTLY -- deliberately NOT
    checkRetrieve.py's `retrieve_counts()`, which goes through
    qiskit_braket_provider.BraketProvider and applies a `{k[::-1]: v}`
    bit-reversal to every key (see decode_mcm_qpuf_result.py's docstring on
    convention A vs B). This project's manifest decode rule assumes the RAW
    Braket convention (the one the prior real hardware result's provenance
    established) -- going through qiskit_braket_provider here would
    silently hand back the REJECTED convention A instead. Only
    checkRetrieve.py's convention-agnostic metadata helpers (task_uuid,
    extract_qpu_time, serialize_additional_metadata) are reused.
    """
    from checkRetrieve import task_uuid, extract_qpu_time, serialize_additional_metadata

    return {
        "task_arn": arn,
        "uuid": task_uuid(arn),
        "measurement_counts": dict(result.measurement_counts),
        "qpu_time_seconds": extract_qpu_time(result),
        "additional_metadata": serialize_additional_metadata(result),
        "_counts_convention": "RAW BRAKET (result.measurement_counts directly, "
                              "no qiskit_braket_provider in the path) -- decode "
                              "with the rule in on_grid_seed9_manifest.json's "
                              "decode_convention field.",
    }


def run_retrieve(args) -> None:
    """
    Fetch one completed task's raw counts by ARN. DRY-RUN/SCAFFOLDING as of
    this task -- there is no submitted task yet, so this is exercised via a
    mocked `result` object during self-verification, never against real AWS
    in this session. AwsQuantumTask is imported and instantiated only here,
    only when --arn is actually given.
    """
    if not args.arn:
        print("ERROR: --stage retrieve requires --arn <task ARN>")
        sys.exit(1)

    from braket.aws import AwsQuantumTask

    print(f"Retrieving task {args.arn} ...")
    task = AwsQuantumTask(args.arn)
    result = task.result()
    record = _build_retrieved_record(args.arn, result)

    out_dir = os.path.join(RESULTS_ROOT, args.date, "submit")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{record['uuid']}_retrieved.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {out_path}")


# =========================================================================
# Step 4: error mitigation on the noisy simulator
# =========================================================================
#
# Two findings verified live by the Director before this section was
# written (do not re-derive, build around them):
#
# Finding 1 -- ErrorMitigation.fold()/apply_ddd() cannot run on the
# assembled two-stage circuit directly: fold()'s remove_final_measurements
# judges "final" per-qubit, not per-circuit, so on a circuit where one
# qubit's LAST op is a mid-circuit measure while OTHER qubits still have
# later gates, it strips BOTH measurement groups, folds everything as one
# continuous block, then reattaches every measurement at the very end --
# silently re-deferring the MCM Step 3 just fixed. Fix: fold/DDD each
# PIECE separately (single terminal measurement group per piece, exactly
# the shape these functions handle correctly), then reassemble.
#
# Finding 2 -- even transpile(circ, sim, optimization_level=0) reorders
# instructions (verified live: a measure on an otherwise-idle qubit moves
# after independent later gates, zero basis translation needed). So
# mitig_analysis._run() (which calls exactly that transpile internally)
# must NOT be used on the final assembled circuit -- _sim_run_notranspile
# below calls sim.run() directly instead.
#
# An ADDITIONAL bug found and fixed while implementing this (not one of
# the two above, verified independently): ErrorMitigation.fold()'s
# _reattach_measurements uses `enumerate(measured_physical_qubits(original))`
# to rebuild measure instructions, which silently renumbers clbits to
# start at 0 -- correct for a single-stage circuit (ionq_mitigation_common's
# own build_qpe_circuit, what this module was written for) but WRONG for
# piece 2 in isolation, whose real clbits are n_prec..2*n_prec-1 of the
# shared register (verified empirically: fold() on a circuit measuring
# qubits into clbits 4-7 reattaches them to clbits 0-3 instead). Fix:
# _build_mitigation_pieces gives each piece its OWN local n_prec-bit
# register (so fold()'s 0-indexed assumption is satisfied for both
# pieces), and _assemble_mitigated_pieces remaps piece 2's local clbits
# 0..n_prec-1 to the final circuit's n_prec..2*n_prec-1 explicitly.
# error_mitig_ionq.py itself is NOT modified (per constraints) -- this is
# worked around entirely in this file.

IDEAL_BINS_M4 = None  # set from m_j at runtime in run_mitigate


def _build_mitigation_pieces(n_prec: int, n_targ: int, U: np.ndarray,
                             target_init_seed: int, eigenstate: bool = False):
    """
    Like _build_submission_pieces (Step 3), but each piece gets its OWN
    LOCAL n_prec-bit classical register starting at clbit 0, instead of
    sharing the full 2*n_prec-bit register -- required for fold()'s
    _reattach_measurements to renumber correctly (see the section docstring
    above). _assemble_mitigated_pieces remaps piece 2's clbits back to the
    final register's upper half.
    """
    n_total = n_targ + 2 * n_prec
    q = QuantumRegister(n_total, "q")
    c1 = ClassicalRegister(n_prec, "c1")
    c2 = ClassicalRegister(n_prec, "c2")

    targ_idx = list(range(0, n_targ))
    prec_a_idx = list(range(n_targ, n_targ + n_prec))
    prec_b_idx = list(range(n_targ + n_prec, n_targ + 2 * n_prec))

    qc1 = QuantumCircuit(q, c1)
    if eigenstate:
        for i in targ_idx:
            qc1.x(q[i])
    else:
        init_rng = np.random.default_rng(seed=target_init_seed)
        for i in targ_idx:
            qc1.ry(init_rng.uniform(0, np.pi), q[i])
            qc1.rz(init_rng.uniform(0, 2 * np.pi), q[i])
    qpe1 = build_qpe_stage(n_prec, U)
    qc1.append(qpe1, prec_a_idx + targ_idx)
    qc1.measure([q[i] for i in prec_a_idx], [c1[k] for k in range(n_prec)])

    qc2 = QuantumCircuit(q, c2)
    qpe2 = build_qpe_stage(n_prec, U)
    qc2.append(qpe2, prec_b_idx + targ_idx)
    qc2.measure([q[i] for i in prec_b_idx], [c2[k] for k in range(n_prec)])

    return qc1, qc2


def _assemble_mitigated_pieces(piece1: QuantumCircuit, piece2: QuantumCircuit,
                               n_total: int, n_prec: int) -> QuantumCircuit:
    """
    Program-order concatenation (piece 1 entirely, then piece 2 entirely)
    into a fresh flat-register circuit, remapping each piece's LOCAL clbits
    (0..n_prec-1) to the final circuit's clbits -- piece 1 unchanged
    (0..n_prec-1), piece 2 offset by n_prec (n_prec..2*n_prec-1). No
    transpile() call, ever, in or after this function (Finding 2).
    """
    q = QuantumRegister(n_total, "q")
    c = ClassicalRegister(2 * n_prec, "c")
    assembled = QuantumCircuit(q, c)
    for piece, clbit_offset in ((piece1, 0), (piece2, n_prec)):
        for instr in piece.data:
            qubit_idx = [piece.find_bit(qb).index for qb in instr.qubits]
            clbit_idx = [piece.find_bit(cb).index + clbit_offset for cb in instr.clbits]
            assembled.append(instr.operation,
                            [q[i] for i in qubit_idx],
                            [c[i] for i in clbit_idx])
    return assembled


def _check_program_order(qc: QuantumCircuit, n_targ: int, n_prec: int,
                         label: str = "") -> bool:
    """
    Step 3's QASM-text check, generalized to a Python QuantumCircuit object
    (no export needed): confirm the LAST stage-1 measure instruction
    (clbit < n_prec) occurs BEFORE the FIRST stage-2 instruction (anything
    touching a prec_b qubit, or any measure with clbit >= n_prec). Run this
    on every arm's final assembled circuit BEFORE simulating it.
    """
    prec_b_qubits = set(range(n_targ + n_prec, n_targ + 2 * n_prec))
    last_stage1_measure_idx = None
    first_stage2_idx = None
    for i, instr in enumerate(qc.data):
        qidx = [qc.find_bit(q).index for q in instr.qubits]
        cidx = [qc.find_bit(c).index for c in instr.clbits]
        is_stage1_measure = (instr.operation.name == "measure" and cidx
                            and cidx[0] < n_prec)
        is_stage2 = (any(q in prec_b_qubits for q in qidx)
                    or (instr.operation.name == "measure" and cidx and cidx[0] >= n_prec))
        if is_stage1_measure:
            last_stage1_measure_idx = i
        if is_stage2 and first_stage2_idx is None:
            first_stage2_idx = i
    ok = (last_stage1_measure_idx is not None and first_stage2_idx is not None
         and last_stage1_measure_idx < first_stage2_idx)
    tag = f" [{label}]" if label else ""
    print(f"[program-order check{tag}] last stage-1 measure @ idx "
          f"{last_stage1_measure_idx}, first stage-2 instruction @ idx "
          f"{first_stage2_idx}  -- {'OK' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(f"program-order check FAILED for arm {label!r} -- "
                          "do not trust this arm's simulated counts.")
    return ok


def _sim_run_notranspile(circ: QuantumCircuit, nm, shots: int, seed: int) -> dict:
    """
    Direct AerSimulator execution, NO transpile() call (Finding 2): every
    circuit reaching this function is already transpiled into the exact
    target basis (rz/rx/rxx/id) piece-by-piece before assembly, so there is
    nothing left for a transpile pass to do except reorder instructions --
    which is exactly what must not happen to an already program-order-
    correct assembled circuit.
    """
    from qiskit_aer import AerSimulator
    sim = AerSimulator(noise_model=nm, basis_gates=["rz", "rx", "rxx", "id"])
    return sim.run(circ, shots=shots, seed_simulator=seed).result().get_counts()


def _make_observables(ideal_bins) -> dict:
    """
    4 linear functionals of the 256-dim (m1,m2) probability vector
    (idx = m1*16+m2, i.e. m1 = idx>>4, m2 = idx&15) -- each a sum over a
    fixed index subset, exactly the "expectation value of a projector"
    shape that justifies ZNE extrapolation (mitig_analysis.p_correct's
    docstring makes the same argument for the single-bin case).
    """
    ideal_set = set(int(b) for b in ideal_bins)
    idx = np.arange(256)
    m1, m2 = idx >> 4, idx & 15
    diff = np.abs(m1 - m2) % 16
    dist = np.minimum(diff, 16 - diff)
    mask_eq = (m1 == m2)
    mask_m1_ideal = np.isin(m1, list(ideal_set))
    mask_m2_ideal = np.isin(m2, list(ideal_set))
    mask_dist_le1 = (dist <= 1)
    return {
        "p_m1_eq_m2":    lambda p: float(np.sum(p[mask_eq])),
        "p_m1_in_ideal": lambda p: float(np.sum(p[mask_m1_ideal])),
        "p_m2_in_ideal": lambda p: float(np.sum(p[mask_m2_ideal])),
        "p_dist_le1":    lambda p: float(np.sum(p[mask_dist_le1])),
    }


def _p256(counts: dict) -> np.ndarray:
    """counts -> 256-dim probability vector via the qiskit convention,
    n_prec=8 treating the full 8-clbit outcome as one combined integer
    (m_full = m1*16+m2 -- see the decode cross-check in run_mitigate)."""
    return to_probabilities(counts_to_m(counts, 8, convention="qiskit"), 8)


# ---- Bootstrap wrappers for a general observable (mirror mitig_analysis.py's
# resampling discipline exactly; only the scalar-extraction step changes from
# p_correct(p, ideal_bin) to obs_fn(p)) --------------------------------------

def bootstrap_delta_unpaired_obs(counts_a: dict, counts_b: dict, obs_fn,
                                 n_boot: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a_s, b_s, deltas = [], [], []
    for _ in range(n_boot):
        pa = _p256(resample_counts(counts_a, rng))
        pb = _p256(resample_counts(counts_b, rng))
        va, vb = obs_fn(pa), obs_fn(pb)
        a_s.append(va); b_s.append(vb); deltas.append(vb - va)
    ci = bootstrap_ci(deltas)
    ci["significant"] = bool(ci["lo"] is not None and (ci["lo"] > 0 or ci["hi"] < 0))
    return {"a": bootstrap_ci(a_s), "b": bootstrap_ci(b_s), "delta": ci}


def bootstrap_rem_paired_obs(raw_counts: dict, cal0: dict, cal1: dict, obs_fn,
                             n_boot: int = 500, method: str = "simplex",
                             seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    raw_s, rem_s, delta_s, conds = [], [], [], []
    for _ in range(n_boot):
        r  = resample_counts(raw_counts, rng)
        c0 = resample_counts(cal0, rng)
        c1 = resample_counts(cal1, rng)
        p_raw = _p256(r)
        mats  = confusion_matrices(c0, c1, 8, convention="qiskit")
        a     = tensored_confusion(mats)
        p_rem = rem_correct(p_raw, a, method=method)
        v_raw, v_rem = obs_fn(p_raw), obs_fn(p_rem)
        raw_s.append(v_raw); rem_s.append(v_rem); delta_s.append(v_rem - v_raw)
        conds.append(condition_number(a))
    return {"raw": bootstrap_ci(raw_s), "rem": bootstrap_ci(rem_s),
            "delta": bootstrap_ci(delta_s), "condition_number": bootstrap_ci(conds)}


def bootstrap_zne_obs(counts_by_scale: dict, obs_fn, n_boot: int = 500,
                      seed: int = 0,
                      models=("richardson", "linear", "exponential")) -> dict:
    rng = np.random.default_rng(seed)
    scales = sorted(counts_by_scale)
    base_idx = scales.index(1.0)
    per_model = {m: [] for m in models}
    delta_per_model = {m: [] for m in models}
    for _ in range(n_boot):
        vals = [obs_fn(_p256(resample_counts(counts_by_scale[s], rng))) for s in scales]
        fit = extrapolate(scales, vals, models=models)
        baseline = vals[base_idx]
        for m in models:
            per_model[m].append(fit.get(m))
            v = fit.get(m)
            delta_per_model[m].append(None if v is None else v - baseline)
    baseline_point = obs_fn(_p256(counts_by_scale[1.0]))
    point = extrapolate(scales, [obs_fn(_p256(counts_by_scale[s])) for s in scales],
                        models=models)
    delta_point = {m: (None if point.get(m) is None else point[m] - baseline_point)
                  for m in models}
    return {"point_estimate": point, "ci": {m: bootstrap_ci(v) for m, v in per_model.items()},
            "delta_point_estimate": delta_point,
            "delta_ci": {m: bootstrap_ci(v) for m, v in delta_per_model.items()}}


def bootstrap_zne_rem_obs(counts_by_scale: dict, cal0: dict, cal1: dict, obs_fn,
                          n_boot: int = 500, seed: int = 0,
                          models=("richardson", "linear", "exponential"),
                          method: str = "simplex") -> dict:
    rng = np.random.default_rng(seed)
    scales = sorted(counts_by_scale)
    base_idx = scales.index(1.0)
    per_model = {m: [] for m in models}
    delta_per_model = {m: [] for m in models}
    for _ in range(n_boot):
        a = tensored_confusion(confusion_matrices(resample_counts(cal0, rng),
                                                  resample_counts(cal1, rng),
                                                  8, convention="qiskit"))
        vals, baseline_raw = [], None
        for i, s in enumerate(scales):
            p_raw = _p256(resample_counts(counts_by_scale[s], rng))
            vals.append(obs_fn(rem_correct(p_raw, a, method=method)))
            if i == base_idx:
                baseline_raw = obs_fn(p_raw)
        fit = extrapolate(scales, vals, models=models)
        for m in models:
            per_model[m].append(fit.get(m))
            v = fit.get(m)
            delta_per_model[m].append(None if v is None else v - baseline_raw)
    a_point = tensored_confusion(confusion_matrices(cal0, cal1, 8, convention="qiskit"))
    vals_point, baseline_point_raw = [], None
    for i, s in enumerate(scales):
        p_raw = _p256(counts_by_scale[s])
        vals_point.append(obs_fn(rem_correct(p_raw, a_point, method=method)))
        if i == base_idx:
            baseline_point_raw = obs_fn(p_raw)
    point = extrapolate(scales, vals_point, models=models)
    delta_point = {m: (None if point.get(m) is None else point[m] - baseline_point_raw)
                  for m in models}
    return {"point_estimate": point, "ci": {m: bootstrap_ci(v) for m, v in per_model.items()},
            "delta_point_estimate": delta_point,
            "delta_ci": {m: bootstrap_ci(v) for m, v in delta_per_model.items()}}


def run_mitigate(args) -> None:
    import matplotlib.pyplot as plt

    n_targ, n_prec = 3, 4
    n_total = n_targ + 2 * n_prec
    rng = np.random.default_rng(seed=args.seed)
    U, m_j = random_diagonal_on_grid_unitary(n_targ, n_prec, rng, return_m_j=True)
    ideal_bins = sorted(set(m_j))
    observables = _make_observables(ideal_bins)
    print(f"seed={args.seed}  m_j={m_j}  ideal_bins={ideal_bins}")

    NOM  = dict(f_1q=0.9998, f_2q=0.9904, p_readout=0.0036)   # today's live device-query values
    COMP = dict(f_1q=0.9998, f_2q=0.9882, p_readout=0.0036)   # Step 2's original F2Q, for comparison
    SHOTS = 10000
    DETUNING_RAD_PER_S = 2.0e5  # same ballpark as mitig_analysis.validate()'s default
    ZNE_SCALES = (1, 1.5, 2)

    seeds_used = {}
    def sim(circ, nm, name, seed_val):
        seeds_used[name] = seed_val
        return _sim_run_notranspile(circ, nm, SHOTS, seed_val)

    em = ErrorMitigation()
    nm_nominal = build_noise_model(**NOM)
    nm_comp    = build_noise_model(**COMP)

    def fresh_pieces():
        qc1, qc2 = _build_mitigation_pieces(n_prec, n_targ, U, TARGET_INIT_SEED,
                                            eigenstate=False)
        qc1_t = transpile(qc1, basis_gates=['rz', 'rx', 'rxx', 'id'], optimization_level=0)
        qc2_t = transpile(qc2, basis_gates=['rz', 'rx', 'rxx', 'id'], optimization_level=0)
        return qc1_t, qc2_t

    gate_counts = {}

    def assemble_check_count(piece1, piece2, label):
        assembled = _assemble_mitigated_pieces(piece1, piece2, n_total, n_prec)
        _check_program_order(assembled, n_targ, n_prec, label=label)
        prof = imc.count_native(assembled)
        # Report, don't hard-stop: this is simulation-only, and the Director
        # wants the counts even where they'd fail a real submission (e.g.
        # ZNE folds at scale>=1.5 -- ionq_mitigation_common.py's own
        # documented forward-looking warning about high-n_targ families
        # crossing 2,000 around lambda~1.5). Still call assert_gate_count_ok
        # so the check itself (and its exact message) is exercised, just
        # catch it instead of letting it abort the whole arm matrix.
        try:
            imc.assert_gate_count_ok(assembled, label=label)
            under_limit = True
        except ValueError as e:
            print(f"  GATE GUARD: {e}")
            under_limit = False
        gate_counts[label] = {"n_1q": prof["n_1q"], "n_2q": prof["n_2q"],
                              "total": prof["n_gates"], "under_2000_limit": under_limit}
        return assembled

    # ---- RAW arm (nominal + comparison F2Q) --------------------------------
    qc1_t, qc2_t = fresh_pieces()
    assembled_raw = assemble_check_count(qc1_t, qc2_t, "raw")
    counts_raw = sim(assembled_raw, nm_nominal, "raw", 301)
    counts_raw_comp = sim(assembled_raw, nm_comp, "raw_f2q_0.9882", 302)

    # ---- Decode cross-check (mandatory, before trusting anything else) ----
    m_full_via_countstom = counts_to_m(counts_raw, 8, convention="qiskit")
    mismatches = []
    for bitstring in counts_raw:
        m1_db, m2_db = decode_bits(bitstring, n_prec)
        m_full = m1_db * 16 + m2_db
        # decode_bits already gives (m1,m2); counts_to_m(n_prec=8) gives the
        # combined integer for the SAME bitstring directly from its own counts
        # dict -- recompute it standalone (not by re-deriving from decode_bits)
        # for a genuine independent cross-check.
        single = counts_to_m({bitstring: 1}, 8, convention="qiskit")
        m_full_cm = next(iter(single))
        m1_cm, m2_cm = m_full_cm >> 4, m_full_cm & 15
        if (m1_db, m2_db) != (m1_cm, m2_cm):
            mismatches.append((bitstring, (m1_db, m2_db), (m1_cm, m2_cm)))
    print(f"\n[decode cross-check] decode_bits vs counts_to_m(n_prec=8): "
         f"{len(counts_raw) - len(mismatches)}/{len(counts_raw)} bitstrings agree")
    if mismatches:
        print("  MISMATCHES:", mismatches[:10])
        raise RuntimeError(
            "decode_bits and counts_to_m(n_prec=8, convention='qiskit') "
            "DISAGREE on the observed sample -- stopping per instructions "
            "rather than picking one convention silently."
        )
    print("  decode cross-check PASSED -- both decoders agree on every "
         "observed bitstring.")

    # ---- REM calibration (on the assembled RAW circuit) --------------------
    cal = em.rem_calibration_circuits(assembled_raw, model="tensored")
    cal_counts = {}
    for i, (label, circ) in enumerate(cal):
        cal_counts[label] = sim(circ, nm_nominal, f"rem_{label}", 310 + i)
    mats = confusion_matrices(cal_counts["cal0"], cal_counts["cal1"], 8, convention="qiskit")
    a_rem = tensored_confusion(mats)
    cond = condition_number(a_rem)
    print(f"\nREM confusion matrix condition number: {cond:.3f}")

    # ---- ZNE ladder ---------------------------------------------------------
    counts_by_scale = {}
    for i, s in enumerate(ZNE_SCALES):
        f1 = em.fold(qc1_t, s, method="random", seed=0)
        f2 = em.fold(qc2_t, s, method="random", seed=0)
        assembled_s = assemble_check_count(f1, f2, f"zne_scale_{s}")
        counts_by_scale[float(s)] = sim(assembled_s, nm_nominal, f"zne_{s}", 320 + i)

    # ---- DDD ------------------------------------------------------------
    # ddd_off_detuned: no DDD pulses, WITH the coherent idle-error injected
    # -- the baseline DDD is actually compared against (mirrors
    # mitig_analysis.validate()'s "none" baseline, which is ALSO
    # detuning-injected: DDD's effect must be measured against a baseline
    # that has the SAME coherent noise, not a plain-Markovian raw arm, or
    # the comparison tests nothing about what DDD is for).
    qc1_detuned = inject_coherent_idle_error(qc1_t, em, DETUNING_RAD_PER_S)
    qc2_detuned = inject_coherent_idle_error(qc2_t, em, DETUNING_RAD_PER_S)
    assembled_ddoff_detuned = assemble_check_count(qc1_detuned, qc2_detuned, "ddd_off_detuned")
    counts_ddoff_detuned = sim(assembled_ddoff_detuned, nm_nominal, "ddd_off_detuned", 330)

    # ddd_on_plain (Director's "no injected error" DDD arm): DDD pulses,
    # pure Markovian (depolarizing+readout) noise only -- no coherent drift
    # for the pulses to have anything to refocus against.
    dd1 = em.apply_ddd(qc1_t, sequence="XY4")
    dd2 = em.apply_ddd(qc2_t, sequence="XY4")
    assembled_ddon_plain = assemble_check_count(dd1, dd2, "ddd_on_plain")
    counts_ddon_plain = sim(assembled_ddon_plain, nm_nominal, "ddd_on_plain", 331)

    # ddd_on_detuned (Director's "with injected coherent idle error" arm):
    # DDD pulses AND the coherent drift -- the genuine test of whether the
    # pulses refocus it. Inject AFTER DDD (mitig_analysis docstring: a
    # decoupled window is split into cancelling sub-windows only if the
    # phase is injected after the pulses that are supposed to cancel it).
    dd1_detuned = inject_coherent_idle_error(dd1, em, DETUNING_RAD_PER_S)
    dd2_detuned = inject_coherent_idle_error(dd2, em, DETUNING_RAD_PER_S)
    assembled_ddon_detuned = assemble_check_count(dd1_detuned, dd2_detuned, "ddd_on_detuned")
    counts_ddon_detuned = sim(assembled_ddon_detuned, nm_nominal, "ddd_on_detuned", 332)

    # =========================================================================
    # Metrics
    # =========================================================================
    N_BOOT = 500
    results = {}

    def obs_table(counts, label):
        p = _p256(counts)
        return {name: fn(p) for name, fn in observables.items()}

    for label, counts in (("raw", counts_raw), ("raw_f2q_0.9882", counts_raw_comp),
                          ("ddd_off_detuned", counts_ddoff_detuned),
                          ("ddd_on_plain", counts_ddon_plain),
                          ("ddd_on_detuned", counts_ddon_detuned)):
        results[label] = {"point": obs_table(counts, label)}

    # REM point + paired bootstrap, per observable
    p_raw_256 = _p256(counts_raw)
    p_rem_256 = rem_correct(p_raw_256, a_rem, method="simplex")
    results["rem"] = {"point": {name: fn(p_rem_256) for name, fn in observables.items()}}
    rem_boot = {name: bootstrap_rem_paired_obs(counts_raw, cal_counts["cal0"],
                                               cal_counts["cal1"], fn,
                                               n_boot=N_BOOT, seed=400)
               for name, fn in observables.items()}

    # ZNE point (per model) + bootstrap, per observable
    zne_fits, zne_boot = {}, {}
    for name, fn in observables.items():
        vals = [fn(_p256(counts_by_scale[s])) for s in ZNE_SCALES]
        zne_fits[name] = extrapolate(list(ZNE_SCALES), vals)
        zne_boot[name] = bootstrap_zne_obs(counts_by_scale, fn, n_boot=N_BOOT, seed=401)

    # ZNE+REM point + bootstrap, per observable
    znerem_fits, znerem_boot = {}, {}
    for name, fn in observables.items():
        vals = [fn(rem_correct(_p256(counts_by_scale[s]), a_rem, method="simplex"))
               for s in ZNE_SCALES]
        znerem_fits[name] = extrapolate(list(ZNE_SCALES), vals)
        znerem_boot[name] = bootstrap_zne_rem_obs(counts_by_scale, cal_counts["cal0"],
                                                  cal_counts["cal1"], fn,
                                                  n_boot=N_BOOT, seed=402)

    # DDD unpaired deltas, per observable
    ddd_delta = {name: bootstrap_delta_unpaired_obs(counts_ddoff_detuned, counts_ddon_detuned,
                                                    fn, n_boot=N_BOOT, seed=403)
                for name, fn in observables.items()}

    # ---- Print summary table ------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'arm':<20}{'P(m1==m2)':>18}{'P(m1 in ideal)':>18}{'P(m2 in ideal)':>18}{'P(dist<=1)':>18}")
    print("-" * 100)
    for label in ("raw", "raw_f2q_0.9882", "ddd_off_detuned", "ddd_on_plain", "ddd_on_detuned", "rem"):
        row = results[label]["point"]
        print(f"{label:<20}" + "".join(f"{row[k]:>18.4f}" for k in
                                       ("p_m1_eq_m2", "p_m1_in_ideal", "p_m2_in_ideal", "p_dist_le1")))
    for name in observables:
        print(f"zne[{name}] linear extrap = {zne_fits[name].get('linear'):.4f}   "
             f"model_spread={zne_fits[name].get('model_spread', float('nan')):.4f}   "
             f"unphysical={zne_fits[name].get('unphysical')}")
        print(f"zne+rem[{name}] linear extrap = {znerem_fits[name].get('linear'):.4f}   "
             f"model_spread={znerem_fits[name].get('model_spread', float('nan')):.4f}")

    print(f"\nGate counts vs {imc.MAX_GATES_PER_CIRCUIT}-gate limit:")
    for label, gc in gate_counts.items():
        print(f"  {label:<18} 1q={gc['n_1q']:5d}  2q={gc['n_2q']:5d}  "
             f"total={gc['total']:5d}  "
             f"({'UNDER' if gc['total'] <= imc.MAX_GATES_PER_CIRCUIT else 'OVER'})")

    print(f"\nseed_simulator values used: {seeds_used}")

    # ---- Save ----------------------------------------------------------------
    out_dir = os.path.join(RESULTS_ROOT, args.date, "mitigate")
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "seed": args.seed, "m_j": m_j, "ideal_bins": ideal_bins,
        "nominal_noise": NOM, "comparison_noise": COMP, "shots": SHOTS,
        "detuning_rad_per_s": DETUNING_RAD_PER_S, "zne_scales": list(ZNE_SCALES),
        "seeds_used": seeds_used, "gate_counts": gate_counts,
        "decode_cross_check": {"n_bitstrings": len(counts_raw), "n_mismatches": len(mismatches)},
        "rem_condition_number": cond,
        "results_point": {k: v["point"] for k, v in results.items()},
        "rem_bootstrap": rem_boot, "zne_fits": zne_fits, "zne_bootstrap": zne_boot,
        "znerem_fits": znerem_fits, "znerem_bootstrap": znerem_boot,
        "ddd_delta_bootstrap": ddd_delta,
    }
    out_json = os.path.join(out_dir, "mitigate_results.json")
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nwrote {out_json}")

    # ---- Plot -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    plot_m1_vs_m2(counts_raw, n_prec, m_j, SHOTS, title_suffix="raw (nominal)", ax=axes[0, 0])
    # REM has no "counts" of its own (it's a post-processing correction on
    # the raw counts' probability vector) -- show the raw calibration-
    # corrected marginal isn't a scatter of counts, so reuse the raw counts
    # panel's scatter but label it clearly as REM's INPUT, and annotate the
    # point-estimate table entry instead for REM's actual effect.
    plot_m1_vs_m2(counts_raw, n_prec, m_j, SHOTS,
                 title_suffix=f"REM input (raw) -- P(m1==m2) rem={results['rem']['point']['p_m1_eq_m2']:.3f}",
                 ax=axes[0, 1])
    plot_m1_vs_m2(counts_by_scale[2.0], n_prec, m_j, SHOTS, title_suffix="ZNE scale=2 (raw fold)", ax=axes[0, 2])
    plot_m1_vs_m2(counts_ddoff_detuned, n_prec, m_j, SHOTS, title_suffix="DDD off (detuned)", ax=axes[1, 0])
    plot_m1_vs_m2(counts_ddon_plain, n_prec, m_j, SHOTS, title_suffix="DDD on (no injected error)", ax=axes[1, 1])
    plot_m1_vs_m2(counts_ddon_detuned, n_prec, m_j, SHOTS, title_suffix="DDD on (detuned)", ax=axes[1, 2])
    fig.suptitle(f"Step 4 mitigation arms, seed={args.seed}, m_j={m_j}, {SHOTS} shots each\n"
                f"ZNE+REM panel not separately scattered (post-processing on the 256-dim "
                f"vector, not a circuit) -- see mitigate_results.json for its numbers",
                fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(out_dir, "on_grid_mitigation_circle_plots.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["noiseless", "noisy", "submit", "retrieve", "mitigate"],
                    default="noiseless")
    ap.add_argument("--seed", type=int, default=9,
                    help="RNG seed for random_diagonal_on_grid_unitary's m_j draw")
    ap.add_argument("--date", default="2026-09-08",
                    help="date-stamped results subdirectory under on_grid_results/")
    ap.add_argument("--dry-run", action="store_true",
                    help="--stage submit: explicit dry run (also the default "
                         "with no flags). Always wins over "
                         "--i-have-yons-go-ahead if both are given.")
    ap.add_argument("--i-have-yons-go-ahead", action="store_true",
                    help="--stage submit: required (but not sufficient -- an "
                         "interactive y/N confirmation follows) to attempt a "
                         "real submission. Absent by default.")
    ap.add_argument("--arn", default=None,
                    help="--stage retrieve: the Braket task ARN to fetch.")
    args = ap.parse_args()

    if args.stage == "noiseless":
        run_noiseless(args)
    elif args.stage == "noisy":
        run_noisy(args)
    elif args.stage == "submit":
        run_submit(args)
    elif args.stage == "retrieve":
        run_retrieve(args)
    elif args.stage == "mitigate":
        run_mitigate(args)


if __name__ == "__main__":
    main()
