#!/usr/bin/env python3
"""
two_stage_on_grid_baseline.py
==============================
NOISELESS-ONLY. No AWS/Braket imports anywhere in this file, no
device.run(), no hardware submission of any kind.

Builds the on-grid two-stage QPE-QPUF instances (every eigenphase of U an
exact multiple of 1/2^n_prec -- see `on_grid_unitary` in
submit_QPUF_ntarg.py) and confirms the noiseless P(m1==m2)=1 baseline that
motivates them: for Haar-random U, P(m1,m2) is a mixture over independent
single-stage QPE draws on each eigenvector, and off-grid eigenphases are
the only source of m1 != m2 noiselessly (see `on_grid_unitary`'s
docstring). Putting every eigenphase on the grid removes that leakage
entirely, giving a clean P=1 reference: any deviation measured on real
hardware is then attributable to noise, not to this effect.

This script only builds/simulates locally (branch-exact statevector via
`ideal_joint`, plus one local qiskit-primitives/Aer sampled cross-check).
It never touches a real device.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from submit_QPUF_ntarg import (
    on_grid_unitary, rotated_on_grid_unitary, product_rotated_on_grid_unitary,
    build_full_circuit, build_qpe_stage, build_qpe_stage_product,
    haar_random_unitary, _stable_matrix_power,
)
from decode_mcm_qpuf_result import ideal_joint, circular_distance_pmf

TARGET_INIT_SEED = 99  # reused from decode_mcm_qpuf_result.py's module default


# =========================================================================
# Helpers
# =========================================================================

def extract_ks(D: np.ndarray, n_targ: int, n_prec: int) -> tuple:
    """
    Recover the integer k_i's from a diagonal on-grid U's phase on the
    single-target-qubit-set basis states |0..010..0> (qubit i set): that
    state's phase is exp(2*pi*i*k_i/2^n_prec) by construction. Reading
    ks back this way (instead of re-drawing them) avoids duplicating
    on_grid_unitary's rng-consuming draw logic just to report a label.
    """
    d = 2 ** n_prec
    ks = []
    for i in range(n_targ):
        j = 1 << i
        phi = float(np.angle(D[j, j]))
        k = int(round(phi * d / (2 * np.pi))) % d
        ks.append(k)
    return tuple(ks)


def decode_bits(b: str, n_prec: int) -> tuple:
    """
    Decode one qiskit-convention measurement bitstring `b` (single
    ClassicalRegister "c" of size 2*n_prec, as build_full_circuit emits)
    into (m1, m2), MSB-first per build_qpe_stage's convention.

    Qiskit orders bitstrings with the leftmost char = highest classical
    bit index (c[2*n_prec-1]) and the rightmost char = c[0] (confirmed
    empirically: x(q0); measure(q0,c0) alone yields bitstring "...0001").
    build_full_circuit maps c[0..n_prec-1] = stage 1 (c[0] = MSB of m1)
    and c[n_prec..2n_prec-1] = stage 2 (c[n_prec] = MSB of m2). Working
    through the index algebra: c[k] = b[2*n_prec-1-k], so the trailing
    n_prec chars of b, REVERSED, are c[0..n_prec-1] MSB-first = m1; the
    leading n_prec chars, reversed, are m2.
    """
    L = 2 * n_prec
    s1 = b[n_prec:L]
    s2 = b[0:n_prec]
    m1 = int(s1[::-1], 2)
    m2 = int(s2[::-1], 2)
    return m1, m2


def check_stable_power_exactness(D: np.ndarray, n_prec: int):
    """
    _stable_matrix_power SVD-reprojects every matrix power for numerical
    hygiene at large Haar powers (error would otherwise double per
    squaring). For a diagonal U the exact power is closed-form
    (diag(phases**p)), so rather than adding a diagonal special-case to
    a routine whose SVD reprojection is load-bearing for generic
    (non-diagonal) U elsewhere, we just confirm here that reprojection
    introduces no measurable extra error in the diagonal case.
    """
    max_power = 2 ** (n_prec - 1)  # largest controlled-power exponent build_qpe_stage uses
    phases = np.diagonal(D)
    exact = np.diag(phases ** max_power)
    approx = _stable_matrix_power(D, max_power)
    err = float(np.max(np.abs(exact - approx)))
    print(f"  _stable_matrix_power exactness (diagonal U, power=2^{n_prec-1}): "
          f"max|exact-approx| = {err:.2e}  (expect <1e-12)")
    assert err < 1e-12, "SVD-reprojected diagonal power drifted from exact closed form"


def product_rotated_eigenstate_joint(n_targ: int, n_prec: int, ks: tuple,
                                     seed: int) -> np.ndarray:
    """
    Branch-exact P(m1,m2) for product_rotated_on_grid_unitary's eigenstate
    target V|1..1> = tensor_i v_i|1> -- NOT |1..1> itself, which is only an
    eigenvector of the diagonal on-grid U, not this variant's generic
    product eigenbasis.

    Standalone rather than reusing/extending ideal_joint or
    build_full_circuit's eigenstate=True path: both only know how to
    prepare |1..1>, and adding a per-qubit-unitary override to either just
    for this one verification call isn't worth the extra surface on
    builders several other scripts depend on. Duplicates ideal_joint's
    branch-exact projection logic (H->QPE1->exact MCM projection->QPE2)
    with a different target-prep step; the MCM-projection logic itself is
    copied unmodified from there, not reinvented.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed=seed)
    U = product_rotated_on_grid_unitary(n_targ, n_prec, rng=rng, ks=ks)

    # Replay product_rotated_on_grid_unitary's v_i draw order (documented in
    # its docstring: with ks explicit, its only rng draws are
    # haar_random_unitary(2, rng) for i=0..n_targ-1, in order) from a fresh
    # same-seed Generator, to recover the v_i's without that function
    # needing to return them separately.
    rng_replay = np.random.default_rng(seed=seed)
    v_list = [haar_random_unitary(2, rng=rng_replay) for _ in range(n_targ)]

    n_total = n_targ + 2 * n_prec
    targ_idx = list(range(0, n_targ))
    prec_a_idx = list(range(n_targ, n_targ + n_prec))
    prec_b_idx = list(range(n_targ + n_prec, n_targ + 2 * n_prec))

    q = QuantumRegister(n_total, "q")
    qc1 = QuantumCircuit(q)
    for i in targ_idx:
        qc1.x(q[i])
        qc1.append(UnitaryGate(v_list[i]), [q[i]])
    qpe1 = build_qpe_stage(n_prec, U)
    qc1.append(qpe1, prec_a_idx + targ_idx)
    sv1 = Statevector.from_instruction(qc1)
    qpe2 = build_qpe_stage(n_prec, U)

    d = 2 ** n_prec
    joint = np.zeros((d, d))
    idx = np.arange(sv1.dim)
    for m1 in range(d):
        bits = [(m1 >> (n_prec - 1 - j)) & 1 for j in range(n_prec)]
        mask = np.ones_like(idx, dtype=bool)
        for j, qb in enumerate(prec_a_idx):
            mask &= ((idx >> qb) & 1) == bits[j]
        amp = np.where(mask, sv1.data, 0.0)
        prob_m1 = float(np.sum(np.abs(amp) ** 2))
        if prob_m1 < 1e-14:
            continue
        proj = Statevector(amp / np.sqrt(prob_m1))
        sv2 = proj.evolve(qpe2, qargs=prec_b_idx + targ_idx)
        probs_m2 = sv2.probabilities(qargs=prec_b_idx[::-1])
        joint[m1, :] = prob_m1 * probs_m2

    return joint


def check_known_answer_product(n_prec: int = 5):
    """
    Known-answer test on build_qpe_stage_product directly (single QPE
    stage, no MCM, single target qubit): factor = diag(1, exp(2*pi*i*k /
    2^n_prec)), target initialised to |1> -- the EXACT eigenvector of that
    factor with eigenphase k/2^n_prec. QPE on an exact eigenvector must
    return m=k with probability 1 for every k in [0, 2^n_prec), or the
    controlled-power ordering or inverse-QFT placement is wrong (CLAUDE.md
    known-bug regression list items 1-2): get the MSB-first control-power
    assignment wrong and the returned bin is some other permutation of k;
    get the inverse-QFT SWAP/reversal placement wrong and probability
    smears across bins instead of landing deterministically on one.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    d = 2 ** n_prec
    for k in range(d):
        factor = np.diag([1.0 + 0j, np.exp(2j * np.pi * k / d)])
        qpe = build_qpe_stage_product(n_prec, [factor])
        qc = QuantumCircuit(n_prec + 1)
        qc.x(n_prec)  # target = |1>, the eigenvector with eigenphase k/2^n_prec
        qc.append(qpe, range(n_prec + 1))
        sv = Statevector.from_instruction(qc)
        # Same reversed-qargs convention as ideal_joint: qargs[0] is the
        # LSB of the returned probability array, so reversing the
        # MSB-first precision-qubit order makes the array index equal m
        # directly.
        probs = sv.probabilities(qargs=list(range(n_prec))[::-1])
        m = int(np.argmax(probs))
        assert m == k and np.isclose(probs[m], 1.0, atol=1e-9), (
            f"build_qpe_stage_product known-answer test failed: k={k} "
            f"decoded as m={m} (peak prob {probs[m]:.6f})"
        )
    print(f"  known-answer test (build_qpe_stage_product, n_prec={n_prec}): "
          f"all {d} k in [0, 2^{n_prec}) decode to m=k with probability 1 -- "
          f"MSB-first control-power ordering and inverse-QFT placement confirmed.")


def ideal_joint_product(n_prec: int, n_targ: int, factors: list, eigenstate: bool,
                        target_init_seed: int = TARGET_INIT_SEED) -> np.ndarray:
    """
    Branch-exact P(m1,m2) for the factorized path, driven through the SAME
    build_full_circuit(..., factors=...) used for real submission/sampling
    -- unlike ideal_joint (which independently re-builds init+qpe by hand),
    a bug specific to build_full_circuit's factors branch would show up
    here too.

    Statevector.from_instruction can't process measure() instructions
    directly, so build_full_circuit's own instruction list is split at its
    two measurement blocks (init+qpe1 before the first measure; qpe2
    between the two measurement blocks) and the same exact-branch
    projection ideal_joint uses is applied to those SAME instructions,
    instead of hand-building a third parallel circuit.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc_full = build_full_circuit(n_prec, n_targ, None, target_init_seed=target_init_seed,
                                 eigenstate=eigenstate, factors=factors)
    n_total = n_targ + 2 * n_prec
    prec_a_idx = list(range(n_targ, n_targ + n_prec))
    prec_b_idx = list(range(n_targ + n_prec, n_targ + 2 * n_prec))

    meas_positions = [i for i, ci in enumerate(qc_full.data) if ci.operation.name == "measure"]
    stage1_data = qc_full.data[:meas_positions[0]]
    stage2_data = qc_full.data[meas_positions[n_prec - 1] + 1: meas_positions[n_prec]]

    def _sub_circuit(data):
        qc = QuantumCircuit(n_total)
        for ci in data:
            idxs = [qc_full.find_bit(qb).index for qb in ci.qubits]
            qc.append(ci.operation, idxs)
        return qc

    sv1 = Statevector.from_instruction(_sub_circuit(stage1_data))
    qc2 = _sub_circuit(stage2_data)

    d = 2 ** n_prec
    joint = np.zeros((d, d))
    idx = np.arange(sv1.dim)
    for m1 in range(d):
        bits = [(m1 >> (n_prec - 1 - j)) & 1 for j in range(n_prec)]
        mask = np.ones_like(idx, dtype=bool)
        for j, qb in enumerate(prec_a_idx):
            mask &= ((idx >> qb) & 1) == bits[j]
        amp = np.where(mask, sv1.data, 0.0)
        prob_m1 = float(np.sum(np.abs(amp) ** 2))
        if prob_m1 < 1e-14:
            continue
        proj = Statevector(amp / np.sqrt(prob_m1))
        sv2 = proj.evolve(qc2)
        probs_m2 = sv2.probabilities(qargs=prec_b_idx[::-1])
        joint[m1, :] = prob_m1 * probs_m2

    return joint


def run_factorized_joint_check(n_targ: int, n_prec: int, ks: tuple, seed: int):
    """
    P(m1==m2)=1.0 exact joint via the factorized path (build_qpe_stage_product,
    driven through build_full_circuit(..., factors=...)), for both the
    diagonal-factorized and product-rotated-factorized cases, both target
    types -- the factorized-path counterpart of run_instance's dense-U
    checks, confirming the additive factors=... branch reproduces the same
    noiseless P=1 baseline as the dense path it's meant to replace.
    """
    print("\n" + "=" * 78)
    print(f"FACTORIZED-PATH ideal joint check (n_targ={n_targ}, n_prec={n_prec}, "
          f"ks={ks}, seed={seed})")
    print("=" * 78)

    _, factors_diag = on_grid_unitary(n_targ, n_prec, ks=ks, return_factors=True)
    _, factors_pr = product_rotated_on_grid_unitary(
        n_targ, n_prec, rng=np.random.default_rng(seed=seed), ks=ks, return_factors=True)

    for u_label, factors in (("diagonal (factorized)", factors_diag),
                             ("product-rotated (factorized)", factors_pr)):
        for tgt_label, eigenstate in (("eigenstate target |1..1>", True),
                                      ("product-state target", False)):
            joint = ideal_joint_product(n_prec, n_targ, factors, eigenstate)
            p_diag = float(np.trace(joint)) / joint.sum()
            print(f"  {u_label}, {tgt_label}: P(m1==m2) = {p_diag:.4f}")
            assert np.isclose(p_diag, 1.0, atol=1e-9), (
                f"factorized path P(m1==m2) != 1 for {u_label}, {tgt_label}"
            )


# =========================================================================
# Per-instance reporting
# =========================================================================

def report_case(label: str, n_prec: int, n_targ: int, U: np.ndarray,
                eigenstate: bool, expected_bin: int | None):
    joint = ideal_joint(n_prec=n_prec, n_targ=n_targ,
                        target_init_seed=TARGET_INIT_SEED,
                        U=U, eigenstate=eigenstate)
    n = joint.sum()
    p_diag = float(np.trace(joint)) / n
    dist_counts = circular_distance_pmf(joint, d=2 ** n_prec)
    p_within1 = float(dist_counts[:2].sum()) / n
    marg1 = (joint.sum(axis=1) / n).tolist()

    print(f"\n{label}")
    print(f"  P(m1==m2)            = {p_diag:.4f}")
    print(f"  P(dist_d(m1,m2)<=1)  = {p_within1:.4f}")
    print(f"  m1 marginal          = {np.round(marg1, 4).tolist()}")
    if expected_bin is not None:
        argmax1 = int(np.argmax(marg1))
        print(f"  expected bin (sum k_i mod 2^n_prec) = {expected_bin}   "
              f"argmax(m1 marginal) = {argmax1}   "
              f"{'MATCH' if argmax1 == expected_bin else 'MISMATCH'}")
    return p_diag


def run_instance(instance_label: str, n_targ: int, n_prec: int, ks: tuple,
                 rot_seed: int):
    d = 2 ** n_prec
    expected_bin = sum(ks) % d
    D = on_grid_unitary(n_targ, n_prec, ks=ks)
    rng_v = np.random.default_rng(seed=rot_seed)
    R = rotated_on_grid_unitary(n_targ, n_prec, rng=rng_v, ks=ks)

    print("\n" + "=" * 78)
    print(f"INSTANCE: {instance_label}  (n_targ={n_targ}, n_prec={n_prec}, "
          f"ks={ks}, rotation seed={rot_seed})")
    print("=" * 78)

    check_stable_power_exactness(D, n_prec)

    for u_label, U, is_diag in (("diagonal U", D, True), ("rotated U", R, False)):
        for tgt_label, eigenstate in (("eigenstate target |1..1>", True),
                                      ("product-state target", False)):
            # |1..1> is a true eigenvector of U (with eigenphase sum(ks)) only
            # for the diagonal variant; conjugating by V rotates the eigenbasis
            # away from the computational basis, so |1..1> becomes a generic
            # superposition under rotated U and the expected-bin check does
            # not apply there (P(m1==m2)=1 still holds either way).
            eb = expected_bin if (eigenstate and is_diag) else None
            report_case(f"{u_label}, {tgt_label}", n_prec, n_targ, U,
                       eigenstate, eb)


# =========================================================================
# Sampled MCM cross-check (genuine mid-circuit measurement, local simulator)
# =========================================================================

def _confirm_statevector_sampler_mcm():
    """
    Standalone confirmation, run before trusting StatevectorSampler for a
    circuit containing an MCM: H on q0, measure q0->c0, then measure q0
    into c1 again. If the sampler correctly collapses the state after the
    first measurement, c0 and c1 must agree on every shot; if it instead
    draws both "measurements" independently from the pre-measurement
    superposition, they'd disagree on ~50% of shots.
    """
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    qc = QuantumCircuit(1, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(0, 1)
    sampler = StatevectorSampler()
    res = sampler.run([qc], shots=2000).result()[0]
    counts = res.data.c.get_counts()
    n = sum(counts.values())
    agree = sum(v for b, v in counts.items() if b[-1] == b[-2])
    frac = agree / n
    print(f"  StatevectorSampler MCM-collapse self-test: agreement fraction "
          f"= {frac:.4f} (expect 1.0000)")
    assert frac > 0.999, "StatevectorSampler did not honor MCM collapse"


def get_sampler():
    """
    Prefer AerSimulator; fall back to StatevectorSampler only after
    confirming it actually honors mid-circuit-measurement collapse
    (qiskit's primitives interface doesn't guarantee this across
    versions, so we don't assume it silently).
    """
    try:
        from qiskit_aer import AerSimulator
        print("Sampler: AerSimulator (qiskit_aer available).")
        return "aer", AerSimulator()
    except ImportError:
        print("Sampler: qiskit_aer not importable; falling back to "
              "StatevectorSampler after confirming MCM collapse.")
        _confirm_statevector_sampler_mcm()
        from qiskit.primitives import StatevectorSampler
        return "statevector_sampler", StatevectorSampler()


def sample_agreement_fraction(kind: str, sampler, qc, n_prec: int, shots: int) -> float:
    if kind == "aer":
        from qiskit import transpile
        qc_t = transpile(qc, basis_gates=['rz', 'rx', 'rxx', 'measure'],
                         optimization_level=1)
        result = sampler.run(qc_t, shots=shots).result()
        counts = result.get_counts()
    else:
        result = sampler.run([qc], shots=shots).result()[0]
        counts = result.data.c.get_counts()

    total = sum(counts.values())
    agree = 0
    for b, n in counts.items():
        m1, m2 = decode_bits(b, n_prec)
        if m1 == m2:
            agree += n
    return agree / total


def run_sampled_check(n_targ: int, n_prec: int, ks: tuple, shots: int = 4000):
    print("\n" + "=" * 78)
    print(f"SAMPLED MCM CROSS-CHECK (diagonal U variant, {shots} shots)")
    print("=" * 78)
    D = on_grid_unitary(n_targ, n_prec, ks=ks)
    kind, sampler = get_sampler()
    for tgt_label, eigenstate in (("eigenstate target", True),
                                  ("product-state target", False)):
        qc = build_full_circuit(n_prec, n_targ, D, target_init_seed=TARGET_INIT_SEED,
                                eigenstate=eigenstate)
        frac = sample_agreement_fraction(kind, sampler, qc, n_prec, shots)
        print(f"  {tgt_label}: sampled P(m1==m2) = {frac:.4f}  ({shots} shots)")


def run_sampled_check_factorized(n_targ: int, n_prec: int, ks: tuple, shots: int = 4000):
    """
    Same genuine-MCM sampled cross-check as run_sampled_check, but through
    build_full_circuit(..., factors=...) (diagonal-factorized U) instead of
    the dense-U path -- confirms the additive factors branch also survives
    an actual mid-circuit-measurement collapse under the sampler, not just
    the branch-exact statevector projection in run_factorized_joint_check.
    """
    print("\n" + "=" * 78)
    print(f"SAMPLED MCM CROSS-CHECK, FACTORIZED PATH (diagonal, {shots} shots)")
    print("=" * 78)
    _, factors_diag = on_grid_unitary(n_targ, n_prec, ks=ks, return_factors=True)
    kind, sampler = get_sampler()
    for tgt_label, eigenstate in (("eigenstate target", True),
                                  ("product-state target", False)):
        qc = build_full_circuit(n_prec, n_targ, None, target_init_seed=TARGET_INIT_SEED,
                                eigenstate=eigenstate, factors=factors_diag)
        frac = sample_agreement_fraction(kind, sampler, qc, n_prec, shots)
        print(f"  {tgt_label}: sampled P(m1==m2) = {frac:.4f}  ({shots} shots)")


# =========================================================================
# product_rotated_on_grid_unitary check
# =========================================================================

def run_product_rotated_check(n_targ: int, n_prec: int, ks: tuple, seed: int):
    print("\n" + "=" * 78)
    print(f"PRODUCT-ROTATED on-grid U check (n_targ={n_targ}, n_prec={n_prec}, "
          f"ks={ks}, seed={seed})")
    print("=" * 78)

    rng = np.random.default_rng(seed=seed)
    U = product_rotated_on_grid_unitary(n_targ, n_prec, rng=rng, ks=ks)

    # Product-state (RY/RZ) target: ideal_joint's ordinary path applies.
    joint_prod = ideal_joint(n_prec=n_prec, n_targ=n_targ,
                             target_init_seed=TARGET_INIT_SEED,
                             U=U, eigenstate=False)
    p_prod = float(np.trace(joint_prod)) / joint_prod.sum()
    print(f"  product-state target: P(m1==m2) = {p_prod:.4f}")

    # Eigenstate target V|1..1> -- NOT |1..1> itself (see
    # product_rotated_eigenstate_joint's docstring): standalone branch-exact
    # construction since |1..1> is only guaranteed an eigenvector of the
    # diagonal variant.
    joint_eig = product_rotated_eigenstate_joint(n_targ, n_prec, ks, seed)
    p_eig = float(np.trace(joint_eig)) / joint_eig.sum()
    print(f"  eigenstate target V|1..1>: P(m1==m2) = {p_eig:.4f}")


# =========================================================================
# Transpiled gate counts vs the IonQ 2,000-gate hard limit
# =========================================================================

def run_gate_count_report(n_targ: int, ks: tuple):
    from qiskit import transpile

    print("\n" + "=" * 78)
    print("TRANSPILED GATE COUNTS vs IonQ ~2,000-gate hard limit")
    # probe_gate_limit.py's own docstring quotes 1,915/2,583-gate figures for
    # a Haar U at this size; re-running that script's OWN build_and_transpile
    # right now (this installed qiskit==2.0.0) gives 4,562 gates for the
    # identical Haar seed=10 draw at n_prec=3 -- those docstring numbers are
    # stale relative to the currently-installed qiskit (transpiled
    # UnitaryGate.control() synthesis is qiskit-version-dependent), not a
    # bug in this script's counting. See docs/LOG.md addendum.
    print("(NOTE: probe_gate_limit.py's docstring-quoted 1,915/2,583 are "
          "STALE for the currently installed qiskit -- see docs/LOG.md.)")
    print("(NOTE: all counts below are LOCAL, PROVISIONAL -- qiskit 2.0.0, "
          "pending reconciliation against the braket-arm env's qiskit "
          "2.3.0. Do not treat as final.)")
    print("=" * 78)
    for n_prec in (3, 4, 5):
        D = on_grid_unitary(n_targ, n_prec, ks=ks)
        rng_v = np.random.default_rng(seed=0)
        R = rotated_on_grid_unitary(n_targ, n_prec, rng=rng_v, ks=ks)
        PR = product_rotated_on_grid_unitary(n_targ, n_prec, rng=np.random.default_rng(seed=0), ks=ks)
        for u_label, U in (("diagonal", D), ("rotated", R), ("prod-rotated", PR)):
            qc = build_full_circuit(n_prec, n_targ, U,
                                    target_init_seed=TARGET_INIT_SEED)
            qc_hw = transpile(qc, basis_gates=['rz', 'rx', 'rxx', 'measure', 'reset'],
                              optimization_level=1)
            ops = qc_hw.count_ops()
            n_1q = ops.get('rz', 0) + ops.get('rx', 0)
            n_2q = ops.get('rxx', 0)
            n_total = n_1q + n_2q
            print(f"  n_prec={n_prec}  {u_label:24s} U: 1q={n_1q:5d}  2q={n_2q:5d}  "
                  f"total={n_total:5d}  "
                  f"({'UNDER' if n_total <= 2000 else 'OVER'} the 2,000-gate limit)")

        # Factorized-path counterparts (build_qpe_stage_product): each
        # controlled-U^{2^p} is n_targ independent controlled-SINGLE-QUBIT
        # gates instead of one dense 2^n_targ x 2^n_targ controlled unitary
        # -- reported next to the dense rows above for direct comparison.
        _, factors_diag = on_grid_unitary(n_targ, n_prec, ks=ks, return_factors=True)
        _, factors_pr = product_rotated_on_grid_unitary(
            n_targ, n_prec, rng=np.random.default_rng(seed=0), ks=ks, return_factors=True)
        for u_label, factors in (("diagonal-factorized", factors_diag),
                                 ("prod-rotated-factorized", factors_pr)):
            qc = build_full_circuit(n_prec, n_targ, None,
                                    target_init_seed=TARGET_INIT_SEED, factors=factors)
            qc_hw = transpile(qc, basis_gates=['rz', 'rx', 'rxx', 'measure', 'reset'],
                              optimization_level=1)
            ops = qc_hw.count_ops()
            n_1q = ops.get('rz', 0) + ops.get('rx', 0)
            n_2q = ops.get('rxx', 0)
            n_total = n_1q + n_2q
            print(f"  n_prec={n_prec}  {u_label:24s} U: 1q={n_1q:5d}  2q={n_2q:5d}  "
                  f"total={n_total:5d}  "
                  f"({'UNDER' if n_total <= 2000 else 'OVER'} the 2,000-gate limit)")


# =========================================================================
# Plot: ideal joint heatmaps, on-grid vs the existing Haar regression instance
# =========================================================================

PLOTS_DIR = os.path.join(HERE, "gate_limit_probe_results", "plots")


def make_plot() -> str:
    """
    Three-panel P(m1,m2) heatmap: the on-grid k=(1,2,4), n_prec=4 instance
    (eigenstate target, product-state target) next to the existing Haar
    seed=10, n_prec=3 regression instance, for contrast. Reuses `ideal_joint`
    and `on_grid_unitary` directly rather than recomputing the joint a
    different way, so this plot can never silently drift from the numbers
    `run_instance`/`run_regression` print.
    """
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "on_grid_two_stage_ideal_joint.png")

    n_targ, n_prec, ks = 3, 4, (1, 2, 4)
    D = on_grid_unitary(n_targ, n_prec, ks=ks)
    joint_eig = ideal_joint(n_prec=n_prec, n_targ=n_targ,
                            target_init_seed=TARGET_INIT_SEED,
                            U=D, eigenstate=True)
    joint_prod = ideal_joint(n_prec=n_prec, n_targ=n_targ,
                             target_init_seed=TARGET_INIT_SEED,
                             U=D, eigenstate=False)
    joint_haar = ideal_joint()  # defaults: Haar U, seed=10, n_prec=3, target_init_seed=99

    panels = [
        (joint_eig, "On-grid k=(1,2,4), n_prec=4\ndiagonal U, eigenstate target |1..1>"),
        (joint_prod, "On-grid k=(1,2,4), n_prec=4\ndiagonal U, product-state target"),
        (joint_haar, "Haar U, seed=10, n_prec=3\n(regression instance, for contrast)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(top=0.72, wspace=0.45)

    # Per-panel vmax/colorbar rather than one shared scale (as in
    # decode_mcm_qpuf_result.py's 2-panel plot): the eigenstate panel is a
    # single delta spike at probability 1.0, which would flatten the other
    # two panels to near-black under a shared scale.
    for ax, (joint, title) in zip(axes, panels):
        d = joint.shape[0]
        p_diag = float(np.trace(joint)) / joint.sum()
        vmax = float(joint.max())
        im = ax.imshow(joint, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
        ax.set_xlabel("m2")
        ax.set_ylabel("m1")
        ax.set_xticks(range(d))
        ax.set_yticks(range(d))
        ax.set_title(f"{title}\nP(m1==m2) = {p_diag:.4f}", fontsize=9.5)
        # Mark the m1==m2 diagonal -- the QPUF verification condition.
        ax.plot([0, d - 1], [0, d - 1], color="red", lw=1.0, ls="--", alpha=0.8)
        for i in range(d):
            for j in range(d):
                if joint[i, j] > 0.02:
                    ax.text(j, i, f"{joint[i, j]:.2f}", ha="center", va="center",
                           fontsize=6 if d <= 8 else 5.5,
                           color="white" if joint[i, j] > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06, label="probability")

    fig.suptitle(
        "Ideal (noiseless) two-stage QPE-QPUF joint P(m1,m2), branch-exact "
        "statevector sim\n"
        "MSB-first convention: m = sum_k c[k]*2^(n_prec-1-k), c[0] = QPE "
        "qubit carrying the highest power of U (see decode_mcm_qpuf_result.py "
        "module docstring)\n"
        "red dashed line = m1==m2 diagonal, the QPUF verification condition",
        fontsize=10, y=0.99,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


# =========================================================================
# Regression check on the generalized ideal_joint (Haar path unchanged)
# =========================================================================

def run_regression():
    joint = ideal_joint()  # defaults: U=None -> Haar draw, seed=10, target_init_seed=99, n_prec=3
    p_diag = float(np.trace(joint)) / joint.sum()
    print(f"Regression check (Haar U, seed=10, target_init_seed=99, n_prec=3): "
          f"P(m1==m2) = {p_diag:.4f}  (expect 0.8791)")


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-prec", type=int, default=4)
    ap.add_argument("--n-targ", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1,
                    help="RNG seed for the secondary random-k instance and its rotation V")
    ap.add_argument("--regression", action="store_true",
                    help="print P(m1==m2) for the original Haar path and exit")
    ap.add_argument("--plot", action="store_true",
                    help="write the ideal-joint heatmap comparison plot and exit")
    args = ap.parse_args()

    if args.regression:
        run_regression()
        return

    if args.plot:
        make_plot()
        return

    # Primary known-answer instance: all 8 subset sums of (1,2,4) are
    # distinct mod 16, so the m1 marginal reproduces the target populations
    # bin-by-bin exactly (see module docstring / task spec).
    run_instance("primary (k=(1,2,4))", n_targ=3, n_prec=4, ks=(1, 2, 4),
                rot_seed=args.seed)

    # Secondary instance: a seeded random k obeying the "not all zero, not
    # all equal" constraints, at the CLI-specified dimensions.
    rng = np.random.default_rng(seed=args.seed)
    d = 2 ** args.n_prec
    while True:
        ks_secondary = tuple(int(rng.integers(0, d)) for _ in range(args.n_targ))
        if any(ks_secondary) and len(set(ks_secondary)) > 1:
            break
    run_instance(f"secondary (seed={args.seed})", n_targ=args.n_targ,
                n_prec=args.n_prec, ks=ks_secondary, rot_seed=args.seed)

    run_product_rotated_check(n_targ=3, n_prec=4, ks=(1, 2, 4), seed=args.seed)
    run_sampled_check(n_targ=3, n_prec=4, ks=(1, 2, 4))
    run_gate_count_report(n_targ=3, ks=(1, 2, 4))

    # Factorized path (build_qpe_stage_product, additive alongside the
    # dense build_qpe_stage path exercised above).
    check_known_answer_product()
    run_factorized_joint_check(n_targ=3, n_prec=4, ks=(1, 2, 4), seed=args.seed)
    run_sampled_check_factorized(n_targ=3, n_prec=4, ks=(1, 2, 4))


if __name__ == "__main__":
    main()
