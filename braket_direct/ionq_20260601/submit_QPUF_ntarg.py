#!/usr/bin/env python3
"""
submit_QPUF_ntarg.py
====================
Submits a two-stage QPE QPUF circuit to IonQ Forte-Enterprise-1 on AWS
Braket, generalised to n_targ > 1 target qubits using exact Haar-random
n-qubit unitaries.

Circuit shape (single quantum + single classical register, as required by
Braket OpenQASM 3). Stage 2 uses a fresh precision-ancilla block because
Forte-Enterprise-1 does NOT expose `reset` via its OpenQASM submission
path (verified against `device.properties.action.supportedOperations`):

    target[n_targ]    : random product state, one RY+RZ per qubit
    prec_a[n_prec]    : QPE ancillae for stage 1
    prec_b[n_prec]    : QPE ancillae for stage 2 (disjoint from prec_a)

      H^{⊗n_prec} ── ctrl-U^{2^{n_prec-1-k}} ── invQFT ── measure → c[0..n_prec-1]
                                                                      (MCM — target collapses)
      H^{⊗n_prec} ── ctrl-U^{2^{n_prec-1-k}} ── invQFT ── measure → c[n_prec..2n_prec-1]

The same Haar-random U is used in both stages (this is the PE-QPUF
construction — the second-stage outcomes verify the eigenstate collapsed by
the stage-1 mid-circuit measurement).

Qubit budget: N_TARG + 2 * N_PREC must fit Forte-Enterprise-1 (35 qubits).

Haar sampling: complex Ginibre → QR → phase-fix diag(R)/|diag(R)|
(Mezzadri, arXiv:math-ph/0609050).

Controlled-U synthesis: U^{2^k} computed classically via np.linalg.matrix_power,
wrapped as qiskit UnitaryGate, then .control(1). Qiskit's transpiler decomposes
the controlled arbitrary unitary into native gates at submission time.
"""

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DEVICE_NAME = "Forte-Enterprise-1"
DEVICE_ARN  = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"
RES_ARN = "arn:aws:braket:us-east-1:767397707562:reservation/08fda262-2902-4a28-b88e-0969df8830c7"
N_PREC      = 15          # precision qubits (shared across both stages)
N_TARG      = 1          # target qubits — Haar-random unitary acts on these
N_SHOTS     = 200

SEED        = 10           # RNG seed for the Haar-random unitary
TARGET_INIT_SEED = 99       # RNG seed for the target-state initialisation

# IonQ Forte-1 gate-time model (used only for the wall-clock estimate)
ONE_QUBIT_TIME_S = 130e-6           # per 1q native gate
TWO_QUBIT_TIME_S = 970e-6           # per 2q native gate
READOUT_TIME_S   = 150e-6 + 50e-6   # 200 µs per readout event
STARTUP_TIME_S   = 0.5              # fixed per-task overhead
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate

JOB_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "job_results")
LOG_FILE        = os.path.join(JOB_RESULTS_DIR, "job_log.txt")


# ── Haar-random unitary ────────────────────────────────────────────────────────

def haar_random_unitary(d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample a d x d Haar-distributed unitary matrix.

    Method (Mezzadri, arXiv:math-ph/0609050):
      1. Draw a complex Ginibre matrix Z ~ (N(0,1) + i N(0,1)) / sqrt(2).
      2. QR-decompose Z = Q R.
      3. Phase-fix: Lambda = diag(R) / |diag(R)|; return Q @ diag(Lambda).
         (Raw QR is NOT Haar — the diagonal phase fix is mandatory.)
    """
    if rng is None:
        rng = np.random.default_rng()

    Z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag_R = np.diag(R)
    Lambda = diag_R / np.abs(diag_R)
    return Q * Lambda  # broadcast: multiplies each column of Q by Lambda[j]


def _resolve_ks(n_targ: int, n_prec: int,
                rng: np.random.Generator | None,
                ks: tuple[int, ...] | None) -> tuple[int, ...]:
    """
    Shared ks validation/draw logic for on_grid_unitary and
    product_rotated_on_grid_unitary (factored out so both enforce the same
    hard constraints instead of duplicating them).

    NOTE ON RNG CONSUMPTION: when ks is given explicitly, this function
    draws nothing from rng. product_rotated_on_grid_unitary's docstring
    relies on that: with ks explicit, its only rng draws are the n_targ
    per-qubit Haar unitaries, in a fixed order, so a caller can replay them
    deterministically from a fresh same-seed rng without this function
    needing to return them separately.
    """
    d = 2 ** n_prec
    if ks is None:
        if rng is None:
            rng = np.random.default_rng()
        while True:
            candidate = tuple(int(rng.integers(0, d)) for _ in range(n_targ))
            if any(candidate) and len(set(candidate)) > 1:
                return candidate
    if len(ks) != n_targ:
        raise ValueError(f"len(ks)={len(ks)} != n_targ={n_targ}")
    if any(not (0 <= k < d) for k in ks):
        raise ValueError(f"each k_i must be in [0, {d}); got {ks}")
    if not any(ks):
        raise ValueError("not all k_i may be zero (U would be identity)")
    if len(set(ks)) == 1:
        raise ValueError("not all k_i may be equal (degenerate case)")
    return ks


def _check_subset_sums_distinct(ks: tuple[int, ...], n_prec: int) -> None:
    """
    Soft degeneracy guard (warns, does not raise): "not all k_i equal" rules
    out only the single most-degenerate case, not partial collisions among
    the 2^n_targ eigenphases sum_{i in subset} k_i mod 2^n_prec. Collisions
    are legal -- P(m1==m2)=1 still holds (the mixture-over-eigenvectors
    derivation doesn't need distinct phases) -- they just mean some
    eigenspaces are 2+-dimensional, so the m1 marginal groups multiple
    computational basis states into a shared bin rather than one bin per
    basis state. Worth flagging, not worth failing on.
    """
    d = 2 ** n_prec
    n = len(ks)
    sums = [sum(k for i, k in enumerate(ks) if (j >> i) & 1) % d for j in range(2 ** n)]
    if len(set(sums)) < len(sums):
        warnings.warn(
            f"on_grid_unitary: subset sums of ks={ks} are not all distinct mod "
            f"2^{n_prec} -- some eigenspaces of U are 2+ dimensional. "
            "P(m1==m2)=1 still holds; the m1 marginal groups multiple basis "
            "states per bin instead of one bin per basis state.",
            stacklevel=3,
        )


def on_grid_unitary(n_targ: int, n_prec: int,
                    rng: np.random.Generator | None = None,
                    ks: tuple[int, ...] | None = None,
                    return_factors: bool = False):
    """
    Build a diagonal U = tensor_i P(2*pi*k_i / 2^n_prec) whose every
    eigenphase is an exact multiple of the n_prec-bit QPE grid spacing.

    Why: noiselessly, P(m1==m2) = sum_j |c_j|^2 P_j(m1) P_j(m2) -- a mixture
    over independent single-stage QPE draws on each eigenvector j. QPE only
    returns j's phase exactly when that phase already sits on the 2^-n_prec
    grid; any off-grid phase spreads probability across bins, so m1 and m2
    can disagree even with zero physical noise (more n_prec does not fix
    this for a fixed off-grid phase -- Haar U in general has none). Forcing
    every eigenphase onto the grid makes the noiseless two-stage circuit
    P(m1==m2)=1 exactly, giving a clean reference: any deviation on real
    hardware is then attributable to noise, not to this off-grid leakage.

    U[j,j] = exp(2*pi*i * ((sum of k_i over set bits i of j) mod 2^n_prec)
                  / 2^n_prec)
    -- equivalent to the tensor-product-of-phase-gates form above because
    UnitaryGate is little-endian over the supplied qubit list, so bit i of
    row index j corresponds to target qubit i.

    return_factors=True additionally returns the n_targ per-qubit diagonal
    2x2 factors [diag(1, exp(2*pi*i*k_i/2^n_prec)) for i in range(n_targ)]
    (U = tensor_i factors[i], same index convention) as (U, factors), for
    callers that want to build the product-form circuit
    (build_qpe_stage_product) instead of the dense one. Default is
    unchanged (returns U alone) so existing call sites are unaffected.
    """
    d = 2 ** n_prec
    ks = _resolve_ks(n_targ, n_prec, rng, ks)
    _check_subset_sums_distinct(ks, n_prec)

    dim = 2 ** n_targ
    diag = np.empty(dim, dtype=complex)
    for j in range(dim):
        s = sum(k for i, k in enumerate(ks) if (j >> i) & 1)
        diag[j] = np.exp(2j * np.pi * (s % d) / d)
    U = np.diag(diag)

    if not return_factors:
        return U
    factors = [np.diag([1.0 + 0j, np.exp(2j * np.pi * ks[i] / d)]) for i in range(n_targ)]
    return U, factors


def random_diagonal_on_grid_unitary(n_targ: int, n_prec: int,
                                    rng: np.random.Generator,
                                    return_m_j: bool = False):
    """
    Fully generic on-grid diagonal U: ONE independent random integer m_j per
    computational basis state j (j = 0..2^n_targ-1), U = diag(exp(2*pi*i*
    m_j/2^n_prec)). Unlike on_grid_unitary (per-QUBIT phases k_i, giving U
    the tensor-product structure U = tensor_i P(2*pi*k_i/2^n_prec)), this
    has no tensor-product structure at all -- it is per-BASIS-STATE, so the
    eigenbasis is still the computational basis (same MCM-disturbance
    rationale as on_grid_unitary) but the phases carry no per-qubit pattern.
    Kept additive: on_grid_unitary is untouched.

    Dense path only: goes straight through the unmodified build_qpe_stage/
    build_full_circuit (no build_qpe_stage_product path -- a fully generic
    per-basis-state diagonal has no tensor-product structure for that path
    to exploit).

    rng is required (not optional/defaulted) -- caller controls seeding
    explicitly for this one, unlike on_grid_unitary's ks=None convenience
    overload; no fixed-values kwarg here since the deliverable is the
    seeded-random path, not a known-answer instance.

    Hard ValueError if every m_j is equal (global phase, same degenerate-
    identity-up-to-phase rationale as on_grid_unitary's not-all-equal
    check). Soft warning (not a failure) if the m_j are not all mutually
    distinct -- unlike on_grid_unitary's subset-sum collisions, there is no
    subset-sum structure here (each basis state gets its own independent
    draw), so this is a plain "not all distinct" check on the m_j list
    directly. Collisions are legal: P(m1==m2)=1 still holds; a repeated m_j
    just makes the corresponding eigenspace 2+-dimensional.

    return_m_j=True additionally returns the drawn list of m_j (as (U,
    m_j)) so a caller can compute eigenbins/eigen-populations directly
    instead of re-eigendecomposing a matrix whose diagonal it already knows.
    """
    d = 2 ** n_prec
    dim = 2 ** n_targ
    m_j = [int(rng.integers(0, d)) for _ in range(dim)]

    if len(set(m_j)) == 1:
        raise ValueError("not all m_j may be equal (U would be a global phase)")
    if len(set(m_j)) < len(m_j):
        warnings.warn(
            f"random_diagonal_on_grid_unitary: m_j={m_j} are not all mutually "
            f"distinct -- some eigenspaces of U are 2+ dimensional. "
            "P(m1==m2)=1 still holds; the m1 marginal groups multiple basis "
            "states per bin instead of one bin per basis state.",
            stacklevel=2,
        )

    diag = np.exp(2j * np.pi * np.array(m_j, dtype=float) / d)
    U = np.diag(diag)

    if not return_m_j:
        return U
    return U, m_j


def rotated_on_grid_unitary(n_targ: int, n_prec: int,
                            rng: np.random.Generator | None = None,
                            ks: tuple[int, ...] | None = None) -> np.ndarray:
    """
    U = V @ D @ V^dagger, D = on_grid_unitary(...), V a Haar-random unitary.

    D alone has the computational basis as its eigenbasis, which is a
    degenerate probe for the planned hardware MCM-disturbance test (that
    test models the disturbance as a coherent rotation IN THE EIGENBASIS OF
    U -- if the eigenbasis is the computational basis the rotation and the
    measurement basis coincide, hiding exactly the effect being probed).
    Conjugating by a generic Haar V keeps every eigenphase on-grid (the
    noiseless P(m1==m2)=1 baseline still holds, since eigenvalues are
    unchanged by a similarity transform) while making the eigenbasis
    generic.
    """
    if rng is None:
        rng = np.random.default_rng()
    D = on_grid_unitary(n_targ, n_prec, rng=rng, ks=ks)
    V = haar_random_unitary(2 ** n_targ, rng=rng)
    return V @ D @ V.conj().T


def product_rotated_on_grid_unitary(n_targ: int, n_prec: int,
                                    rng: np.random.Generator | None = None,
                                    ks: tuple[int, ...] | None = None,
                                    return_factors: bool = False):
    """
    Cheaper generic-eigenbasis on-grid U: U = tensor_i (v_i P(2*pi*k_i/2^n_prec)
    v_i^dagger), with v_i an independent Haar-random SINGLE-qubit (2x2)
    unitary -- unlike rotated_on_grid_unitary's single dense n_targ-qubit V,
    this stays a tensor product of single-qubit unitaries, so each
    controlled-U^{2^p} in build_qpe_stage decomposes into n_targ independent
    controlled-single-qubit gates instead of one dense 2^n_targ x 2^n_targ
    controlled unitary. Eigenphases stay on-grid (same k_i's as
    on_grid_unitary -- P(m1==m2)=1 still holds); the eigenbasis becomes the
    generic PRODUCT basis {v_i|0>, v_i|1>} instead of the computational
    basis, which is the property rotated_on_grid_unitary wants for the
    planned MCM-disturbance probe, at a fraction of the transpiled cost.

    Kronecker order matches on_grid_unitary: target qubit 0 is the
    RIGHTMOST kron operand (bit 0 of the computational-basis index is
    qubit 0, UnitaryGate's little-endian convention).

    RNG CONSUMPTION ORDER (load-bearing for verification code that needs
    the v_i's without this function returning them): when ks is given
    explicitly, _resolve_ks draws nothing, so the only draws are
    haar_random_unitary(2, rng=rng) for i = 0, 1, ..., n_targ-1 in that
    order. A caller can reconstruct the same v_i sequence by replaying
    those draws from a fresh Generator seeded identically.

    return_factors=True additionally returns the n_targ per-qubit 2x2
    factors u_i = v_i @ P_i @ v_i^dagger (U = tensor_i factors[i], same
    index convention) as (U, factors), for callers that want to build the
    product-form circuit (build_qpe_stage_product) instead of the dense
    one. Default is unchanged (returns U alone) so existing call sites are
    unaffected.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = 2 ** n_prec
    ks = _resolve_ks(n_targ, n_prec, rng, ks)
    _check_subset_sums_distinct(ks, n_prec)

    U = np.array([[1.0]], dtype=complex)
    u_list = []
    for i in range(n_targ):
        theta_i = 2 * np.pi * ks[i] / d
        P_i = np.diag([1.0, np.exp(1j * theta_i)])
        v_i = haar_random_unitary(2, rng=rng)
        u_i = v_i @ P_i @ v_i.conj().T
        u_list.append(u_i)
        U = np.kron(u_i, U)

    if not return_factors:
        return U
    return U, u_list


def _stable_matrix_power(U: np.ndarray, power: int) -> np.ndarray:
    """
    U**power, re-projected onto the nearest unitary via SVD (polar
    decomposition). Repeated squaring in float64 squares the error each
    step, so at N_PREC=32 the raw U^{2^31} drifts ~1e-6 from unitary and
    Qiskit's UnitaryGate unitarity check rejects it.
    """
    M = np.linalg.matrix_power(U, power)
    W, _, Vh = np.linalg.svd(M)
    return W @ Vh


# ── QPE stage builder ──────────────────────────────────────────────────────────

def build_qpe_stage(n_prec: int, U: np.ndarray) -> QuantumCircuit:
    """
    Build one QPE stage: H on every precision qubit, controlled-U^{2^{n_prec-1-k}}
    for precision qubit k (qubit 0 is the MSB of the measured integer), then
    inverse QFT on the precision register.

    Qubit layout: [prec[0], ..., prec[n_prec-1], targ[0], ..., targ[n_targ-1]].
    """
    n_targ = int(round(np.log2(U.shape[0])))
    assert 2 ** n_targ == U.shape[0], "U dimension must be a power of 2"
    total = n_prec + n_targ

    qc = QuantumCircuit(total, name="QPE")
    prec = list(range(n_prec))
    targ = list(range(n_prec, n_prec + n_targ))

    qc.h(prec)

    # Stage exponent: precision qubit k (k=0 first) controls U^{2^{n_prec-1-k}}.
    # This puts the most significant bit at qubit 0 of the precision register.
    for k in range(n_prec):
        power = 2 ** (n_prec - 1 - k)
        U_pow = _stable_matrix_power(U, power)
        cU = UnitaryGate(U_pow, label=f"U^{power}").control(1)
        # Order: [control, target_0, target_1, ...]
        qc.append(cU, [prec[k]] + targ)

    iqft = QFTGate(n_prec).inverse()
    # QFTGate is little-endian (qubit k weight 2^k) but prec is built MSB-first
    # (prec[0] weight 2^{n_prec-1}); must reverse index order or phase smears
    # across output bins instead of landing in one bin (known-bug list item 4).
    qc.append(iqft, prec[::-1])

    return qc


def _factor_power(u_i: np.ndarray, n_prec: int, power: int) -> np.ndarray:
    """
    Exact u_i^power for a single 2x2 unitary factor, computed closed-form
    instead of via repeated-squaring + SVD reprojection
    (_stable_matrix_power) -- unlike an 8x8 dense controlled-U, a 2x2
    factor's spectral decomposition is cheap, so there's no reason to
    accept squaring error here at all.

    Diagonal case (the on-grid construction diag(1, exp(2*pi*i*k/2^n_prec))
    used by on_grid_unitary): exponentiate via exact MODULAR INTEGER
    arithmetic on the phase index k (new_k = k*power mod 2^n_prec), not
    `phase**power` -- raising a float phase to a large integer power the
    naive way is exactly the drift _stable_matrix_power exists to correct
    for the dense case, and here it's free to avoid entirely.

    Non-diagonal case (e.g. product_rotated_on_grid_unitary's v_i-rotated
    factor u_i = v_i @ diag(1, e^{i*theta}) @ v_i^dagger): eigendecompose
    via np.linalg.eig (robust generic fallback -- this function only sees
    a plain 2x2 matrix, not how it was constructed) and raise the
    eigenvalues to `power` in floating point, then reassemble. u_i is
    unitary (normal) and, for every factor this project actually
    constructs, non-degenerate (k_i=0 collapses to the identity, which is
    diagonal and caught above), so np.linalg.eig returns an eigenvector
    matrix that is unitary to machine precision and v_i^dagger is a valid
    inverse.
    """
    if np.allclose(u_i, np.diag(np.diagonal(u_i)), atol=1e-10):
        d = 2 ** n_prec
        theta = float(np.angle(u_i[1, 1]))
        k = int(round(theta * d / (2 * np.pi))) % d
        new_k = (k * power) % d
        return np.diag([1.0 + 0j, np.exp(2j * np.pi * new_k / d)])

    eigvals, eigvecs = np.linalg.eig(u_i)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.conj().T


def build_qpe_stage_product(n_prec: int, factors: list[np.ndarray]) -> QuantumCircuit:
    """
    Product-form equivalent of build_qpe_stage for U = tensor_i factors[i]
    (true of on_grid_unitary and product_rotated_on_grid_unitary, NOT
    rotated_on_grid_unitary's dense V). Each controlled-U^{2^p} becomes
    n_targ independent controlled-SINGLE-QUBIT gates instead of one dense
    2^n_targ x 2^n_targ controlled unitary -- qiskit's generic
    controlled-unitary synthesis doesn't exploit tensor-product structure,
    so for product-form U this produces an identical circuit action (see
    two_stage_on_grid_baseline.py's Operator-equivalence check) at a
    fraction of the transpiled gate count.

    factors[i] is the 2x2 unitary acting on target qubit i, same
    Kronecker/index convention as on_grid_unitary and
    product_rotated_on_grid_unitary (U = tensor_i factors[i], qubit 0 the
    rightmost kron operand). Same qubit layout, MSB-first control-power
    convention (precision qubit k controls power 2^(n_prec-1-k)), and
    inverse-QFT-on-reversed-prec placement as build_qpe_stage -- see that
    function's docstring. Duplicated rather than shared because the
    controlled-gate-application loop body genuinely differs (n_targ
    single-qubit gates vs. one n_targ-qubit gate); build_qpe_stage itself
    is untouched by this addition.
    """
    n_targ = len(factors)
    total = n_prec + n_targ

    qc = QuantumCircuit(total, name="QPE_product")
    prec = list(range(n_prec))
    targ = list(range(n_prec, n_prec + n_targ))

    qc.h(prec)

    # Same MSB-first convention as build_qpe_stage: precision qubit k
    # (k=0 first) controls power 2^{n_prec-1-k}.
    for k in range(n_prec):
        power = 2 ** (n_prec - 1 - k)
        for i in range(n_targ):
            u_i_pow = _factor_power(factors[i], n_prec, power)
            cU = UnitaryGate(u_i_pow, label=f"u{i}^{power}").control(1)
            qc.append(cU, [prec[k], targ[i]])

    iqft = QFTGate(n_prec).inverse()
    # Same reversed-index placement as build_qpe_stage -- see that
    # function's comment (known-bug list item 4).
    qc.append(iqft, prec[::-1])

    return qc


def build_full_circuit(n_prec: int, n_targ: int, U: np.ndarray | None,
                       target_init_seed: int,
                       eigenstate: bool = False,
                       factors: list[np.ndarray] | None = None) -> QuantumCircuit:
    """
    Two-stage QPE QPUF, flattened to one quantum + one classical register
    (Braket OpenQASM 3 allows only one of each).

    Forte-Enterprise-1 does NOT expose `reset` via the OpenQASM submission
    path (verified by querying its `supportedOperations`), so we cannot
    reuse the precision register across stages. Instead we allocate two
    disjoint precision blocks. MCM on the stage-1 block still collapses
    the target (`supportsUnassignedMeasurements: True`), so the stage-2
    block performs QPE on the same collapsed eigenstate — preserving the
    PE-QPUF construction.

        q[0 .. n_targ-1]                            → target
        q[n_targ .. n_targ+n_prec-1]                → prec_a  (stage 1)
        q[n_targ+n_prec .. n_targ+2*n_prec-1]       → prec_b  (stage 2)
        c[0 .. n_prec-1]                            → stage-1 outcomes (low bits)
        c[n_prec .. 2*n_prec-1]                     → stage-2 outcomes (high bits)

    factors: optional list of n_targ 2x2 unitaries (U = tensor_i
    factors[i], from on_grid_unitary(..., return_factors=True) or
    product_rotated_on_grid_unitary(..., return_factors=True)). When
    given, both QPE stages are built via build_qpe_stage_product instead
    of the dense build_qpe_stage(n_prec, U) -- factors takes precedence
    over U for circuit construction. U itself becomes optional in that
    case (pass U=None if the caller has no other use for it, e.g. no need
    to report an expected eigenphase); it plays no role in the built
    circuit when factors is supplied. When factors is None (the default),
    behavior is exactly the original dense-U path, unchanged.
    """
    if factors is None and U is None:
        raise ValueError("build_full_circuit requires U, factors, or both")

    n_total    = n_targ + 2 * n_prec
    q          = QuantumRegister(n_total, "q")
    c          = ClassicalRegister(2 * n_prec, "c")
    qc         = QuantumCircuit(q, c)

    targ_idx   = list(range(0, n_targ))
    prec_a_idx = list(range(n_targ,             n_targ + n_prec))
    prec_b_idx = list(range(n_targ + n_prec,    n_targ + 2 * n_prec))

    # Target initialisation: |1...1> (a known-answer eigenstate probe) or,
    # by default, one RY+RZ per qubit seeded independently of U.
    if eigenstate:
        for i in targ_idx:
            qc.x(q[i])
    else:
        init_rng = np.random.default_rng(seed=target_init_seed)
        for i in targ_idx:
            theta0 = init_rng.uniform(0, np.pi)
            phi0   = init_rng.uniform(0, 2 * np.pi)
            qc.ry(theta0, q[i])
            qc.rz(phi0,   q[i])

    # Stage builder: product-form (build_qpe_stage_product) when factors is
    # supplied, else the original dense-U path (build_qpe_stage), unchanged.
    def _build_stage() -> QuantumCircuit:
        if factors is not None:
            return build_qpe_stage_product(n_prec, factors)
        return build_qpe_stage(n_prec, U)

    # Stage 1 QPE on prec_a + targ. build_qpe_stage[_product] expects the
    # precision qubits first, then the target qubits.
    qpe1 = _build_stage()
    qc.append(qpe1, prec_a_idx + targ_idx)

    # Mid-circuit measurement of stage-1 precision register. This collapses
    # the target onto an eigenstate of U; the stage-2 QPE that follows then
    # acts on that collapsed state.
    qc.measure([q[i] for i in prec_a_idx], [c[k] for k in range(n_prec)])

    # Stage 2 QPE on prec_b + (already-collapsed) targ.
    qpe2 = _build_stage()
    qc.append(qpe2, prec_b_idx + targ_idx)

    qc.measure([q[i] for i in prec_b_idx], [c[n_prec + k] for k in range(n_prec)])

    return qc


# ── Logging ────────────────────────────────────────────────────────────────────

def append_job_log(record: dict):
    """Append a JSON record (one per line) to job_log.txt."""
    os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Job record written to: {LOG_FILE}")


def _encode_unitary(U: np.ndarray) -> dict:
    """Serialize a complex matrix to JSON-friendly real/imag lists."""
    return {
        "shape": list(U.shape),
        "real":  U.real.tolist(),
        "imag":  U.imag.tolist(),
    }


def _rewrite_qasm_for_braket(qasm: str) -> str:
    """
    Make qiskit's OpenQASM 3 output acceptable to AWS Braket.

    Braket's parser accepts only its own built-in gates: no `include`
    statements and no user-defined `gate` blocks. We:
      1. Strip any `gate rxx(...) { ... }` definition (matched by brace
         depth — robust to multi-line bodies).
      2. Rename every `rxx(<angle>) ...` call site to `xx(<angle>) ...`
         (Braket's native Ising-XX gate; matrix is identical to qiskit
         RXX, so no parameter rescaling is needed).
    """
    import re

    out_lines  = []
    skip_until = -1
    lines      = qasm.splitlines()
    i          = 0
    while i < len(lines):
        line = lines[i]

        # Drop "gate rxx(...) { ... }" block (balanced braces, any depth).
        if re.match(r'\s*gate\s+rxx\b', line):
            depth = line.count('{') - line.count('}')
            i += 1
            while i < len(lines) and depth > 0:
                depth += lines[i].count('{') - lines[i].count('}')
                i += 1
            continue

        # Rename call sites: `rxx(angle) q0, q1;` → `xx(angle) q0, q1;`.
        # Use a word boundary so we don't accidentally rewrite something
        # like `myrxx(...)` if it ever appears.
        line = re.sub(r'\brxx\s*\(', 'xx(', line)

        out_lines.append(line)
        i += 1

    return '\n'.join(out_lines) + ('\n' if qasm.endswith('\n') else '')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Build the Haar-random unitary ─────────────────────────────────────────
    rng = np.random.default_rng(seed=SEED)
    d = 2 ** N_TARG
    U = haar_random_unitary(d, rng=rng)

    # Sanity check: U should be unitary to within numerical precision.
    err = float(np.max(np.abs(U.conj().T @ U - np.eye(d))))
    if err > 1e-10:
        print(f"WARNING: |U†U − I|_max = {err:.2e} (expected ~1e-15)")

    print(f"QPU         : {DEVICE_NAME}  ({DEVICE_ARN})")
    print(f"RES_ARN     : {RES_ARN}")
    print(f"N_PREC      : {N_PREC}")
    print(f"N_TARG      : {N_TARG}  (U is {d}×{d})")
    print(f"N_SHOTS     : {N_SHOTS}")
    print(f"SEED        : {SEED}")
    print(f"|U†U − I|   : {err:.2e}")

    # ── Build the circuit ──────────────────────────────────────────────────────
    qc = build_full_circuit(N_PREC, N_TARG, U, target_init_seed=TARGET_INIT_SEED)
    print(f"\nCircuit qubits   : {qc.num_qubits}  "
          f"(prec={N_PREC} + targ={N_TARG})")

    # ── Resolve device (amazon-braket-sdk; bypass qiskit-braket-provider) ─────
    # qiskit-braket-provider's adapter refuses to translate `reset`
    # (NotImplementedError in providers/adapter.py), so we can't go through it
    # for an MCM+reset circuit. Instead we build/transpile in Qiskit, export
    # OpenQASM 3, and submit via the Braket SDK directly. IonQ Forte-1 +
    # Braket Direct accept `reset` at the QASM level.
    try:
        from braket.aws import AwsDevice, DirectReservation
        from braket.experimental_capabilities import EnableExperimentalCapability
        from braket.ir.openqasm import Program as OpenQasmProgram
        from qiskit import qasm3
    except ImportError as e:
        print(f"\nERROR: missing dependency: {e}")
        print("       pip install amazon-braket-sdk qiskit")
        sys.exit(1)

    device = AwsDevice(DEVICE_ARN)
    print(f"Device resolved  : {device.name}")

    # ── Qubit-count check ──────────────────────────────────────────────────────
    device_n_qubits = device.properties.paradigm.qubitCount
    if qc.num_qubits > device_n_qubits:
        print(f"\nERROR: circuit needs {qc.num_qubits} qubits but {DEVICE_NAME} "
              f"only has {device_n_qubits}.")
        print(f"       Reduce N_PREC and/or N_TARG so that "
              f"N_TARG + 2 * N_PREC ≤ {device_n_qubits}.")
        sys.exit(1)
    print(f"Device qubits    : {device_n_qubits}  "
          f"(circuit uses {qc.num_qubits})")

    # ── Transpile ──────────────────────────────────────────────────────────────
    # rz/rx/rxx ≈ IonQ MS-basis; Braket Direct's server-side compiler maps
    # these to native gpi/gpi2/zz at submission time. `reset` survives the
    # transpile and is preserved in the QASM 3 export below.
    print(f"\nTranspiling for {DEVICE_NAME} ...")
    qc_hw = transpile(
        qc,
        basis_gates=['rz', 'rx', 'rxx', 'measure', 'reset'],
        optimization_level=1,
    )
    n_gates = qc_hw.size()
    ops     = qc_hw.count_ops()
    n_1q    = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q    = ops.get('rxx', 0)
    n_meas  = ops.get('measure', 0)
    print(f"Transpiled depth : {qc_hw.depth()}")
    print(f"Transpiled gates : {n_gates}")
    print(f"  1q gates (rz+rx) : {n_1q}")
    print(f"  2q gates (rxx)   : {n_2q}")
    print(f"  measurements     : {n_meas}")

    # ── Wall-clock estimate (IonQ Forte-1 gate-time model) ────────────────────
    # (a) literal — readout charged once per shot, as written in the formula
    # (b) per-meas — readout charged per measurement event in the circuit
    per_shot_gates    = n_1q * ONE_QUBIT_TIME_S + n_2q * TWO_QUBIT_TIME_S
    t_est_literal_s   = STARTUP_TIME_S + N_SHOTS * (per_shot_gates + READOUT_TIME_S)
    t_est_permeas_s   = STARTUP_TIME_S + N_SHOTS * (per_shot_gates + n_meas * READOUT_TIME_S)

    def _fmt(t_s):
        return f"{t_s:.1f} s  ({t_s/60:.2f} min, {t_s/3600:.3f} h)"

    print(f"\nEstimated runtime (literal readout/shot)    : {_fmt(t_est_literal_s)}")
    print(f"Estimated runtime (readout × n_measurements): {_fmt(t_est_permeas_s)}")

    # ── Confirm before submitting ─────────────────────────────────────────────
    try:
        resp = input("\nSubmit this job to the QPU? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted — no job submitted.")
        return

    # ── Submit ─────────────────────────────────────────────────────────────────
    # Export to OpenQASM 3, then submit via the Braket SDK directly.
    # The DirectReservation context manager (recommended by Braket) routes
    # every task submitted inside its `with` block to the reservation, so it
    # bills against the reserved window instead of on-demand QPU time.
    #
    # Braket's OpenQASM 3 parser is restrictive: it rejects both `include`
    # statements AND user-defined `gate` declarations — only its built-in
    # gates are allowed. Qiskit's `rxx` isn't a Braket built-in, but Braket's
    # `xx(angle)` is the same matrix (verified: xx(θ) = exp(-iθ/2 X⊗X) =
    # qiskit RXX(θ)). So we (a) declare Braket built-ins as basis_gates so
    # qiskit emits no defs for them, (b) drop the include line, and (c)
    # rewrite the `rxx` definition + every `rxx(...)` call to `xx(...)`.
    BRAKET_BUILTIN_GATES = ['h', 'cx', 'cnot',
                            'rx', 'ry', 'rz', 'xx',
                            'x',  'y',  'z',
                            's',  't',  'swap', 'i']
    qasm_src = qasm3.dumps(qc_hw,
                            includes=(),
                            basis_gates=BRAKET_BUILTIN_GATES)
    qasm_src = _rewrite_qasm_for_braket(qasm_src)

    # `reset` is rejected by Braket's default OpenQASM parser; opting in to
    # experimental capabilities turns on the MCM+reset feature that Forte-1
    # supports under Braket Direct.
    print(f"\nSubmitting {N_SHOTS} shots to {DEVICE_NAME} under reservation {RES_ARN} ...")
    with DirectReservation(device, reservation_arn=RES_ARN), \
         EnableExperimentalCapability():
        task = device.run(
            OpenQasmProgram(source=qasm_src),
            shots=N_SHOTS,
        )
    job_id       = task.id    # Braket task ARN — same format as the old code
    submitted_at = datetime.now(timezone.utc).isoformat()

    print(f"Job submitted successfully.")
    print(f"Job ID    : {job_id}")
    print(f"Timestamp : {submitted_at}")

    record = {
        "job_id":               job_id,
        "datetime":             submitted_at,
        "qpu":                  DEVICE_NAME,
        "device_arn":           DEVICE_ARN,
        "reservation_arn":      RES_ARN,
        "circuit_type":         "QPUF_ntarg",
        "n_prec":               N_PREC,
        "n_targ":               N_TARG,
        "n_shots":              N_SHOTS,
        "n_gates":              n_gates,
        "n_1q_gates":           n_1q,
        "n_2q_gates":           n_2q,
        "n_measurements":       n_meas,
        "est_runtime_s_literal": t_est_literal_s,
        "est_runtime_s_permeas": t_est_permeas_s,
        "seed":                 SEED,
        "target_init_seed":     TARGET_INIT_SEED,
        "unitary":              _encode_unitary(U),
    }
    append_job_log(record)


if __name__ == "__main__":
    main()
