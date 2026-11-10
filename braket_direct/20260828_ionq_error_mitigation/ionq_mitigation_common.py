#!/usr/bin/env python3
"""
ionq_mitigation_common.py
==========================
Shared machinery for the REM/DDD/ZNE error-mitigation benchmark campaign on
IonQ Forte-Enterprise-1, submitted ON-DEMAND (no Braket Direct reservation
-- see MITIGATION_BRIEF_IONQ.md).

Everything submit_ionq_mitigation.py, error_mitig_ionq.py, mitig_analysis.py
and checkRetrieve.py need lives here so all of them agree on the circuit,
the gate counts and the timing model.


Why this differs from rigetti_qpuf_common.py
----------------------------------------------
1. ALL-TO-ALL CONNECTIVITY.
   Forte-Enterprise-1 gives every qubit pair a direct 2q gate, so there is
   no coupling map, no SWAP routing, no chiplet partition and no placement
   scorer here -- all of that lives only on the Rigetti side because
   Cepheus is a fixed lattice. See error_mitig_ionq.py's docstring for the
   corresponding statement about DDD/ZNE.

2. ON-DEMAND SUBMISSION, NOT A RESERVATION.
   There is no RES_ARN and no DirectReservation context manager anywhere in
   this campaign. Every submission is a bare
       AwsDevice(DEVICE_ARN).run(OpenQasmProgram(...), shots=...)
   following the pattern already verified working in
   noise_mitigation/ionq_noise_mitig.py.

3. VERBATIM STATUS -- NOW A CONFIRMED DEVICE FACT (was open, resolved
   2026-08-28).
   device_caps.json / device_properties_raw.json in this directory were
   fetched live from AwsDevice(DEVICE_ARN) on the DCV box and saved here:
   supportsPartialVerbatimBox=False, plain `verbatim` IS in
   supportedPragmas -- so it's whole-circuit-box-or-nothing, no partial
   wrapping. That resolves what MITIGATION_BRIEF_IONQ.md Section 9.1 used
   to call blocking. `to_braket_qasm(verbatim=True)` below already produces
   exactly that shape. NOTE this only tells you the box is accepted, not
   that our submitted rz/rx/rxx gates are the literal pulses executed --
   nativeGateSet is GPI/GPI2/ZZ, so a basis translation happens server-side
   even under verbatim; verbatim suppresses optimization/rewriting, not
   basis translation.

4. GATE-COUNT-BASED RUNTIME MODEL, NOT DEPTH-BASED.
   On a trapped-ion chain gates are serialized (no same-cycle parallel
   execution the way disjoint 2q gates run in one cycle on a
   superconducting lattice), so the timing model here tracks total gate
   COUNT x per-shot, matching noise_mitigation/ionq_noise_mitig.py and
   braket_direct/ionq_20260601/submit_QPUF_ntarg.py, not
   rigetti_qpuf_common.py's serial/parallel split.

5. TIMING CONSTANTS AND FIDELITIES ARE NOW CALIBRATION-DERIVED (was
   unvalidated placeholders copied from noise_mitigation/ionq_noise_mitig.py;
   resolved 2026-08-28).
   ONE_QUBIT_TIME_S, TWO_QUBIT_TIME_S, READOUT_TIME_S, F1Q, F2Q and
   SPAM_FIDELITY below are read off device_properties_raw.json's
   provider.timing / provider.fidelity, fetched live from the device on the
   DCV box -- not guesses anymore. STARTUP_TIME_S remains an unfetched
   placeholder (no corresponding field in the caps dump). The previously
   documented CONTRADICTION with the Rigetti model over how `rz` should be
   timed (RZ_TIME_S in error_mitig_ionq.py) is a modeling-convention
   question, not a numeric one, and is UNCHANGED by this update -- see
   MITIGATION_BRIEF_IONQ.md Section 9.2. Per-gate durations being real now
   does NOT make the wall-clock estimate in estimate_runtime_s() validated:
   no submitted task's actual qpu_time_seconds has been compared against
   this model's prediction yet.
"""

import json
import os
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate

# -- DEVICE ----------------------------------------------------------------
DEVICE_NAME = "Forte-Enterprise-1"
DEVICE_ARN  = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"
# No RES_ARN: this campaign submits on-demand, not under a Braket Direct
# reservation. See module docstring point 2.

# -- NATIVE GATE SET ---------------------------------------------------------
# rz/rx/rxx is the Braket-level operation set our circuits are transpiled to;
# by this repo's own account (submit_QPUF_ntarg.py, ionq_noise_mitig.py)
# IonQ's server-side compiler re-synthesises this into native gpi/gpi2/zz at
# submission time. Whether that recompilation preserves our exact gate
# sequence (needed for ZNE folds and DD pulses to survive) is OPEN --
# MITIGATION_BRIEF_IONQ.md Section 9.1.
IONQ_BASIS = ["rz", "rx", "rxx", "measure"]

# -- GATE-TIME MODEL AND FIDELITIES -----------------------------------------
# CALIBRATION-DERIVED as of 2026-08-28: fetched live from AwsDevice(DEVICE_ARN)
# on the DCV box and saved to device_properties_raw.json (provider.timing /
# provider.fidelity) in this directory. These replace the
# noise_mitigation/ionq_noise_mitig.py placeholders this file used to copy --
# per-gate durations and F1Q/F2Q are now real numbers, not guesses. NOTE:
# `qpu_time_seconds` predicted from them (estimate_runtime_s below) has still
# never been checked against an actual completed task's billed runtime, so
# the WALL-CLOCK estimate remains unvalidated even though its per-gate inputs
# are not -- see MITIGATION_BRIEF_IONQ.md Section 9.2.
ONE_QUBIT_TIME_S = 6.3e-05
TWO_QUBIT_TIME_S = 0.00065
READOUT_TIME_S   = 0.00025
STARTUP_TIME_S   = 0.5

# F = (F2Q)**n_2q * (F1Q)**n_1q -- device_properties_raw.json
# provider.fidelity.{1Q,2Q}.mean, randomized-benchmarking, fetched 2026-08-28.
F2Q = 0.9882
F1Q = 0.9998

# SPAM (state-prep-and-measure / readout) fidelity, provider.fidelity.spam.mean
# -- i.e. ~0.6% readout error per qubit. This is the quantity REM's confusion
# matrix is actually correcting; kept here as a named constant for the noisy-
# simulation validation harness (mitig_analysis.build_noise_model's
# `p_readout` default) rather than left as an unsourced guess.
SPAM_FIDELITY = 0.994

# Device-enforced (service.shotsRange in device_properties_raw.json, fetched
# 2026-08-28) -- unlike MAX_TOTAL_GATES below, this IS a cited platform
# limit: a task outside this range is rejected by Braket, not merely
# expensive. Debias (provider.errorMitigation...minimumShots) additionally
# requires >= 2500 shots, tighter than the device floor.
SHOTS_RANGE = (100, 5000)
DEBIAS_MIN_SHOTS = 2500

# Hard cap on (gates per circuit) * shots. THIS IS A SELF-IMPOSED BUDGET
# GUESS FOR THIS PROJECT, NOT A CITED AWS OR IONQ PLATFORM LIMIT -- do not
# read it as an authoritative device ceiling (CLAUDE.md: "ground truth beats
# prior claims"). Sized to cover the largest planned family (D2: n_prec=4,
# n_targ=3, haar -- 2,592,000 gate*shots at 2000 shots/task) with ~54%
# headroom; revisit if a ZNE ladder is ever added to a high-n_targ family
# (D2 at lambda=3 alone would be ~7.8M). If AWS/IonQ publish a real limit,
# replace this with a citation instead of a budget guess.
MAX_TOTAL_GATES = 4_000_000

# Hard cap on gates PER CIRCUIT. UNLIKE MAX_TOTAL_GATES ABOVE, THIS IS A
# REAL, EMPIRICALLY CONFIRMED AWS/IONQ PLATFORM LIMIT, not a project budget
# guess -- do not conflate the two, confusing them is exactly the bug this
# constant exists to prevent. Confirmed by real submissions to
# Forte-Enterprise-1 on 2026-08-30: a probe circuit at 2,583 native gates
# was rejected at CreateQuantumTask validation with the verbatim error
#     "The total number of gates in the circuit is too high. Please ensure
#      that the number of gates does not exceed 2,000. It is currently
#      2,583."
# while a 1,915-gate probe was accepted and a 3,231-gate probe was also
# rejected (see braket_direct/ionq_20260601/probe_gate_limit.py). The
# counted quantity is exactly count_native(qc)["n_gates"] == n_1q + n_2q
# (measurements are NOT counted) -- verified by matching AWS's reported
# count to this repo's own count_native() on all three probe circuits.
# This limit is PER-CIRCUIT and SHOT-INDEPENDENT: it has nothing to do with
# N_SHOTS or SHOTS_RANGE above. AWS's own published documentation reportedly
# states 5,000; the real, observed-at-the-service ceiling is 2,000 --
# ground truth (CreateQuantumTask's actual rejection) beats the doc
# (CLAUDE.md: "ground truth beats prior claims -- including documentation").
#
# FORWARD-LOOKING WARNING: C2 and D2 can never take a full ZNE ladder under
# this limit. C2 (n_prec=5, n_targ=2) is 934 gates at lambda=1 but 2,802 at
# lambda=3; D2 (n_prec=4, n_targ=3, the tightest family at 1,296 gates,
# lambda=1) crosses 2,000 by roughly lambda=1.5 (1,944 gates) and is 3,888
# at lambda=3. Anyone later adding a ZNE ladder to a high-n_targ family
# will hit this DEVICE limit, not the MAX_TOTAL_GATES budget above.
MAX_GATES_PER_CIRCUIT = 2_000

CONN_CACHE = os.path.join(os.path.dirname(__file__), "device_caps.json")


# -- Haar-random unitary ------------------------------------------------------

def haar_random_unitary(d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample a d x d Haar-distributed unitary (Mezzadri, arXiv:math-ph/0609050):
    complex Ginibre -> QR -> phase-fix diag(R)/|diag(R)|. The phase fix is
    mandatory; raw QR is not Haar.
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag_R = np.diag(R)
    return Q * (diag_R / np.abs(diag_R))


def _stable_matrix_power(U: np.ndarray, power: int) -> np.ndarray:
    """
    U**power, re-projected onto the nearest unitary via SVD. Repeated squaring
    in float64 squares the error each step, so at large n_prec the raw power
    drifts far enough from unitary that qiskit's UnitaryGate check rejects it.
    """
    M = np.linalg.matrix_power(U, power)
    W, _, Vh = np.linalg.svd(M)
    return W @ Vh


# -- Circuit builders ----------------------------------------------------------

def build_qpe_stage(n_prec: int, U: np.ndarray) -> QuantumCircuit:
    """
    One QPE stage: H on every precision qubit, controlled-U^{2^{n_prec-1-k}}
    from precision qubit k, then inverse QFT on the precision register.
    Precision qubit 0 is the MSB of the measured integer.

    Qubit order expected by the caller: [prec[0..n_prec-1], targ[0..n_targ-1]].
    """
    n_targ = int(round(np.log2(U.shape[0])))
    assert 2 ** n_targ == U.shape[0], "U dimension must be a power of 2"

    qc   = QuantumCircuit(n_prec + n_targ, name="QPE")
    prec = list(range(n_prec))
    targ = list(range(n_prec, n_prec + n_targ))

    qc.h(prec)
    for k in range(n_prec):
        power = 2 ** (n_prec - 1 - k)
        cU = UnitaryGate(_stable_matrix_power(U, power), label=f"U^{power}").control(1)
        qc.append(cU, [prec[k]] + targ)

    # QFTGate is little-endian (qubit k weight 2^k) but prec is built
    # MSB-first (prec[0] weight 2^(n_prec-1)) by the loop above -- must
    # reverse the index order. CLAUDE.md known-bug list item 1/2 and this
    # repo's own regression (docs/LOG.md, 2026-08-28): the IonQ scripts had
    # exactly this bug unpatched until earlier today. Fixed here from the
    # start; DO NOT drop the [::-1].
    qc.append(QFTGate(n_prec).inverse(), prec[::-1])
    return qc


def known_phase_unitary(n_targ: int, phi: float) -> np.ndarray:
    """
    A diagonal U whose |1...1> eigenstate carries EXACTLY the phase phi: a
    P(2*pi*phi/n_targ) rotation on each target qubit, so the phases add to
    2*pi*phi on |1...1>.

    With a known phi you know the answer: noiseless QPE puts all its weight
    in bin round(phi * 2^n_prec), so any spread is measured error -- the only
    way to say whether a mitigation technique actually helped.
    """
    per = np.exp(2j * np.pi * phi / n_targ)
    d   = 2 ** n_targ
    diag = np.ones(d, dtype=complex)
    for j in range(d):
        diag[j] = per ** bin(j).count("1")
    return np.diag(diag)


def build_qpe_circuit(n_prec: int, n_targ: int, U: np.ndarray,
                      target_init_seed: int | None = None,
                      eigenstate: bool = False) -> QuantumCircuit:
    """
    Textbook single-stage QPE. ONE terminal measurement of the precision
    register -- no mid-circuit measurement, no second stage. This is the
    primary vehicle for the mitigation study, mirroring
    rigetti_qpuf_common.py's build_qpe_circuit / submit_qpe.py: a technique's
    benefit is only measurable against a KNOWN answer.

    Single flat QuantumRegister(n_total, "q") + single ClassicalRegister, per
    CLAUDE.md known-bug item 3 (Braket requires this in exported OpenQASM 3).

        q[0 .. n_targ-1]              targ : input state
        q[n_targ .. n_targ+n_prec-1]  prec : QPE ancillae -> c[0 .. n_prec-1]

    eigenstate=True prepares |1...1> on the target register (pair with
    known_phase_unitary -- an exact eigenvector, so QPE returns one sharp
    bin with zero noise).

    eigenstate=False prepares the same seeded Haar-random target state the
    IonQ QPUF scripts use (ry/rz per target qubit), so a QPE run is directly
    comparable to a QPUF run on the same U.
    """
    n_total = n_targ + n_prec
    q  = QuantumRegister(n_total, "q")
    c  = ClassicalRegister(n_prec, "c")
    qc = QuantumCircuit(q, c)

    targ_idx = list(range(0, n_targ))
    prec_idx = list(range(n_targ, n_targ + n_prec))

    if eigenstate:
        for i in targ_idx:
            qc.x(q[i])
    else:
        if target_init_seed is None:
            raise ValueError("target_init_seed is required when eigenstate=False")
        init_rng = np.random.default_rng(seed=target_init_seed)
        for i in targ_idx:
            qc.ry(init_rng.uniform(0, np.pi),     q[i])
            qc.rz(init_rng.uniform(0, 2 * np.pi), q[i])

    qc.append(build_qpe_stage(n_prec, U), prec_idx + targ_idx)
    qc.measure([q[i] for i in prec_idx], [c[k] for k in range(n_prec)])
    return qc


def ideal_qpe_distribution(qc_logical: QuantumCircuit, n_prec: int,
                           n_targ: int, max_qubits: int = 22) -> dict[int, float] | None:
    """
    Noiseless QPE output distribution over the measured integer m, by exact
    statevector simulation of the pre-transpile circuit -- the reference
    hardware counts get scored against. Returns None above `max_qubits`.

    Bit-order note (same trap as counts_to_m in mitig_analysis.py):
    Statevector.probabilities(qargs=...) makes qargs[0] the LEAST
    significant bit of the returned index. Our prec[0] is the MOST
    significant bit of m, so pass the precision indices REVERSED.
    """
    if qc_logical.num_qubits > max_qubits:
        return None
    from qiskit.quantum_info import Statevector

    bare = qc_logical.remove_final_measurements(inplace=False)
    prec_idx = list(range(n_targ, n_targ + n_prec))
    probs = Statevector.from_instruction(bare).probabilities(qargs=prec_idx[::-1])
    return {m: float(p) for m, p in enumerate(probs) if p > 1e-12}


# -- Transpilation + counting --------------------------------------------------

def transpile_for_ionq(qc: QuantumCircuit, optimization_level: int = 1) -> QuantumCircuit:
    """
    Decompose to rz/rx/rxx. No coupling map: Forte-Enterprise-1 is
    all-to-all, so there is nothing to route (contrast with
    rigetti_qpuf_common.py's transpile_for_rigetti, which routes onto a
    fixed lattice).

    Call this ONCE, before DDD insertion and before ZNE folding, and never
    again afterwards. error_mitig_ionq.apply_ddd / fold operate on the
    native gates produced here; a second transpile pass at
    optimization_level > 0 is exactly what would cancel a DD pulse pair or a
    ZNE fold's C C^dag C -- see MITIGATION_BRIEF_IONQ.md Section 9.1, item 3
    of "what would settle it".
    """
    return transpile(qc, basis_gates=IONQ_BASIS, optimization_level=optimization_level)


def count_native(qc_hw: QuantumCircuit) -> dict:
    """
    Native-gate profile of a transpiled circuit, in the rz/rx/rxx basis.
    IonQ runtime tracks total gate COUNT (trapped-ion gates are serialized),
    not circuit depth -- see module docstring point 4.
    """
    ops    = qc_hw.count_ops()
    n_1q   = ops.get("rz", 0) + ops.get("rx", 0)
    n_2q   = ops.get("rxx", 0)
    n_meas = ops.get("measure", 0)
    used   = {q for inst in qc_hw.data for q in inst.qubits}

    return {
        "n_1q":   n_1q,
        "n_2q":   n_2q,
        "n_meas": n_meas,
        "n_gates": n_1q + n_2q,
        "depth":  qc_hw.depth(),
        "n_qubits_used": len(used),
        "ops":    {k: int(v) for k, v in ops.items()},
    }


def assert_gate_count_ok(qc_hw: QuantumCircuit, label: str = "") -> None:
    """
    Raise if `qc_hw` exceeds MAX_GATES_PER_CIRCUIT, the real per-circuit
    AWS/IonQ CreateQuantumTask limit (see that constant's comment). This is
    NOT the same check as MAX_TOTAL_GATES: that one is a shot-scaled project
    budget guess, this one is the device's actual per-circuit rejection
    threshold and does not depend on N_SHOTS at all. Call once per submitted
    circuit -- including every ZNE fold scale, since folding is exactly
    what pushes a circuit over the limit.
    """
    n_gates = count_native(qc_hw)["n_gates"]
    if n_gates > MAX_GATES_PER_CIRCUIT:
        where = f" '{label}'" if label else ""
        raise ValueError(
            f"circuit{where} has {n_gates:,} native gates (n_1q+n_2q), "
            f"exceeding the AWS/IonQ device limit of "
            f"{MAX_GATES_PER_CIRCUIT:,} gates/circuit "
            f"(ionq_mitigation_common.MAX_GATES_PER_CIRCUIT -- an "
            f"empirically confirmed CreateQuantumTask rejection threshold, "
            f"NOT the MAX_TOTAL_GATES project budget). Reduce N_PREC, "
            f"N_TARG, or the ZNE fold scale for this circuit."
        )


def measured_physical_qubits(qc_hw: QuantumCircuit) -> list[int]:
    """
    Qubit index feeding each clbit, ordered by clbit index. Needed because
    Braket's measurement_counts is ordered by ascending qubit index, which
    has no fixed relation to clbit order once a circuit has been rebuilt by
    fold()/apply_ddd() -- see mitig_analysis.py's module docstring.
    """
    out: dict[int, int] = {}
    for inst in qc_hw.data:
        if inst.operation.name == "measure":
            out[qc_hw.find_bit(inst.clbits[0]).index] = qc_hw.find_bit(inst.qubits[0]).index
    return [out[k] for k in sorted(out)]


# -- Runtime + fidelity model ---------------------------------------------------

def estimate_runtime_s(profile: dict, n_shots: int) -> float:
    """
    Gate-count-based wall-clock estimate, matching ionq_noise_mitig.py's
    _runtime_estimate_s. Trapped-ion gates are serialized per shot, so
    runtime tracks (1q gates * t_1q + 2q gates * t_2q + readout) * shots.
    """
    per_shot_gates = profile["n_1q"] * ONE_QUBIT_TIME_S + profile["n_2q"] * TWO_QUBIT_TIME_S
    return STARTUP_TIME_S + n_shots * (per_shot_gates + READOUT_TIME_S)


def estimate_fidelity(profile: dict) -> float:
    """Crude product-of-gate-fidelities survival probability."""
    return (F2Q ** profile["n_2q"]) * (F1Q ** profile["n_1q"])


def fmt_time(t_s: float) -> str:
    return f"{t_s:.1f} s  ({t_s/60:.2f} min, {t_s/3600:.3f} h)"


def print_circuit_report(title: str, profile: dict, n_shots: int,
                         n_logical_qubits: int, caps_are_real: bool,
                         qubit_formula: str = "n_targ + n_prec") -> dict:
    """Human-readable circuit + cost report. Returns {"runtime_s": ...}."""
    rt_s = estimate_runtime_s(profile, n_shots)
    fid  = estimate_fidelity(profile)
    total_gates = profile["n_gates"] * n_shots

    print("-" * 78)
    print(title)
    print("-" * 78)
    print(f"  logical qubits ({qubit_formula}){' ' * max(0, 18 - len(qubit_formula))}: {n_logical_qubits}")
    print(f"  physical qubits touched            : {profile['n_qubits_used']}")
    print(f"  1-qubit gates (rz + rx)            : {profile['n_1q']:,}")
    print(f"  2-qubit gates (rxx)                : {profile['n_2q']:,}")
    print(f"  total gates                        : {profile['n_gates']:,}"
          f"  [device cap {MAX_GATES_PER_CIRCUIT:,}]")
    print(f"  measurements                       : {profile['n_meas']}")
    print(f"  shots                              : {n_shots:,}")
    print(f"  total gates * shots                : {total_gates:,}  [cap {MAX_TOTAL_GATES:,}]")
    print(f"  est. circuit fidelity              : {fid:.4f}"
          f"{'   <-- effectively noise' if fid < 0.05 else ''}")
    print(f"  est. runtime                       : {fmt_time(rt_s)}")
    if not caps_are_real:
        print("  NOTE: device_caps.json not found -- run query_device_caps.py on")
        print("        the DCV to confirm F1Q/F2Q/timing against live calibration")
        print("        (see MITIGATION_BRIEF_IONQ.md Section 9.2).")
    return {"runtime_s": rt_s, "total_gates": total_gates, "fidelity": fid}


# -- OpenQASM 3 export for Braket -----------------------------------------------

def _rewrite_qasm_for_braket(qasm: str) -> str:
    """
    Braket's OpenQASM 3 parser accepts only its own built-in gates: no
    `include`, no user-defined `gate` blocks. qiskit's `rxx` is not a Braket
    built-in, but Braket's `xx(angle)` is the identical matrix
    (xx(theta) = exp(-i*theta/2 X(x)X) = qiskit RXX(theta); verified in
    ionq_noise_mitig.py / submit_QPUF_ntarg.py). So: drop the emitted `gate
    rxx {...}` definition block and rewrite every `rxx(` call to `xx(`.
    """
    import re
    out_lines, lines, i = [], qasm.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"\s*gate\s+rxx\b", line):
            depth = line.count("{") - line.count("}")
            i += 1
            while i < len(lines) and depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
            continue
        line = re.sub(r"\brxx\s*\(", "xx(", line)
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines) + ("\n" if qasm.endswith("\n") else "")


BRAKET_BUILTIN_GATES = ["h", "cx", "cnot", "rx", "ry", "rz", "xx",
                        "x", "y", "z", "s", "t", "swap", "i"]


def to_braket_qasm(qc_hw: QuantumCircuit, verbatim: bool = False,
                   physical: bool | None = None) -> str:
    """
    Export to OpenQASM 3 that Braket's parser accepts, rewriting rxx -> xx
    (see _rewrite_qasm_for_braket).

    verbatim=True wraps the gate body in
    `#pragma braket verbatim / box { ... }`. RESOLVED 2026-08-28 (was
    UNRESOLVED -- see MITIGATION_BRIEF_IONQ.md Section 9.1 for the prior
    open-question framing, now stale): device_caps.json /
    device_properties_raw.json, fetched live from Forte-Enterprise-1, show
    supportsPartialVerbatimBox=False with plain `verbatim` present in
    supportedPragmas -- so it is whole-circuit-box-or-nothing, no partial
    wrapping, which is exactly the shape this function already produces when
    verbatim=True (there is no partial-box code path here to be wrong about).
    Also confirmed: maximumQubitArrays=1 in the same caps dump, i.e. CLAUDE.md
    known-bug item 3 (single flat QuantumRegister, no separate named
    registers) is now a device-confirmed requirement, not repo lore.

    Caution this does NOT resolve: nativeGateSet is GPI/GPI2/ZZ while
    supportedOperations is the rz/rx/xx-level list this function emits, so a
    basis translation happens server-side even inside a verbatim box --
    verbatim suppresses optimization/rewriting of the gates we submit, it
    does not mean those gates ARE the physical pulses executed. Do not claim
    otherwise in analysis of hardware results.
    """
    import re
    from qiskit import qasm3

    if physical is None:
        physical = verbatim

    src = qasm3.dumps(qc_hw, includes=(), basis_gates=BRAKET_BUILTIN_GATES)
    src = _rewrite_qasm_for_braket(src)

    if not physical and not verbatim:
        return src

    def to_phys(s: str) -> str:
        return re.sub(r"\bq\[(\d+)\]", r"$\1", s)

    header, gates, measures = [], [], []
    for raw in src.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("OPENQASM") or line.startswith("include"):
            header.append(line)
        elif re.match(r"qubit\b|qubit\[", line):
            continue  # physical addressing declares no quantum register
        elif re.match(r"bit\b|bit\[", line):
            header.append(line)
        elif "measure" in line:
            measures.append(to_phys(line))
        else:
            gates.append(to_phys(line))

    out = list(header)
    if verbatim:
        out.append("#pragma braket verbatim")
        out.append("box {")
        out += ["  " + g for g in gates]
        out.append("}")
    else:
        out += gates
    out += measures
    return "\n".join(out) + "\n"


# device_caps.json supportedOperations, fetched live 2026-08-28 (this
# directory's device_caps.json / device_properties_raw.json). Note there is
# NO `rxx` here -- only bare `xx` -- and no `i` (identity). Under verbatim
# there is no server-side gate translation, so anything off this list fails
# the task outright rather than being silently recompiled; see
# assert_supported_gates below. The `i` gap matters the moment DDD is added
# to this campaign: error_mitig_ionq.py's DDD null control ("II") emits
# `circ.id(qubit)`, which is NOT in this list and will break under verbatim
# -- flagged there, not fixed here (no DDD arm in this campaign yet).
IONQ_SUPPORTED_OPS = frozenset([
    "x", "y", "z", "h", "s", "si", "t", "ti", "v", "vi",
    "rx", "ry", "rz", "cnot", "swap", "xx", "yy", "zz",
])


def assert_supported_gates(qasm_src: str) -> None:
    """
    Pre-submission guard: every gate token actually emitted in `qasm_src`
    must be in IONQ_SUPPORTED_OPS. Catches an illegal gate locally, before it
    ever costs a task -- verbatim submission gives no server-side rewriting
    to fall back on, so a gate off this list is a hard failure on the
    device, not a silent recompile.
    """
    import re
    bad = set()
    for raw in qasm_src.splitlines():
        line = raw.strip()
        if not line or line.startswith((
                "OPENQASM", "include", "bit", "qubit", "#pragma", "box", "}")):
            continue
        if "measure" in line:
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*[\(\s]", line)
        if m and m.group(1) not in IONQ_SUPPORTED_OPS:
            bad.add(m.group(1))
    if bad:
        raise ValueError(
            f"QASM contains gate(s) not in Forte-Enterprise-1's "
            f"supportedOperations (device_caps.json): {sorted(bad)}. Under "
            f"verbatim these are NOT server-side translated and would fail "
            f"the task outright, not be silently recompiled.")


# -- Device capabilities ---------------------------------------------------------

def fetch_device_caps(device) -> dict:
    """
    Pull the capabilities we care about off a live AwsDevice: qubit count,
    native gate set, connectivity (expected fully-connected for IonQ), and
    the OpenQASM action flags -- in particular supportsPartialVerbatimBox,
    which is exactly the flag MITIGATION_BRIEF_IONQ.md Section 9.1 says has
    never been captured for this device in this repository. Run
    query_device_caps.py and save its output to close that gap.
    """
    p = device.properties
    para = p.paradigm
    graph = {}
    conn = getattr(para, "connectivity", None)
    if conn is not None and getattr(conn, "connectivityGraph", None):
        graph = {str(k): [str(v) for v in vs]
                 for k, vs in conn.connectivityGraph.items()}

    action = p.action.get("braket.ir.openqasm.program")
    flags = {}
    if action is not None:
        for attr in ("supportedOperations", "supportedPragmas",
                     "forbiddenPragmas", "supportedResultTypes",
                     "supportedModifiers", "supportPhysicalQubits",
                     "requiresContiguousQubitIndices",
                     "requiresAllQubitsMeasurement",
                     "supportsPartialVerbatimBox",
                     "supportsUnassignedMeasurements"):
            val = getattr(action, attr, None)
            if val is not None:
                flags[attr] = val if isinstance(val, (bool, int, str)) else [str(v) for v in val]

    return {
        "device_name":     device.name,
        "device_arn":      DEVICE_ARN,
        "qubit_count":     para.qubitCount,
        "native_gate_set": [str(g) for g in (getattr(para, "nativeGateSet", None) or [])],
        "connectivity":    graph,
        "fully_connected": bool(getattr(conn, "fullyConnected", False)) if conn else False,
        "openqasm":        flags,
    }


def load_device_caps() -> tuple[dict, bool]:
    """
    Return (caps, is_real). Reads device_caps.json if query_device_caps.py
    has been run and its output saved into this directory; otherwise
    synthesises an all-to-all placeholder.

    36 qubits per CLAUDE.md's stated Forte-Enterprise-1 characterization
    (Section 0) -- NOT independently re-confirmed by this file; no live
    device_caps.json exists in this campaign yet (Section 9.1).
    """
    if os.path.exists(CONN_CACHE):
        with open(CONN_CACHE) as f:
            return json.load(f), True
    n = 36
    return {
        "device_name":     DEVICE_NAME + " (PLACEHOLDER)",
        "device_arn":      DEVICE_ARN,
        "qubit_count":     n,
        "native_gate_set": IONQ_BASIS,
        "connectivity":    {},
        "fully_connected": True,
        "openqasm":        {},
    }, False


# -- Job logging and reporting ---------------------------------------------------

def encode_unitary(U: np.ndarray) -> dict:
    """Serialize a complex matrix to JSON-friendly real/imag lists."""
    return {"shape": list(U.shape), "real": U.real.tolist(), "imag": U.imag.tolist()}


def append_job_log(results_dir: str, record: dict) -> str:
    """Append one JSON record per line to <results_dir>/job_log.txt."""
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "job_log.txt")

    needs_nl = os.path.exists(log_file) and os.path.getsize(log_file) > 0
    if needs_nl:
        with open(log_file, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b"\n"
    with open(log_file, "a") as f:
        if needs_nl:
            f.write("\n")
        f.write(json.dumps(record) + "\n")
    return log_file


def task_tags(workload: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """
    Resource tags for one on-demand task. No ReservationId tag (contrast
    with rigetti_qpuf_common.py's task_tags): this campaign has no
    reservation to associate with.
    """
    tags = {"Campaign": "ionq-error-mitigation-20260828", "Workload": workload}
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


def confirm_submit(question: str, assume_yes: bool = False) -> bool:
    """
    Final go/no-go before anything is billed. True means submit. See
    rigetti_qpuf_common.py's identically-named function for why this is not
    a bare input() -- same reasoning applies verbatim here.
    """
    if assume_yes:
        print(f"\n{question} y   (--yes)")
        return True

    if not sys.stdin.isatty():
        print(f"\n{question}")
        print("\nERROR: stdin is not a terminal, so this question cannot be "
              "answered interactively.")
        print("       Re-run directly in a terminal, or pass --yes to submit "
              "without the prompt.")
        return False

    try:
        answer = input(question).strip().lower()
    except EOFError:
        print("\n\nERROR: end of input while waiting for confirmation "
              "(stdin closed).")
        print("       Re-run in a terminal, or pass --yes to skip the prompt.")
        return False

    if answer in ("y", "yes"):
        return True
    if answer == "":
        print("Aborted -- nothing submitted. (Enter alone means No here; "
              "type 'y' to submit, or pass --yes.)")
    else:
        print(f"Aborted -- nothing submitted. (Got {answer!r}, expected 'y'.)")
    return False
