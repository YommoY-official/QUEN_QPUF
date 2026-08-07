#!/usr/bin/env python3
"""
rigetti_qpuf_common.py
======================
Shared machinery for running the two-stage PE-QPUF on Rigetti
Cepheus-1-108Q under a Braket Direct reservation, with client-side noise
mitigation (ZNE + readout-error mitigation).

Everything that submit_test.py, submit_qpuf_mitigation.py and
checkRetrieve.py need lives here so all three agree on the circuit, the
gate counts and the runtime model.


Why this differs from the IonQ (Forte-Enterprise-1) scripts
-----------------------------------------------------------
1. NO mid-circuit measurement / reset.
   Braket exposes MCM+reset only on IonQ Forte (behind
   EnableExperimentalCapability). Rigetti does not offer it. So the QPUF
   uses the DEFERRED-MEASUREMENT form: two disjoint precision registers,
   both measured terminally. Since prec_a is never reused after stage 1,
   this is statistically identical to mid-circuit measuring it -- and it
   keeps the circuit a single unitary block, which is exactly what ZNE
   folding requires.

2. FIXED LATTICE, not all-to-all.
   IonQ's trap gives every qubit-pair a direct 2q gate. Cepheus is a
   superconducting lattice, so the transpiler must insert SWAPs. QPE is
   the natural worst case: every precision qubit must reach the target,
   and the inverse QFT is all-to-all within the precision register.
   Measured penalty (SABRE routing, degree-4 lattice, two-stage QPUF):
   only ~1.0-1.35x the all-to-all 2q count for N_PREC <= 6, because 108
   qubits give the router plenty of room to spread out. So routing is
   NOT the thing that kills this circuit on Rigetti -- per-gate fidelity
   is. We still transpile against the REAL connectivity graph, because
   an unrouted count would be a guess and verbatim mode needs real
   lattice edges anyway.

3. NO vendor-native error mitigation.
   IonQ's `Debias` is IonQ-only. On Rigetti we do mitigation client-side:
     - ZNE  : global gate folding, C -> C (C^dag C)^k, odd lambda = 2k+1,
              submitted as one task per lambda, extrapolated offline.
     - REM  : two readout-calibration circuits (all-|0>, all-|1>) on the
              same physical qubits, giving per-qubit confusion matrices
              that the analysis inverts.

4. GATES RUN IN PARALLEL.
   On a trapped-ion chain gates are serialized, so IonQ runtime tracks
   the GATE COUNT. On a superconducting lattice, disjoint gates run in
   the same cycle, so runtime tracks the CIRCUIT DEPTH. Both estimates
   are reported; the depth-based one is the realistic one for Rigetti.

5. VERBATIM MODE MATTERS FOR ZNE.
   Non-verbatim submissions are recompiled by Rigetti's Quilc, which will
   happily cancel the C^dag C pairs that ZNE folding just inserted --
   silently reducing every lambda back to lambda=1. Submitting inside a
   `#pragma braket verbatim box` disables that recompilation. Verbatim
   requires physical qubits and native gates only, which is why we
   transpile against the real coupling map ourselves.
"""

import json
import os

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.transpiler import CouplingMap

# -- DEVICE --------------------------------------------------------------------
DEVICE_NAME = "Cepheus-1-108Q"
DEVICE_ARN  = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
AWS_REGION  = "us-west-1"

# Braket Direct reservation ARN for the booked window. Leave "" to submit
# on-demand (which costs per-task money outside the window).
RES_ARN = "arn:aws:braket:us-west-1:767397707562:reservation/339f1ef2-2aec-459e-8e09-8a945b52755a"

# -- NATIVE GATE SET -----------------------------------------------------------
# Rigetti's Braket native set is rx(k*pi/2) / rz(theta) / cz (+ iswap, xy on
# some chips). We transpile to rz/rx/cz because that is the safest common
# subset and the one verbatim mode is guaranteed to accept.
# Run query_device_caps.py on the DCV to confirm against the live device; it
# writes device_caps.json, which load_native_basis() picks up if present.
RIGETTI_BASIS = ["rz", "rx", "cz"]

# -- GATE-TIME MODEL -----------------------------------------------------------
# PLACEHOLDERS -- Rigetti does not publish per-gate durations through Braket.
# Order of magnitude is right (ns-scale gates, us-scale readout), but the
# absolute runtime is dominated by SHOT_RESET_S and STARTUP_TIME_S, both of
# which you should re-fit from the first real task (checkRetrieve.py prints
# measured task time and Rigetti's own qpuRuntimeEstimation next to these).
ONE_QUBIT_TIME_S = 40e-9      # rx / rz
TWO_QUBIT_TIME_S = 180e-9     # cz
READOUT_TIME_S   = 2.0e-6     # per readout event
SHOT_RESET_S     = 200e-6     # inter-shot reset / relaxation delay (DOMINANT)
STARTUP_TIME_S   = 15.0       # per-task overhead: Quilc compile + chip load

# Per-gate fidelity model for the "is this circuit worth running" check.
# TUNE to the live calibration (query_device_caps.py dumps the per-qubit and
# per-edge fidelities Rigetti reports).
F2Q = 0.97       # CZ  -- Cepheus-class median
F1Q = 0.999      # rx

CONN_CACHE  = os.path.join(os.path.dirname(__file__), "device_caps.json")


# -- Haar-random unitary -------------------------------------------------------

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

    # ENDIANNESS: the controlled-U loop above gives prec[k] the weight
    # 2^(n_prec-1-k), i.e. prec[0] is the MSB. Qiskit's QFTGate is
    # little-endian -- it gives qubit k the weight 2^k. Feeding it `prec`
    # directly mismatches those conventions, and the result is NOT a harmless
    # relabelling of the output bins: it SPREADS a sharp phase across many
    # bins. (Checked: phi=1/8, n_prec=5, exact eigenstate gives bin 4 with
    # p=1.0 reversed, versus a ~0.18 smear unreversed.) Passing the register
    # reversed lines the two conventions up.
    qc.append(QFTGate(n_prec).inverse(), prec[::-1])
    return qc


def build_qpuf_two_stage(n_prec: int, n_targ: int, U: np.ndarray,
                         target_init_seed: int) -> QuantumCircuit:
    """
    Two-stage PE-QPUF -- the same construction submitted to Forte-Enterprise-1,
    in its deferred-measurement form (see module docstring, point 1).

        q[0 .. n_targ-1]                        targ    : Haar-random init state
        q[n_targ .. n_targ+n_prec-1]            prec_a  : QPE stage 1 -> c[0:n_prec]
        q[n_targ+n_prec .. n_targ+2*n_prec-1]   prec_b  : QPE stage 2 -> c[n_prec:]

    Both QPE stages use the SAME Haar-random U. Each shot yields two integers
    m1 (prec_a) and m2 (prec_b); ideally m1 == m2, because stage 1 collapses
    the target onto an eigenstate of U and stage 2 re-reads that same
    eigenphase. The QPUF response is the m1/m2 agreement statistic.

    Single quantum + single classical register, as Braket OpenQASM 3 requires.
    """
    n_total = n_targ + 2 * n_prec
    q  = QuantumRegister(n_total, "q")
    c  = ClassicalRegister(2 * n_prec, "c")
    qc = QuantumCircuit(q, c)

    targ_idx   = list(range(0, n_targ))
    prec_a_idx = list(range(n_targ,          n_targ + n_prec))
    prec_b_idx = list(range(n_targ + n_prec, n_targ + 2 * n_prec))

    # Target initialisation: one RY+RZ per target qubit, seeded independently
    # of U so the state and the unitary are separately reproducible.
    init_rng = np.random.default_rng(seed=target_init_seed)
    for i in targ_idx:
        qc.ry(init_rng.uniform(0, np.pi),     q[i])
        qc.rz(init_rng.uniform(0, 2 * np.pi), q[i])

    qc.append(build_qpe_stage(n_prec, U), prec_a_idx + targ_idx)   # stage 1
    qc.append(build_qpe_stage(n_prec, U), prec_b_idx + targ_idx)   # stage 2

    qc.measure([q[i] for i in prec_a_idx], [c[k] for k in range(n_prec)])
    qc.measure([q[i] for i in prec_b_idx], [c[n_prec + k] for k in range(n_prec)])
    return qc


# -- Plain QPE (single stage) --------------------------------------------------

def known_phase_unitary(n_targ: int, phi: float) -> np.ndarray:
    """
    A diagonal U whose |1...1> eigenstate carries EXACTLY the phase phi:
    a P(2*pi*phi/n_targ) rotation on each of the n_targ target qubits, so the
    phases add to 2*pi*phi on |1...1>.

    Why bother instead of a Haar draw: with a known phi you know the answer.
    QPE should put all its weight in bin round(phi * 2^n_prec), so any spread
    is measured error -- which is the only way to say whether the mitigation
    actually helped. With n_targ=1 and phi=1/8 this is the textbook T-gate
    QPE.
    """
    per = np.exp(2j * np.pi * phi / n_targ)
    d   = 2 ** n_targ
    diag = np.ones(d, dtype=complex)
    for j in range(d):
        # |j> picks up `per` once per target qubit that is set.
        diag[j] = per ** bin(j).count("1")
    return np.diag(diag)


def build_qpe_circuit(n_prec: int, n_targ: int, U: np.ndarray,
                      target_init_seed: int | None = None,
                      eigenstate: bool = False) -> QuantumCircuit:
    """
    Textbook single-stage QPE. ONE terminal measurement of the precision
    register -- no mid-circuit measurement, no second stage, nothing about
    the readout that needs interpreting.

        q[0 .. n_targ-1]              targ : input state
        q[n_targ .. n_targ+n_prec-1]  prec : QPE ancillae -> c[0 .. n_prec-1]

    Precision qubit 0 is the MSB, so the measured integer is
        m = sum_k c[k] * 2^(n_prec-1-k)
    and the phase estimate is m / 2^n_prec.

    eigenstate=True prepares |1...1> on the target register (use with
    known_phase_unitary -- that state is an exact eigenvector, so QPE returns
    a single sharp bin).

    eigenstate=False prepares the same seeded Haar-random target state the
    QPUF scripts use, so a QPE run is directly comparable to the QPUF run on
    the same U. A generic state is a superposition of eigenvectors, so the
    output spreads over several eigenphases even with zero noise.
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
    statevector simulation of the pre-transpile circuit. This is the
    reference the hardware counts get scored against.

    Returns None above `max_qubits` (2^n statevector).

    Bit-order note: Statevector.probabilities(qargs=...) makes qargs[0] the
    LEAST significant bit of the returned index. Our prec[0] is the MOST
    significant bit of m, so we pass the precision indices REVERSED and the
    index then equals m directly.
    """
    if qc_logical.num_qubits > max_qubits:
        return None
    from qiskit.quantum_info import Statevector

    bare = qc_logical.remove_final_measurements(inplace=False)
    prec_idx = list(range(n_targ, n_targ + n_prec))
    probs = Statevector.from_instruction(bare).probabilities(qargs=prec_idx[::-1])
    return {m: float(p) for m, p in enumerate(probs) if p > 1e-12}


# -- Device connectivity -------------------------------------------------------

def fetch_device_caps(device) -> dict:
    """
    Pull the capabilities we care about off a live AwsDevice: qubit count,
    native gate set, connectivity graph, and the OpenQASM action flags that
    decide whether verbatim mode and sparse qubit indices are allowed.
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
                     "supportedResultTypes", "supportPhysicalQubits",
                     "requiresContiguousQubitIndices",
                     "requiresAllQubitsMeasurement",
                     "supportsPartialVerbatimBox",
                     "supportsUnassignedMeasurements"):
            val = getattr(action, attr, None)
            if val is not None:
                flags[attr] = val if isinstance(val, (bool, int, str)) else [str(v) for v in val]

    # Per-qubit and per-edge calibration. ErrorMitigation.select_best_qubits
    # scores candidate placements with these, so persist the FULL specs, not
    # just medians -- a median tells you nothing about which chiplet is good
    # today.
    specs = {}
    provider = getattr(p, "provider", None)
    raw = getattr(provider, "specs", None) if provider is not None else None
    if isinstance(raw, dict):
        for group in ("1Q", "2Q"):
            entry = raw.get(group)
            if isinstance(entry, dict):
                specs[group] = {
                    str(k): {kk: vv for kk, vv in v.items()
                             if isinstance(vv, (int, float))}
                    for k, v in entry.items() if isinstance(v, dict)
                }

    exec_windows = []
    service = getattr(p, "service", None)
    for w in (getattr(service, "executionWindows", None) or []):
        exec_windows.append({
            "executionDay": str(getattr(w, "executionDay", "")),
            "windowStartHour": str(getattr(w, "windowStartHour", "")),
            "windowEndHour":   str(getattr(w, "windowEndHour", "")),
        })
    shots_range = getattr(service, "shotsRange", None)

    return {
        "device_name":     device.name,
        "device_arn":      DEVICE_ARN,
        "qubit_count":     para.qubitCount,
        "native_gate_set": [str(g) for g in (getattr(para, "nativeGateSet", None) or [])],
        "connectivity":    graph,
        "fully_connected": bool(getattr(conn, "fullyConnected", False)) if conn else False,
        "openqasm":        flags,
        "specs":           specs,
        "execution_windows": exec_windows,
        "shots_range":     list(shots_range) if shots_range else None,
    }


def _placeholder_grid(n_qubits: int, width: int = 12) -> dict:
    """
    Square-lattice stand-in used ONLY when device_caps.json is absent (i.e.
    offline, no AWS credentials). Rigetti chips are square lattices with
    degree <= 4, so this gives the right routing COST SCALE, but it is not
    the real chip. Run query_device_caps.py on the DCV to replace it.
    """
    graph: dict[str, list[str]] = {str(i): [] for i in range(n_qubits)}
    for i in range(n_qubits):
        r, col = divmod(i, width)
        if col + 1 < width and i + 1 < n_qubits:
            graph[str(i)].append(str(i + 1))
        if i + width < n_qubits:
            graph[str(i)].append(str(i + width))
    return graph


def load_device_caps() -> tuple[dict, bool]:
    """
    Return (caps, is_real). Reads device_caps.json if query_device_caps.py has
    been run; otherwise synthesises a placeholder so gate-count estimates can
    still be produced offline.
    """
    if os.path.exists(CONN_CACHE):
        with open(CONN_CACHE) as f:
            return json.load(f), True
    n = 108
    return {
        "device_name":     DEVICE_NAME + " (PLACEHOLDER)",
        "device_arn":      DEVICE_ARN,
        "qubit_count":     n,
        "native_gate_set": RIGETTI_BASIS,
        "connectivity":    _placeholder_grid(n),
        "fully_connected": False,
        "openqasm":        {},
    }, False


def coupling_map_from_caps(caps: dict) -> CouplingMap:
    """Symmetric CouplingMap over the device's connectivity graph."""
    edges = set()
    for a, nbrs in caps["connectivity"].items():
        for b in nbrs:
            i, j = int(a), int(b)
            edges.add((i, j))
            edges.add((j, i))
    return CouplingMap(sorted(edges))


# -- Transpilation + counting --------------------------------------------------

def transpile_for_rigetti(qc: QuantumCircuit, caps: dict,
                          optimization_level: int = 2,
                          seed_transpiler: int = 42) -> QuantumCircuit:
    """
    Decompose to rz/rx/cz AND route onto the real lattice. Routing is the
    whole point: without a coupling map the 2q count is a fantasy, because
    every controlled-U and every QFT rotation would be assumed free.

    The result has num_qubits == device qubit count, with qubit index i
    meaning PHYSICAL qubit i -- which is what verbatim submission needs.
    """
    return transpile(
        qc,
        basis_gates=RIGETTI_BASIS + ["measure"],
        coupling_map=coupling_map_from_caps(caps),
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )


def count_native(qc_hw: QuantumCircuit) -> dict:
    """
    Native-gate profile of a transpiled circuit.

    Both a serial and a parallel view are returned:
      n_1q / n_2q   -- total gate COUNT (what IonQ runtime tracks)
      depth_2q      -- CZ layers      (what Rigetti runtime tracks)
    """
    ops    = qc_hw.count_ops()
    n_1q   = ops.get("rz", 0) + ops.get("rx", 0) + ops.get("sx", 0) + ops.get("x", 0)
    n_2q   = ops.get("cz", 0) + ops.get("cx", 0) + ops.get("iswap", 0)
    n_meas = ops.get("measure", 0)

    depth_2q = qc_hw.depth(lambda instr: len(instr.qubits) == 2)
    used = {q for inst in qc_hw.data for q in inst.qubits}

    return {
        "n_1q":        n_1q,
        "n_2q":        n_2q,
        "n_meas":      n_meas,
        "n_gates":     n_1q + n_2q,
        "depth":       qc_hw.depth(),
        "depth_2q":    depth_2q,
        "n_qubits_used": len(used),
        "ops":         {k: int(v) for k, v in ops.items()},
    }


def measured_physical_qubits(qc_hw: QuantumCircuit) -> list[int]:
    """
    Physical qubit index feeding each clbit, ordered by clbit index. Needed so
    the readout-calibration circuits hit exactly the qubits the QPUF reads.
    """
    out: dict[int, int] = {}
    for inst in qc_hw.data:
        if inst.operation.name == "measure":
            out[qc_hw.find_bit(inst.clbits[0]).index] = qc_hw.find_bit(inst.qubits[0]).index
    return [out[k] for k in sorted(out)]


# -- Noise mitigation: ZNE folding ---------------------------------------------

def fold_global(qc_hw: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Global ZNE folding: C -> C (C^dag C)^k with lambda = 2k+1.

    The folded circuit has EXACTLY lambda times the gates of the base, so the
    effective noise scales by lambda while the ideal output is unchanged.
    Only odd integer lambda is allowed -- that is what keeps the fold exact
    (no partial/local folding, no fractional bookkeeping).

    IMPORTANT: fold AFTER transpiling, and do NOT re-transpile the result.
    Any optimiser that sees C^dag C will cancel it. That also means the
    folded circuit must be submitted inside a verbatim box, or Rigetti's
    server-side Quilc will undo the fold (see module docstring, point 5).

    Measurements are terminal and are NOT folded -- readout is charged once
    per shot regardless of lambda.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"ZNE scale must be an odd positive integer; got {scale}")

    base = qc_hw.remove_final_measurements(inplace=False)
    folded = base.copy()
    for _ in range((scale - 1) // 2):
        folded.compose(base.inverse(), inplace=True)
        folded.compose(base,           inplace=True)

    # Re-attach the original terminal measurements (same qubit->clbit map).
    creg = ClassicalRegister(qc_hw.num_clbits, "c")
    out  = QuantumCircuit(QuantumRegister(qc_hw.num_qubits, "q"), creg)
    out.compose(folded, inplace=True)
    for cbit, qbit in enumerate(measured_physical_qubits(qc_hw)):
        out.measure(qbit, creg[cbit])
    return out


# -- Noise mitigation: readout calibration -------------------------------------

def readout_calibration_circuits(phys_qubits: list[int], n_device_qubits: int
                                 ) -> list[tuple[str, QuantumCircuit]]:
    """
    Two circuits that measure the readout confusion on exactly the qubits the
    QPUF reads out:

        cal0 : prepare |0...0>, measure  ->  gives p(1|0) per qubit
        cal1 : prepare |1...1>, measure  ->  gives p(0|1) per qubit

    That is the independent-qubit (tensored) readout model: one 2x2 confusion
    matrix per qubit, inverted at analysis time. It costs 2 extra tasks, not
    2^n, and readout crosstalk on Rigetti is weak enough that the tensored
    model is the standard choice.

    Circuits are built directly on physical indices -- they are trivially
    connectivity-compatible (single-qubit gates only), so they never need
    routing and can go into a verbatim box unchanged.
    """
    out = []
    for label, flip in (("cal0", False), ("cal1", True)):
        creg = ClassicalRegister(len(phys_qubits), "c")
        qc = QuantumCircuit(QuantumRegister(n_device_qubits, "q"), creg)
        for k, phys in enumerate(phys_qubits):
            if flip:
                qc.rx(np.pi, phys)          # native-basis X
            qc.measure(phys, creg[k])
        out.append((label, qc))
    return out


# -- Runtime model -------------------------------------------------------------

def estimate_runtime_s(profile: dict, n_shots: int) -> dict:
    """
    Two runtime estimates for one task.

      serial   -- every gate executed one after another. This is the IonQ-style
                 model and is a hard UPPER bound on Rigetti.
      parallel -- gates in a layer execute simultaneously, so the circuit takes
                 depth * gate-time. This is the realistic Rigetti model.

    Both add per-shot readout and the inter-shot reset delay, plus a fixed
    per-task startup. On Rigetti the per-shot reset delay usually dominates
    the gates entirely -- which is why shot count, not circuit size, drives
    the wall clock.
    """
    d1q = max(profile["depth"] - profile["depth_2q"], 0)

    t_gates_serial   = (profile["n_1q"] * ONE_QUBIT_TIME_S
                        + profile["n_2q"] * TWO_QUBIT_TIME_S)
    t_gates_parallel = (d1q * ONE_QUBIT_TIME_S
                        + profile["depth_2q"] * TWO_QUBIT_TIME_S)
    per_shot_fixed   = READOUT_TIME_S + SHOT_RESET_S

    return {
        "t_serial_s":     STARTUP_TIME_S + n_shots * (t_gates_serial + per_shot_fixed),
        "t_parallel_s":   STARTUP_TIME_S + n_shots * (t_gates_parallel + per_shot_fixed),
        "t_circuit_serial_s":   t_gates_serial,
        "t_circuit_parallel_s": t_gates_parallel,
    }


def estimate_fidelity(profile: dict) -> float:
    """Crude product-of-gate-fidelities survival probability."""
    return (F2Q ** profile["n_2q"]) * (F1Q ** profile["n_1q"])


def fmt_time(t_s: float) -> str:
    return f"{t_s:8.2f} s  ({t_s/60:6.2f} min)"


# -- OpenQASM 3 export for Braket ----------------------------------------------

BRAKET_BUILTIN_GATES = ["rx", "ry", "rz", "cz", "cnot", "cx", "h",
                        "x", "y", "z", "s", "t", "swap", "i", "iswap",
                        "cphaseshift", "xy"]


def to_braket_qasm(qc_hw: QuantumCircuit, verbatim: bool = False,
                   physical: bool | None = None) -> str:
    """
    Export to OpenQASM 3 that Braket's parser accepts.

    Braket allows only its own built-in gates: no `include` statements and no
    user-defined `gate` blocks. Declaring the built-ins as basis_gates stops
    qiskit emitting definitions for them.

    physical=True addresses PHYSICAL qubits (`$3`) instead of a virtual
    register (`qubit[n] q; ... q[3]`). Use it for anything going to the QPU:
    we route the circuit ourselves, so the qubit indices are already the ones
    we want, and virtual indices would let Rigetti's compiler silently remap
    the circuit onto different physical qubits -- which would invalidate the
    readout-calibration circuits, since they must hit exactly the qubits the
    QPUF reads. Simulators have no physical qubits, so leave it False there.
    (qiskit already emits `$n` for a circuit that carries a transpiler
    layout; this makes the behaviour explicit and covers circuits built
    directly on physical indices, which carry no layout.)

    verbatim=True additionally wraps the gate body in
    `#pragma braket verbatim / box { ... }`, telling Braket to run it EXACTLY
    as written -- no Quilc recompilation, no gate cancellation. That is
    mandatory for ZNE folding to survive, and it is only valid because
    transpile_for_rigetti already produced native gates on real lattice
    edges. Measurements stay OUTSIDE the box; Braket does not accept them
    inside one.
    """
    import re
    from qiskit import qasm3

    if physical is None:
        physical = verbatim

    src = qasm3.dumps(qc_hw, includes=(), basis_gates=BRAKET_BUILTIN_GATES)

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
            # Physical addressing declares no quantum register.
            continue
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


# -- Job logging ---------------------------------------------------------------

def append_job_log(results_dir: str, record: dict) -> str:
    """Append one JSON record per line to <results_dir>/job_log.txt."""
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "job_log.txt")

    # Guarantee newline separation even if an earlier record was written
    # without a trailing newline.
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


def encode_unitary(U: np.ndarray) -> dict:
    """Serialize a complex matrix to JSON-friendly real/imag lists."""
    return {"shape": list(U.shape), "real": U.real.tolist(), "imag": U.imag.tolist()}


# -- Reporting -----------------------------------------------------------------

def print_circuit_report(title: str, profile: dict, n_shots: int,
                         n_logical_qubits: int, caps_are_real: bool,
                         qubit_formula: str = "n_targ + 2*n_prec") -> dict:
    """
    Human-readable circuit + cost report. Returns the runtime estimate.

    `qubit_formula` is just the label: the two-stage QPUF needs
    n_targ + 2*n_prec qubits, plain QPE needs n_targ + n_prec.
    """
    rt  = estimate_runtime_s(profile, n_shots)
    fid = estimate_fidelity(profile)

    print("-" * 78)
    print(title)
    print("-" * 78)
    print(f"  logical qubits ({qubit_formula}){' ' * max(0, 18 - len(qubit_formula))}: {n_logical_qubits}")
    print(f"  physical qubits touched            : {profile['n_qubits_used']}")
    print(f"  1-qubit gates (rz + rx)            : {profile['n_1q']:,}")
    print(f"  2-qubit gates (cz)                 : {profile['n_2q']:,}")
    print(f"  total gates                        : {profile['n_gates']:,}")
    print(f"  measurements                       : {profile['n_meas']}")
    print(f"  circuit depth                      : {profile['depth']:,}")
    print(f"  2-qubit depth (CZ layers)          : {profile['depth_2q']:,}")
    print(f"  shots                              : {n_shots:,}")
    print(f"  total gate-ops (gates x shots)     : {profile['n_gates'] * n_shots:,}")
    print(f"  est. circuit fidelity              : {fid:.4f}"
          f"{'   <-- effectively noise' if fid < 0.05 else ''}")
    print(f"  est. runtime (parallel/realistic)  : {fmt_time(rt['t_parallel_s'])}")
    print(f"  est. runtime (serial/upper bound)  : {fmt_time(rt['t_serial_s'])}")
    if not caps_are_real:
        print("  NOTE: using PLACEHOLDER lattice -- run query_device_caps.py on the")
        print("        DCV first so routing reflects the real Cepheus topology.")
    return rt
