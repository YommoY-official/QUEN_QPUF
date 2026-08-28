#!/usr/bin/env python3
"""
error_mitig_ionq.py
====================
Error-mitigation transforms for circuits headed to IonQ Forte-Enterprise-1,
adapted from braket_direct/rigetti_cepheus_20260807/error_mitig.py.

Same contract split as the Rigetti module:

    technique   fits circuit -> circuit?   extra tasks
    ---------   -----------------------   -----------
    DDD         YES                        0    pure circuit transform
    ZNE         one circuit PER lambda      N-1  needs N circuits + a fit
    REM         NO                          2+   calibration + post-processing

  - ZNE:  `fold()` really is circuit -> circuit, but a single folded circuit
          is not a mitigated result. `zne_circuit_set` gives you the N
          circuits; the extrapolation lives in mitig_analysis.py.
  - REM:  adds NO gates. `rem_calibration_circuits()` gives you the two (or
          2^n) calibration circuits; the inversion belongs in
          mitig_analysis.py.


What is DELIBERATELY NOT here, and why
---------------------------------------
Rigetti's ErrorMitigation carries an entire chiplet-topology / placement /
routing layer (CHIPLET_SIZE, chiplets(), select_best_qubits(), the SABRE
scoring loop, edge_fidelity/qubit_fidelity lookups against an ISA
calibration dict) because Cepheus-1-108Q is a fixed lattice where the
router's qubit choice changes the SWAP count by a large factor. NONE of
that applies here: Forte-Enterprise-1 is all-to-all, every controlled-U^k
and every inverse-QFT rotation is a direct 2q gate on whatever logical
qubits the circuit already names, and there is no placement decision to
make. Porting that code over would not be "adapting a template", it would
be dead code pretending a lattice constraint exists where none does -- so
it is dropped entirely, not stubbed out.


Rigetti specifics that do NOT carry over verbatim
--------------------------------------------------
  - DD pulse math (X = rx(pi), Y = rz(-pi/2) rx(pi) rz(pi/2)) is
    basis-agnostic and IS reused directly -- IONQ_BASIS is rz/rx/rxx, a
    superset of what these sequences need.
  - The RZ = 0 timing convention is NOT reused. Rigetti's `rz=0` is
    justified as a superconducting virtual-Z frame update; IonQ's own cost
    estimator (ionq_noise_mitig.py) charges every rz the full
    ONE_QUBIT_TIME_S instead, and the two models cannot both be right. See
    RZ_TIME_S below and MITIGATION_BRIEF_IONQ.md Section 9.2 -- this is an
    OPEN, UNRESOLVED question, not a settled port.
  - Whether IonQ's server-side compiler cancels ZNE folds or DD pulses the
    way Rigetti's Quilc does is UNCONFIRMED either way (brief Section 9.1).
    Nothing here claims a verbatim box is required, unnecessary, or even
    available on this device.
"""

import math
from collections import defaultdict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator

from ionq_mitigation_common import (
    IONQ_BASIS, ONE_QUBIT_TIME_S, TWO_QUBIT_TIME_S, READOUT_TIME_S,
    count_native, measured_physical_qubits,
)

# RZ timing: charge the full ONE_QUBIT_TIME_S, matching the convention
# already encoded in noise_mitigation/ionq_noise_mitig.py's cost estimator
# (n_1q = ops['rz'] + ops['rx'], both charged ONE_QUBIT_TIME_S). This is a
# DELIBERATE, NAMED CHOICE, not a default carried over from Rigetti: the
# Rigetti model (error_mitig.py) hard-codes rz=0 as a virtual-Z frame update
# on superconducting hardware, and that is very likely also true of IonQ's
# native gpi/gpi2 -- virtual-Z is standard on trapped ions too -- but
# "likely" is not "verified", and this project's own two cost models
# currently disagree about it (MITIGATION_BRIEF_IONQ.md Section 9.2: with
# rz=0 an XY4 sequence is 520 us and fits inside a 970 us window; with rz
# charged it is 1040 us and does not fit at all -- the DDD insertion
# decision literally flips on this constant). Defaulting to "charge it" here
# matches what the rest of this project's IonQ code currently assumes, so at
# least the two numbers this codebase produces agree with each other -- but
# whoever runs this campaign should resolve which convention is physically
# correct for Forte-Enterprise-1 before trusting any DDD idle-window
# decision this constant drives.
RZ_TIME_S = ONE_QUBIT_TIME_S


class ErrorMitigation:
    """
    Circuit-level error mitigation for IonQ Forte-Enterprise-1.

    Typical use:

        em = ErrorMitigation()
        qc_native = transpile_for_ionq(qc)            # ALWAYS, once, first
        qc_ddd    = em.apply_ddd(qc_native, sequence="XY4")   # optional
        # ZNE (multi-task):
        for scale, circ in em.zne_circuit_set(qc_ddd, [1, 1.5, 2, 2.5, 3]):
            submit(circ)
        # REM (multi-task, post-processing):
        for label, circ in em.rem_calibration_circuits(qc_ddd):
            submit(circ)

    Order matters: DDD before folding, because folding after DDD would
    duplicate the DD pulses (same reasoning as the Rigetti module -- there
    just is no placement stage to put before it here).
    """

    # ---- DD sequences, in IonQ-basis-compatible native gates -----------------
    # X = rx(pi); Y = rz(-pi/2) rx(pi) rz(pi/2); I = identity (null control).
    # Reused byte-for-byte from error_mitig.py -- the algebra does not depend
    # on which device rz/rx run on. Whether the SEQUENCE survives IonQ's
    # server-side re-synthesis into gpi/gpi2 unchanged is a separate,
    # UNVERIFIED question (module docstring above, and brief Section 9.2).
    _X = (("rx", math.pi),)
    _Y = (("rz", -math.pi / 2), ("rx", math.pi), ("rz", math.pi / 2))
    _I = (("id", None),)

    DDD_SEQUENCES = {
        "XX":     (_X, _X),
        "YY":     (_Y, _Y),
        "XY4":    (_X, _Y, _X, _Y),
        "XYXYX":  (_X, _Y, _X, _Y, _X),
        "II":     (_I, _I),          # null control -- see apply_ddd docstring
    }

    # Sequences whose pulses do NOT compose to the identity -- see
    # verify_sequences(). Carried over unchanged: XYXYX -> iX regardless of
    # which device executes the rz/rx pulses.
    NON_IDENTITY_SEQUENCES = ("XYXYX",)

    def __init__(self, one_qubit_time_s: float = ONE_QUBIT_TIME_S,
                 two_qubit_time_s: float = TWO_QUBIT_TIME_S,
                 readout_time_s: float = READOUT_TIME_S,
                 rz_time_s: float = RZ_TIME_S,
                 basis: list[str] | None = None):
        self.basis = list(basis or IONQ_BASIS)
        self.t_1q   = one_qubit_time_s
        self.t_2q   = two_qubit_time_s
        self.t_meas = readout_time_s
        self.t_rz   = rz_time_s

    # =====================================================================
    # Provenance -- same pattern as error_mitig.py: one dict, inherited
    # stage to stage, so a task log written after fold() still has the DDD
    # record even though fold() rebuilds the circuit from scratch.
    # =====================================================================

    @staticmethod
    def _get_meta(qc: QuantumCircuit | None) -> dict:
        return dict(getattr(qc, "_em", {}) or {}) if qc is not None else {}

    @classmethod
    def _set_meta(cls, qc: QuantumCircuit, key: str, value: dict,
                  inherit_from: QuantumCircuit | None = None) -> QuantumCircuit:
        meta = cls._get_meta(inherit_from if inherit_from is not None else qc)
        meta[key] = value
        qc._em = meta
        return qc

    @classmethod
    def provenance(cls, qc: QuantumCircuit) -> dict:
        """Everything every transform did to this circuit (DDD, ZNE)."""
        return cls._get_meta(qc)

    # =====================================================================
    # Timing model
    # =====================================================================

    def op_duration(self, name: str, n_qubits: int) -> float:
        """
        Duration of one native operation, in seconds.

        rz is charged self.t_rz (default: the full ONE_QUBIT_TIME_S -- see
        the RZ_TIME_S module comment for why this is a NAMED, OPEN choice
        rather than an assumed-free virtual-Z, unlike Rigetti's op_duration).
        """
        if name in ("barrier", "delay", "id"):
            return 0.0
        if name == "rz":
            return self.t_rz
        if name == "measure":
            return self.t_meas
        return self.t_2q if n_qubits >= 2 else self.t_1q

    def _schedule(self, qc: QuantumCircuit):
        """
        Layer the circuit and give each layer a real duration.

        Layer duration = the longest operation in it (operations in one
        layer act on disjoint qubits). Whether this is the right model on a
        trapped-ion device where gates are typically serialized rather than
        run in same-cycle layers is itself part of the open timing-model
        question (brief Section 9.2) -- kept structurally identical to
        error_mitig.py's _schedule() so idle-window detection stays
        comparable across the two campaigns, but treat idle windows found
        this way as a MODEL, not a measurement.
        """
        dag    = circuit_to_dag(qc)
        layers = list(dag.layers())
        n      = qc.num_qubits

        layer_ops: list[list] = []
        layer_dur: list[float] = []
        busy = {q: [] for q in range(n)}

        for layer in layers:
            ops, touched, dur = [], set(), 0.0
            for node in layer["graph"].op_nodes():
                idx = [qc.find_bit(q).index for q in node.qargs]
                ops.append((node.op, node.qargs, node.cargs))
                touched.update(idx)
                dur = max(dur, self.op_duration(node.op.name, len(node.qargs)))
            layer_ops.append(ops)
            layer_dur.append(dur)
            for q in range(n):
                busy[q].append(q in touched)
        return layer_ops, layer_dur, busy

    def idle_profile(self, qc: QuantumCircuit) -> dict[int, float]:
        """Total idle time per qubit, in seconds, between its first and last op."""
        layer_ops, layer_dur, busy = self._schedule(qc)
        out: dict[int, float] = {}
        for q, mask in busy.items():
            active = [i for i, b in enumerate(mask) if b]
            if len(active) < 2:
                out[q] = 0.0
                continue
            first, last = active[0], active[-1]
            out[q] = sum(layer_dur[i] for i in range(first, last) if not mask[i])
        return out

    # =====================================================================
    # DDD -- circuit in, circuit out, ZERO extra tasks
    # =====================================================================

    def apply_ddd(self, qc: QuantumCircuit, sequence: str = "XY4",
                  min_idle_s: float | None = None,
                  qubits: list[int] | None = None,
                  allow_non_identity: bool = False,
                  verbose: bool = False) -> QuantumCircuit:
        """
        Digital dynamical decoupling. Idle windows found from REAL DURATIONS
        (see _schedule / op_duration), not from moment counts.

        This is the FIRST time DDD has been run on IonQ anywhere in this
        repository -- see MITIGATION_BRIEF_IONQ.md Section 9.2. Treat "does
        DDD help on IonQ" as a genuinely open experimental question with a
        real chance of a null or negative answer; do not port the Rigetti
        XY4 result (+0.0237 P_correct) into any IonQ text or default as if
        it were a prediction (CLAUDE.md Section 2: negative results are
        deliverables).

        sequence:
          "XX", "YY", "XY4", "XYXYX"  -- real decoupling sequences
          "II"                        -- NULL CONTROL, mandatory (not
             optional) per the brief: on IonQ the II vs XY4 comparison is
             the only way to tell "the pulses helped" apart from "the
             compiler cancelled DDD too" and "the pulses just added time".
             Note the control's own dependency: a compiler is entitled to
             delete identity gates, and whether IonQ preserves them at all
             is unverified (Section 9.1).

        min_idle_s defaults to one 2q-gate time (self.t_2q), same convention
        as error_mitig.py -- but see the RZ_TIME_S discussion above for why
        that default itself is not settled on this device.
        """
        if sequence not in self.DDD_SEQUENCES:
            raise ValueError(f"unknown sequence {sequence!r}; "
                             f"choose from {sorted(self.DDD_SEQUENCES)}")
        if sequence in self.NON_IDENTITY_SEQUENCES and not allow_non_identity:
            raise ValueError(
                f"DD sequence {sequence!r} does NOT compose to the identity "
                f"(verify_sequences() gives iX, a net bit flip). Inserting it "
                f"into an idle window rotates the qubit's state instead of "
                f"refocusing it. Use 'XY4'. Pass allow_non_identity=True only "
                f"if you have separately confirmed you want this."
            )
        pulses = self.DDD_SEQUENCES[sequence]
        if min_idle_s is None:
            min_idle_s = self.t_2q

        layer_ops, layer_dur, busy = self._schedule(qc)
        n_layers = len(layer_ops)

        plan: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        n_windows = n_pulses = 0
        crowded = 0

        targets = set(qubits) if qubits is not None else None
        for q, mask in busy.items():
            if targets is not None and q not in targets:
                continue
            active = [i for i, b in enumerate(mask) if b]
            if len(active) < 2:
                continue
            first, last = active[0], active[-1]

            for start, stop in self._idle_runs(mask, first, last):
                window = sum(layer_dur[start:stop])
                if window < min_idle_s:
                    continue
                n_windows += 1
                placed = self._place_pulses(pulses, layer_dur, start, stop, window)
                if len(set(placed.keys())) < len(pulses):
                    crowded += 1
                for layer_i, pulse_list in placed.items():
                    plan[layer_i][q].extend(pulse_list)
                    n_pulses += len(pulse_list)

        out = QuantumCircuit(*qc.qregs, *qc.cregs)
        out.global_phase = qc.global_phase
        for i in range(n_layers):
            for q, pulse_list in plan.get(i, {}).items():
                for gate in pulse_list:
                    self._emit(out, gate, q)
            for op, qargs, cargs in layer_ops[i]:
                out.append(op, qargs, cargs)

        self._set_meta(out, "ddd", {
            "sequence":     sequence,
            "n_windows":    n_windows,
            "n_pulses":     n_pulses,
            "n_gates_added": out.size() - qc.size(),
            "min_idle_s":   min_idle_s,
            "rz_time_s":    self.t_rz,
            "crowded_windows": crowded,
        }, inherit_from=qc)
        if verbose:
            print(f"  --- DDD ({sequence}) ---")
            print(f"    idle windows filled : {n_windows}")
            print(f"    pulses inserted     : {n_pulses}")
            print(f"    native gates added  : {out.size() - qc.size()}")
            print(f"    rz timing used      : {self.t_rz*1e6:.1f} us  "
                  f"({'== ONE_QUBIT_TIME_S, i.e. charged' if abs(self.t_rz - self.t_1q) < 1e-15 else 'custom'})")
            if crowded:
                print(f"    WARNING: {crowded} window(s) had fewer layers than "
                      f"pulses -- some pulses share a layer with no spacing.")
        return out

    @staticmethod
    def _idle_runs(mask: list[bool], first: int, last: int):
        """Maximal idle runs strictly between the qubit's first and last op."""
        runs, i = [], first + 1
        while i < last:
            if not mask[i]:
                j = i
                while j < last and not mask[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        return runs

    @staticmethod
    def _place_pulses(pulses, layer_dur, start, stop, window):
        """CPMG-style: pulse p at (p + 0.5)/P of the way through the window."""
        out = defaultdict(list)
        p_count = len(pulses)
        cum, bounds = 0.0, []
        for i in range(start, stop):
            bounds.append((i, cum))
            cum += layer_dur[i]
        for p, pulse in enumerate(pulses):
            t = window * (p + 0.5) / p_count
            layer_i = bounds[0][0]
            for i, c in bounds:
                if c <= t:
                    layer_i = i
                else:
                    break
            out[layer_i].append(pulse)
        return out

    @staticmethod
    def _emit(circ: QuantumCircuit, pulse, qubit: int) -> None:
        for name, param in pulse:
            if name == "rx":
                circ.rx(param, qubit)
            elif name == "rz":
                circ.rz(param, qubit)
            elif name == "id":
                circ.id(qubit)
            else:
                raise ValueError(f"unsupported DD pulse gate {name!r}")

    def ddd_report(self, qc: QuantumCircuit) -> dict | None:
        return self._get_meta(qc).get("ddd")

    @classmethod
    def verify_sequences(cls, tol: float = 1e-9) -> dict[str, dict]:
        """Check each DD sequence composes to the identity up to a global phase."""
        out = {}
        for name, pulses in cls.DDD_SEQUENCES.items():
            qc = QuantumCircuit(1)
            for pulse in pulses:
                cls._emit(qc, pulse, 0)
            M = Operator(qc).data
            c = M[0, 0]
            is_id = (abs(c) > tol
                     and np.allclose(M, c * np.eye(2), atol=1e-8))
            out[name] = {
                "identity_up_to_phase": bool(is_id),
                "matrix": M,
                "n_pulses": len(pulses),
            }
        return out

    # =====================================================================
    # ZNE -- circuit in, circuit out, but ONE CIRCUIT PER LAMBDA
    # =====================================================================

    def fold(self, qc: QuantumCircuit, scale: float,
             method: str = "random", seed: int = 0,
             two_qubit_only: bool = False) -> QuantumCircuit:
        """
        Noise-scale a circuit by unitary folding. Identical algorithm to
        error_mitig.py's fold() -- the folding math has nothing device-
        specific in it. See that module's docstring for the full discussion
        of "global" vs "random" vs "all" and why stratified 1q/2q pool
        selection matters (measured on Rigetti: uniform selection scaled 2q
        gates by 1.839x when lambda=1.5 was requested; the same bias
        mechanism applies to any circuit with an uneven 1q:2q gate ratio,
        which QPE has here too).

        Whether a folded circuit's C C^dag C survives IonQ's server-side
        compilation is UNCONFIRMED (brief Section 9.1) -- verify_zne() below
        is the client-side half of checking that; there is currently no
        way to check the device-compiled program for IonQ (task metadata
        contains only submitted-QASM and sharpenedProbabilities, per the
        brief), so this check is necessarily incomplete on this device.
        """
        if scale < 1:
            raise ValueError(f"ZNE scale must be >= 1; got {scale}")
        if method == "global":
            if scale != int(scale) or int(scale) % 2 == 0:
                raise ValueError("method='global' supports only ODD INTEGER "
                                 f"lambda; got {scale}. Use method='random' "
                                 "for fractional scaling.")
            return self._fold_global(qc, int(scale))
        if method not in ("random", "all"):
            raise ValueError(f"unknown fold method {method!r}")

        base = qc.remove_final_measurements(inplace=False)
        foldable = [i for i, inst in enumerate(base.data)
                    if inst.operation.name not in ("barrier", "delay", "id")
                    and (not two_qubit_only or len(inst.qubits) == 2)]
        if not foldable:
            return qc.copy()

        pools: dict[int, list[int]] = defaultdict(list)
        for i in foldable:
            pools[len(base.data[i].qubits)].append(i)

        rng = np.random.default_rng(seed)
        fold_count: dict[int, int] = {}
        for _, pool in sorted(pools.items()):
            G = len(pool)
            total_folds = int(round(G * (scale - 1) / 2))
            base_folds, extra = divmod(total_folds, G)
            if method == "random":
                chosen = set(rng.choice(G, size=extra, replace=False).tolist()) if extra else set()
            else:
                chosen = set(range(extra))
            for k, idx in enumerate(pool):
                fold_count[idx] = base_folds + (1 if k in chosen else 0)

        out = QuantumCircuit(*base.qregs, *base.cregs)
        out.global_phase = base.global_phase
        for i, inst in enumerate(base.data):
            out.append(inst.operation, inst.qubits, inst.clbits)
            for _ in range(fold_count.get(i, 0)):
                out.append(inst.operation.inverse(), inst.qubits, inst.clbits)
                out.append(inst.operation,           inst.qubits, inst.clbits)

        folded = self._reattach_measurements(qc, out)
        self._set_meta(folded, "zne",
                       {"scale": scale, "method": method,
                        "two_qubit_only": two_qubit_only, "seed": seed},
                       inherit_from=qc)
        return folded

    def _fold_global(self, qc: QuantumCircuit, scale: int) -> QuantumCircuit:
        base = qc.remove_final_measurements(inplace=False)
        folded = base.copy()
        for _ in range((scale - 1) // 2):
            folded.compose(base.inverse(), inplace=True)
            folded.compose(base,           inplace=True)
        out = self._reattach_measurements(qc, folded)
        self._set_meta(out, "zne",
                       {"scale": scale, "method": "global",
                        "two_qubit_only": False, "seed": None},
                       inherit_from=qc)
        return out

    @staticmethod
    def _reattach_measurements(original: QuantumCircuit,
                               body: QuantumCircuit) -> QuantumCircuit:
        """Put the original terminal measurements back on the folded body."""
        out = QuantumCircuit(*original.qregs, *original.cregs)
        out.global_phase = body.global_phase
        out.compose(body, qubits=list(range(body.num_qubits)), inplace=True)
        for cbit, qbit in enumerate(measured_physical_qubits(original)):
            out.measure(qbit, cbit)
        return out

    def zne_circuit_set(self, qc: QuantumCircuit,
                        scales=(1, 1.5, 2, 2.5, 3),
                        method: str = "random", seed: int = 0,
                        two_qubit_only: bool = False) -> list[dict]:
        """
        Build the FULL set of circuits one ZNE estimate needs. MULTI-TASK --
        see the class docstring and error_mitig.py's zne_circuit_set for the
        contract-boundary warning; identical here.
        """
        out = []
        base_2q = None
        for s in scales:
            circ = self.fold(qc, s, method=method, seed=seed,
                             two_qubit_only=two_qubit_only)
            prof = count_native(circ)
            if base_2q is None:
                base_2q = prof["n_2q"] or 1
            out.append({
                "scale":             s,
                "circuit":           circ,
                "profile":           prof,
                "achieved_scale_2q": prof["n_2q"] / base_2q,
                "achieved_scale_all": (prof["n_1q"] + prof["n_2q"])
                                      / max(out[0]["profile"]["n_1q"]
                                            + out[0]["profile"]["n_2q"], 1)
                                      if out else 1.0,
            })
        return out

    # =====================================================================
    # REM -- NOT a circuit transform
    # =====================================================================

    def rem_calibration_circuits(self, qc: QuantumCircuit,
                                 model: str = "tensored"
                                 ) -> list[tuple[str, QuantumCircuit]]:
        """
        Readout-error-mitigation CALIBRATION circuits. Adds NO gates to the
        submitted circuit -- REM is unaffected by the Section 9.1 compiler
        question, because there is nothing to cancel (REM adds no gates at
        all, on either device).

        model: "tensored" (2 circuits, default) or "full" (2^n, crosstalk-
        aware, refuses beyond n=8). See error_mitig.py's identically-named
        method for the full discussion of why "simplex" inversion (not raw
        pseudo-inverse) belongs in the analysis step.
        """
        phys = measured_physical_qubits(qc)
        n    = len(phys)
        nq   = qc.num_qubits

        if model == "tensored":
            states = [("cal0", [0] * n), ("cal1", [1] * n)]
        elif model == "full":
            if n > 8:
                raise ValueError(f"full readout calibration needs 2^{n} = "
                                 f"{2**n} circuits; use model='tensored'")
            if n > 5:
                print(f"  NOTE: full calibration on {n} qubits = {2**n} "
                      f"circuits. That is a lot of tasks.")
            states = [(f"cal{b:0{n}b}",
                       [(b >> (n - 1 - k)) & 1 for k in range(n)])
                      for b in range(2 ** n)]
        else:
            raise ValueError(f"unknown REM model {model!r}")

        out = []
        for label, bits in states:
            circ = QuantumCircuit(nq, n)
            for k, (q, bit) in enumerate(zip(phys, bits)):
                if bit:
                    circ.rx(math.pi, q)          # native X
                circ.measure(q, k)
            self._set_meta(circ, "rem", {"model": model, "label": label,
                                         "prepared_bits": bits,
                                         "physical_qubits": phys},
                           inherit_from=qc)
            out.append((label, circ))
        return out

    # =====================================================================
    # Verification
    # =====================================================================

    def verify_ddd(self, before: QuantumCircuit, after: QuantumCircuit) -> dict:
        """
        Client-side check that DDD actually added the gates it claims.

        UNLIKE error_mitig.py's verify_ddd, there is currently no known
        hardware-side counterpart to pair this with: the brief's audit of
        every saved IonQ result found additional_metadata.ionqMetadata
        contains only sharpenedProbabilities, and action.source is the
        SUBMITTED QASM, not a device-compiled program (Section 9.1). So this
        check can confirm the pulses were IN THE SUBMITTED PROGRAM; it
        cannot confirm they survived to execution. Do not read `ok: True`
        here as "DDD ran on hardware" -- only as "DDD is in the QASM we sent".
        """
        pb, pa = count_native(before), count_native(after)
        rep = self.ddd_report(after) or {}
        seq = rep.get("sequence")

        added = after.size() - before.size()
        two_q_unchanged = pb["n_2q"] == pa["n_2q"]

        return {
            "sequence":      seq,
            "n_1q_before":   pb["n_1q"], "n_1q_after": pa["n_1q"],
            "n_2q_before":   pb["n_2q"], "n_2q_after": pa["n_2q"],
            "instructions_added": added,
            "native_gates_added": pa["n_gates"] - pb["n_gates"],
            "pulses_planned": rep.get("n_pulses"),
            "two_qubit_unchanged": two_q_unchanged,
            "ok": added > 0 and two_q_unchanged,
            "note": ("client-side only -- no compiled-program check is "
                     "possible for IonQ with this repo's current metadata "
                     "retrieval (Section 9.1)"),
        }

    def verify_zne(self, circuit_set: list[dict], tol: float = 0.15) -> dict:
        """
        Check the folded circuits' 2q counts actually track lambda -- SAME
        caveat as verify_ddd: this is client-side only and cannot see what
        the device actually executed.
        """
        rows, ok = [], True
        for e in circuit_set:
            err = abs(e["achieved_scale_2q"] - e["scale"]) / max(e["scale"], 1)
            rows.append({"scale": e["scale"],
                         "achieved_scale_2q": e["achieved_scale_2q"],
                         "n_2q": e["profile"]["n_2q"],
                         "rel_error": err,
                         "ok": err <= tol})
            ok = ok and err <= tol
        return {"ok": ok, "rows": rows,
                "note": "client-side only; no compiled-program check is "
                        "possible for IonQ with this repo's current metadata "
                        "retrieval (Section 9.1). The real check is whether "
                        "the primary observable decays monotonically in "
                        "lambda toward the incoherent floor 2^-n_prec -- see "
                        "MITIGATION_BRIEF_IONQ.md Section 9.1, 'Expected "
                        "forms, so the flatness is readable'."}
