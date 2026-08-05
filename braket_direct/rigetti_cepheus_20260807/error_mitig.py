#!/usr/bin/env python3
"""
error_mitig.py
==============
Error-mitigation transforms for circuits headed to Rigetti Cepheus-1-108Q.

One class, `ErrorMitigation`. The contract you asked for -- circuit in,
circuit out -- holds for everything in the "circuit transforms" section
below. Two of the three techniques in the spec do NOT fit that contract, and
rather than pretend otherwise they are separated out and flagged loudly:

    technique   fits circuit -> circuit?   extra tasks
    ---------   -----------------------   -----------
    placement   YES                        0    always-on substrate
    DDD         YES                        0    pure circuit transform
    ZNE         one circuit PER lambda      N-1  needs N circuits + a fit
    REM         NO                          2+   calibration + post-processing

  - ZNE:  `fold()` really is circuit -> circuit, but a single folded circuit
          is not a mitigated result. You need one task per lambda and then an
          extrapolation over their P_correct values. The circuit-building half
          lives here; the estimator half does not.
  - REM:  adds NO gates to your circuit at all. It is a classical
          post-processing step on the counts, driven by separate calibration
          circuits. `rem_calibration_circuits()` gives you those circuits;
          the inversion belongs in the analysis code.

See the module-level notes at the bottom of this docstring for what that
means for the run plan.


Dependencies
------------
Implemented directly against qiskit. Mitiq is NOT used: it is not installed
in this environment, it is cirq-native (so every circuit would round-trip
through a converter), and the spec itself points out that Mitiq's DDD
slack detection is moment-based rather than time-based -- which is the part
we most need to do differently. The sequences and folding formulas here
match Mitiq's definitions, so swapping it back in later is a local change.


Rigetti specifics baked in
--------------------------
  - Native basis is rz / rx / cz. Sequences are built from those directly
    rather than from H/X/Y placeholders, so nothing needs a second transpile
    pass afterwards -- and a second pass is exactly what would cancel the DD
    pulses and the ZNE folds.
  - RZ is a virtual frame change on superconducting hardware: ZERO duration.
    The timing model reflects that, which matters a lot for idle-window
    detection (a run of rz gates is not a busy window).
  - The compiler will undo both DDD and ZNE if it is allowed to. Everything
    here is meant to be submitted inside a verbatim box
    (`to_braket_qasm(..., verbatim=True)`), and verified afterwards against
    the returned compiled program.
"""

import math
from collections import defaultdict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from rigetti_qpuf_common import (
    RIGETTI_BASIS, ONE_QUBIT_TIME_S, TWO_QUBIT_TIME_S, READOUT_TIME_S,
    F1Q, F2Q, load_device_caps, coupling_map_from_caps, count_native,
    measured_physical_qubits,
)

CHIPLET_SIZE = 9          # Cepheus-1-108Q: twelve 9-qubit chiplets, 3x4 array
N_CHIPLETS   = 12

# Fidelity assumed for an intermodule (chiplet-to-chiplet) coupler when the
# live calibration does not give us one. Intermodule couplers are the weakest
# links on this architecture; this default makes the placement scorer avoid
# them without needing calibration data to prove it.
F2Q_INTERMODULE = 0.90


class ErrorMitigation:
    """
    Circuit-level error mitigation for Rigetti Cepheus.

    Typical use:

        em = ErrorMitigation()
        qc_placed = em.select_best_qubits(qc, hub_qubits=targ_idx)  # ALWAYS
        qc_final  = em.apply_ddd(qc_placed, sequence="XY4")         # optional
        # ZNE (multi-task):
        for scale, circ in em.zne_circuit_set(qc_final, [1, 1.5, 2, 2.5, 3]):
            submit(circ)
        # REM (multi-task, post-processing):
        for label, circ in em.rem_calibration_circuits(qc_final):
            submit(circ)

    Order matters and is fixed: placement first (it changes the routed gate
    count, hence the idle structure), then DDD (it fills idle windows), then
    folding (it multiplies whatever is there). Folding before DDD would
    duplicate the DD pulses; DDD before placement would have its windows
    rewritten by the router.
    """

    # ---- DD sequences, in Rigetti native gates -------------------------------
    # X   = rx(pi)
    # Y   = rz(-pi/2) rx(pi) rz(pi/2)   (verified below; rz is free, so a Y
    #       pulse costs exactly as much time as an X pulse)
    # I   = identity, for the null control
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

    def __init__(self, caps: dict | None = None,
                 one_qubit_time_s: float = ONE_QUBIT_TIME_S,
                 two_qubit_time_s: float = TWO_QUBIT_TIME_S,
                 readout_time_s: float = READOUT_TIME_S,
                 basis: list[str] | None = None):
        if caps is None:
            caps, self.caps_are_real = load_device_caps()
        else:
            self.caps_are_real = True
        self.caps  = caps
        self.basis = list(basis or RIGETTI_BASIS)

        self.t_1q     = one_qubit_time_s
        self.t_2q     = two_qubit_time_s
        self.t_meas   = readout_time_s

        self._coupling  = coupling_map_from_caps(caps)
        self._neighbors = self._build_neighbors(caps)
        self._chiplets  = None
        self._route_cache: dict = {}

    # =====================================================================
    # Timing model
    # =====================================================================

    def op_duration(self, name: str, n_qubits: int) -> float:
        """
        Duration of one native operation, in seconds.

        RZ is 0: on superconducting hardware a Z rotation is a frame update
        applied in software to every later pulse, not a physical pulse. Any
        idle-window analysis that charges rz real time will systematically
        under-detect idle windows.
        """
        if name in ("barrier", "delay", "id"):
            return 0.0
        if name == "rz":
            return 0.0
        if name == "measure":
            return self.t_meas
        return self.t_2q if n_qubits >= 2 else self.t_1q

    # =====================================================================
    # Chiplet topology
    # =====================================================================

    @staticmethod
    def _build_neighbors(caps: dict) -> dict[int, set[int]]:
        nbrs: dict[int, set[int]] = defaultdict(set)
        for a, vs in caps["connectivity"].items():
            for b in vs:
                nbrs[int(a)].add(int(b))
                nbrs[int(b)].add(int(a))
        return dict(nbrs)

    def chiplets(self) -> dict[int, list[int]]:
        """
        Partition the device into its 9-qubit chiplets.

        Rigetti numbers Cepheus qubits so that chiplet c owns indices
        [9c, 9c+9). We do not take that on faith: the grouping is accepted
        only if every group induces a CONNECTED subgraph. If it does not
        (different numbering convention, or a placeholder topology), we fall
        back to greedy BFS growth into connected blocks of 9, which gives a
        usable partition even if it is not the vendor's.
        """
        if self._chiplets is not None:
            return self._chiplets

        n = self.caps["qubit_count"]
        by_index = {c: list(range(c * CHIPLET_SIZE,
                                  min((c + 1) * CHIPLET_SIZE, n)))
                    for c in range((n + CHIPLET_SIZE - 1) // CHIPLET_SIZE)}

        if all(self._is_connected(g) for g in by_index.values()):
            self._chiplets = by_index
            return self._chiplets

        # Fallback: grow connected blocks of CHIPLET_SIZE by BFS.
        unassigned = set(range(n))
        groups, cid = {}, 0
        while unassigned:
            seed = min(unassigned, key=lambda q: len(self._neighbors.get(q, ())))
            block, frontier = [], [seed]
            while frontier and len(block) < CHIPLET_SIZE:
                q = frontier.pop(0)
                if q not in unassigned:
                    continue
                unassigned.discard(q)
                block.append(q)
                frontier += sorted(self._neighbors.get(q, ()) & unassigned)
            groups[cid] = sorted(block)
            cid += 1
        self._chiplets = groups
        return self._chiplets

    def _is_connected(self, qubits: list[int]) -> bool:
        if not qubits:
            return False
        want, seen, stack = set(qubits), set(), [qubits[0]]
        while stack:
            q = stack.pop()
            if q in seen:
                continue
            seen.add(q)
            stack += [b for b in self._neighbors.get(q, ()) if b in want and b not in seen]
        return seen == want

    def chiplet_of(self, qubit: int) -> int | None:
        for cid, qs in self.chiplets().items():
            if qubit in qs:
                return cid
        return None

    def is_intermodule(self, a: int, b: int) -> bool:
        """True if edge (a,b) crosses a chiplet boundary."""
        ca, cb = self.chiplet_of(a), self.chiplet_of(b)
        return ca is not None and cb is not None and ca != cb

    # =====================================================================
    # Calibration lookups
    # =====================================================================

    def edge_fidelity(self, a: int, b: int) -> float:
        """
        Two-qubit gate fidelity for edge (a,b) from the live calibration, or
        a default. Intermodule couplers get a deliberately pessimistic
        default so the placement scorer avoids them even with no calibration
        data loaded.
        """
        specs = self.caps.get("specs", {}).get("2Q", {})
        for key in (f"{a}-{b}", f"{b}-{a}", f"{min(a,b)}-{max(a,b)}"):
            entry = specs.get(key)
            if isinstance(entry, dict):
                for field in ("fCZ", "fISWAP", "fCPHASE", "fXY"):
                    if isinstance(entry.get(field), (int, float)):
                        return float(entry[field])
        return F2Q_INTERMODULE if self.is_intermodule(a, b) else F2Q

    def qubit_fidelity(self, q: int) -> float:
        specs = self.caps.get("specs", {}).get("1Q", {})
        entry = specs.get(str(q))
        if isinstance(entry, dict) and isinstance(entry.get("f1QRB"), (int, float)):
            return float(entry["f1QRB"])
        return F1Q

    def readout_fidelity(self, q: int) -> float:
        specs = self.caps.get("specs", {}).get("1Q", {})
        entry = specs.get(str(q))
        if isinstance(entry, dict):
            for field in ("fRO", "fActiveReset", "f1QRB_std_err"):
                if field == "fRO" and isinstance(entry.get(field), (int, float)):
                    return float(entry[field])
        return 0.95

    # =====================================================================
    # CIRCUIT TRANSFORMS -- circuit in, circuit out, ZERO extra tasks
    # =====================================================================

    def select_best_qubits(self, qc: QuantumCircuit,
                           hub_qubits: list[int] | None = None,
                           optimization_level: int = 2,
                           seed_transpiler: int = 42,
                           allow_intermodule: bool | None = None,
                           sites_per_chiplet: int = 3,
                           verbose: bool = False) -> QuantumCircuit:
        """
        Chiplet-local placement. ALWAYS ON -- this is the substrate every
        other technique sits on, not a variable under test.

        Returns a routed circuit on physical qubits, ready for verbatim
        submission. `placement_report()` describes what it chose.

        Why the naive approach is wrong
        -------------------------------
        Picking the connected subgraph with the best average CZ fidelity
        fails here, because QPE's interaction graph is a STAR, not a chain:
        every controlled-U^(2^k) touches the target register, so the target
        is a hub that all n_prec precision qubits must reach. A 9-qubit
        chiplet has max degree ~4, so beyond n_prec = 4 you need SWAPs even
        inside one chiplet, and the SWAP count depends strongly on where the
        hub sits.

        So we (a) anchor the hub at high-degree sites, (b) score by running
        the ACTUAL router on each candidate and evaluating the compiled
        circuit -- a placement with slightly worse edges but two fewer SWAPs
        usually wins -- and (c) reject placements that spill across a chiplet
        boundary unless the register cannot fit.

        hub_qubits: virtual indices of the hub register (for our builders,
        the target qubits are [0, n_targ)). Inferred from the interaction
        graph if omitted.
        """
        n = qc.num_qubits
        if hub_qubits is None:
            hub_qubits = self._infer_hub(qc)

        must_spill = n > CHIPLET_SIZE
        if allow_intermodule is None:
            allow_intermodule = must_spill
        if must_spill and verbose:
            print(f"  placement: {n} qubits > {CHIPLET_SIZE}-qubit chiplet -- "
                  f"spilling across a boundary is unavoidable; minimising it.")

        candidates = self._candidate_layouts(n, hub_qubits, sites_per_chiplet)
        if not candidates:
            raise RuntimeError("no candidate placement found for "
                               f"{n} qubits on {self.caps['device_name']}")

        best = None
        for layout in candidates:
            routed = self._route(qc, layout, optimization_level, seed_transpiler)
            score  = self._score(routed)
            if score["n_intermodule"] > 0 and not allow_intermodule:
                continue
            key = (score["n_intermodule"], -score["log_fidelity"])
            if best is None or key < best[0]:
                best = (key, routed, score, layout)

        if best is None:
            # Every chiplet-local candidate spilled. Fall back to allowing it,
            # but report the count rather than hiding it.
            for layout in candidates:
                routed = self._route(qc, layout, optimization_level, seed_transpiler)
                score  = self._score(routed)
                key = (score["n_intermodule"], -score["log_fidelity"])
                if best is None or key < best[0]:
                    best = (key, routed, score, layout)

        _, routed, score, layout = best
        routed._em_placement = {          # consumed by placement_report()
            "initial_layout":  layout,
            "hub_qubits":      hub_qubits,
            "chiplet_ids":     sorted({self.chiplet_of(q) for q in layout}),
            "n_intermodule":   score["n_intermodule"],
            "n_2q":            score["n_2q"],
            "n_swap_pairs":    score["n_swap_pairs"],
            "log_fidelity":    score["log_fidelity"],
            "est_fidelity":    math.exp(score["log_fidelity"]),
            "candidates_tried": len(candidates),
            "calibration":     "live" if self.caps.get("specs") else "defaults",
        }
        if verbose:
            self.print_placement_report(routed)
        return routed

    def _infer_hub(self, qc: QuantumCircuit) -> list[int]:
        """Highest-degree qubit(s) in the circuit's interaction graph."""
        deg: dict[int, int] = defaultdict(int)
        for inst in qc.data:
            if len(inst.qubits) == 2:
                for q in inst.qubits:
                    deg[qc.find_bit(q).index] += 1
        if not deg:
            return [0]
        top = max(deg.values())
        return [q for q, d in deg.items() if d == top]

    def _candidate_layouts(self, n: int, hub_qubits: list[int],
                           sites_per_chiplet: int) -> list[list[int]]:
        """
        For each chiplet, anchor the hub at its highest-degree sites and grow
        the rest of the register outward by BFS. Returns initial_layout lists
        (virtual index -> physical qubit).
        """
        layouts, seen = [], set()
        for _, qs in sorted(self.chiplets().items()):
            ranked = sorted(qs, key=lambda q: -len(self._neighbors.get(q, ())))
            for anchor in ranked[:sites_per_chiplet]:
                phys = self._bfs_order(anchor, n, prefer=set(qs))
                if len(phys) < n:
                    continue
                layout = [-1] * n
                # Hub goes on the anchor and its nearest neighbours.
                rest = [p for p in phys if p not in phys[:len(hub_qubits)]]
                for i, v in enumerate(hub_qubits):
                    layout[v] = phys[i]
                others = [v for v in range(n) if v not in hub_qubits]
                for v, p in zip(others, rest):
                    layout[v] = p
                if -1 in layout:
                    continue
                key = tuple(layout)
                if key not in seen:
                    seen.add(key)
                    layouts.append(layout)
        return layouts

    def _bfs_order(self, start: int, n: int, prefer: set[int]) -> list[int]:
        """
        BFS from `start`, taking qubits inside `prefer` (the chiplet) before
        stepping outside it. That ordering is what makes the fallback rule
        for n > 9 land the way the spec wants: the hub and its high-traffic
        neighbours stay inside one chiplet, and only the last-added qubits
        cross the boundary.
        """
        out, seen, frontier = [], {start}, [start]
        while frontier and len(out) < n:
            q = frontier.pop(0)
            out.append(q)
            nbrs = sorted(self._neighbors.get(q, ()) - seen,
                          key=lambda x: (x not in prefer, -len(self._neighbors.get(x, ()))))
            for b in nbrs:
                seen.add(b)
                frontier.append(b)
        return out[:n]

    def _route(self, qc: QuantumCircuit, layout: list[int],
               optimization_level: int, seed: int) -> QuantumCircuit:
        key = (id(qc), tuple(layout), optimization_level, seed)
        if key not in self._route_cache:
            self._route_cache[key] = transpile(
                qc,
                basis_gates=self.basis + ["measure"],
                coupling_map=self._coupling,
                initial_layout=layout,
                optimization_level=optimization_level,
                seed_transpiler=seed,
            )
        return self._route_cache[key]

    def _score(self, routed: QuantumCircuit) -> dict:
        """
        Score a routed circuit by the product of the fidelities it actually
        uses -- every routed 2q gate (SWAPs included, since the router has
        already decomposed them into CZs) plus the readout of every measured
        qubit. Returned as a log so long circuits do not underflow.
        """
        log_f, n_2q, n_inter = 0.0, 0, 0
        edge_use: dict[tuple[int, int], int] = defaultdict(int)
        for inst in routed.data:
            if inst.operation.name in ("barrier", "delay"):
                continue
            idx = [routed.find_bit(q).index for q in inst.qubits]
            if len(idx) == 2:
                a, b = idx
                n_2q += 1
                edge_use[(min(a, b), max(a, b))] += 1
                if self.is_intermodule(a, b):
                    n_inter += 1
                log_f += math.log(max(self.edge_fidelity(a, b), 1e-6))
            elif inst.operation.name not in ("measure", "id", "rz"):
                log_f += math.log(max(self.qubit_fidelity(idx[0]), 1e-6))
        for q in measured_physical_qubits(routed):
            log_f += math.log(max(self.readout_fidelity(q), 1e-6))

        # A routed SWAP shows up as 3 CZs on one edge; this is a proxy for
        # how much routing overhead the placement incurred.
        n_swap_pairs = sum(v // 3 for v in edge_use.values())
        return {"log_fidelity": log_f, "n_2q": n_2q,
                "n_intermodule": n_inter, "n_swap_pairs": n_swap_pairs}

    def placement_report(self, qc: QuantumCircuit) -> dict | None:
        """Placement metadata attached by select_best_qubits (log this)."""
        return getattr(qc, "_em_placement", None)

    def print_placement_report(self, qc: QuantumCircuit) -> None:
        rep = self.placement_report(qc)
        if rep is None:
            print("  (no placement metadata -- circuit did not come from "
                  "select_best_qubits)")
            return
        print("  --- placement ---")
        print(f"    chiplet(s)        : {rep['chiplet_ids']}")
        print(f"    initial layout    : {rep['initial_layout']}")
        print(f"    hub (virtual)     : {rep['hub_qubits']}")
        print(f"    routed 2q gates   : {rep['n_2q']}")
        print(f"    ~SWAP pairs       : {rep['n_swap_pairs']}")
        print(f"    intermodule gates : {rep['n_intermodule']}"
              + ("  <-- SPILLED across chiplets" if rep["n_intermodule"] else "  (none)"))
        print(f"    est. fidelity     : {rep['est_fidelity']:.4f}"
              f"   [{rep['calibration']} calibration]")
        print(f"    candidates tried  : {rep['candidates_tried']}")

    # ---------------------------------------------------------------------

    def idle_profile(self, qc: QuantumCircuit) -> dict[int, float]:
        """
        Total idle time per physical qubit, in seconds, between that qubit's
        first and last operation.

        In QPE the controlled-U^(2^k) blocks run sequentially, so precision
        qubit 0 idles through every later controlled power, qubit 1 through
        all but one, and so on -- the idle time is roughly GEOMETRIC across
        the precision register. That predicts a qubit-dependent DDD benefit,
        largest on the earliest-acting precision qubit. This method is how
        you test that prediction: compare the profile against the measured
        per-qubit improvement.
        """
        layer_ops, layer_dur, busy = self._schedule(qc)
        out: dict[int, float] = {}
        for q, mask in busy.items():
            active = [i for i, b in enumerate(mask) if b]
            if len(active) < 2:
                out[q] = 0.0
                continue
            first, last = active[0], active[-1]
            out[q] = sum(layer_dur[i] for i in range(first, last)
                         if not mask[i])
        return out

    def _schedule(self, qc: QuantumCircuit):
        """
        Layer the circuit and give each layer a real duration.

        Layer duration = the longest operation in it, since operations in one
        layer act on disjoint qubits and run concurrently. This is where the
        timing model earns its keep: a layer of rz gates has duration 0, so
        it never masks an idle window.
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

    # Sequences whose pulses do NOT compose to the identity. Inserting one
    # mid-circuit applies a net rotation to the qubit -- it does not decouple,
    # it corrupts. Verified numerically by verify_sequences():
    #   XYXYX -> iX  (a net bit flip), NOT identity.
    NON_IDENTITY_SEQUENCES = ("XYXYX",)

    def apply_ddd(self, qc: QuantumCircuit, sequence: str = "XY4",
                  min_idle_s: float | None = None,
                  qubits: list[int] | None = None,
                  allow_non_identity: bool = False,
                  verbose: bool = False) -> QuantumCircuit:
        """
        Digital dynamical decoupling. CIRCUIT IN, CIRCUIT OUT, ZERO extra
        tasks -- that is DDD's main practical advantage over ZNE and REM.

        Idle windows are found from REAL DURATIONS, not from moment counts.
        A moment-based detector has no idea that a CZ is ~180 ns, an rz is
        free, and a readout is ~2 us; on QPE that mis-ranks which windows are
        worth decoupling. We schedule with the device timing model and insert
        only where the idle window is at least `min_idle_s` (default: the
        time of one 2q gate, i.e. any window big enough for a CZ to have
        fitted).

        Pulses are spread uniformly across the window (CPMG-style: pulse p at
        (p + 0.5)/P of the way through), which is what makes the sequence a
        refocusing filter rather than a burst of gates at one instant.

        sequence:
          "XX", "YY", "XY4", "XYXYX"  -- real decoupling sequences
          "II"                        -- NULL CONTROL. Inserts identity gates
             of the same count and placement. If "II" helps as much as "XX",
             the effect is not decoupling and something is wrong with the
             experiment. Caveat: a compiler is entitled to delete identity
             gates outright, so on hardware this control is only meaningful
             inside a verbatim box -- and even then it tests "did adding
             gates help", not "did adding PULSES help".

        The Rigetti compiler will happily cancel XX -> I. Submit the result
        inside a verbatim box and verify against the returned compiled
        program; `verify_ddd()` below does the counting half.
        """
        if sequence not in self.DDD_SEQUENCES:
            raise ValueError(f"unknown sequence {sequence!r}; "
                             f"choose from {sorted(self.DDD_SEQUENCES)}")
        if sequence in self.NON_IDENTITY_SEQUENCES and not allow_non_identity:
            raise ValueError(
                f"DD sequence {sequence!r} does NOT compose to the identity "
                f"(verify_sequences() gives iX, a net bit flip). Inserting it "
                f"into an idle window rotates the qubit's state instead of "
                f"refocusing it, which corrupts the circuit rather than "
                f"decoupling it. Use 'XY4' for a 4-pulse sequence. Pass "
                f"allow_non_identity=True only if you have separately "
                f"confirmed you want this."
            )
        pulses = self.DDD_SEQUENCES[sequence]
        if min_idle_s is None:
            min_idle_s = self.t_2q

        layer_ops, layer_dur, busy = self._schedule(qc)
        n_layers = len(layer_ops)

        # plan[layer_index][qubit] -> list of pulses to emit before that layer
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

        out._em_ddd = {
            "sequence":     sequence,
            "n_windows":    n_windows,
            "n_pulses":     n_pulses,
            "n_gates_added": out.size() - qc.size(),
            "min_idle_s":   min_idle_s,
            "crowded_windows": crowded,
        }
        if verbose:
            print(f"  --- DDD ({sequence}) ---")
            print(f"    idle windows filled : {n_windows}")
            print(f"    pulses inserted     : {n_pulses}")
            print(f"    native gates added  : {out.size() - qc.size()}")
            if crowded:
                print(f"    WARNING: {crowded} window(s) had fewer layers than "
                      f"pulses, so some pulses share a layer with no spacing "
                      f"between them -- those windows get little decoupling.")
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
        """
        Uniform (CPMG) placement: pulse p lands at (p + 0.5)/P of the window.
        Returns {layer_index: [pulse, ...]}.
        """
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
        return getattr(qc, "_em_ddd", None)

    @classmethod
    def verify_sequences(cls, tol: float = 1e-9) -> dict[str, dict]:
        """
        Check each DD sequence composes to the identity up to a global phase.

        A sequence that does NOT is not a decoupling sequence -- it applies a
        net rotation to the qubit's state, silently corrupting the circuit it
        was inserted into. Run this before trusting any sequence.
        """
        out = {}
        for name, pulses in cls.DDD_SEQUENCES.items():
            qc = QuantumCircuit(1)
            for pulse in pulses:
                cls._emit(qc, pulse, 0)
            M = Operator(qc).data
            # Identity up to global phase iff M = c*I with |c| = 1.
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
        Noise-scale a circuit by unitary folding. Circuit in, circuit out --
        but read the class docstring: one folded circuit is NOT a mitigated
        result. See `zne_circuit_set`.

        method:
          "global" -- C -> C (C^dag C)^k. Only odd integer lambda. Exact, but
                      coarse: lambda jumps 1 -> 3 -> 5, so you cannot sample
                      the 1..3 region where the extrapolation is best
                      conditioned. Present for compatibility with the earlier
                      scripts; not the default and not what the spec wants.
          "random" -- fold individual gates g -> g g^dag g, choosing which
                      gates at random (seeded). Supports FRACTIONAL lambda,
                      so lambda in {1, 1.5, 2, 2.5, 3} works. DEFAULT.
          "all"    -- like "random" but folds gates in circuit order rather
                      than at random, so the extra noise is concentrated
                      early. Useful as a systematic check on whether the
                      extrapolation depends on WHERE the noise was added.

        two_qubit_only=True folds only CZs. On superconducting hardware 2q
        error dominates, so this scales the noise that matters and leaves the
        (nearly free) 1q gates alone -- often a better-conditioned scaling
        than folding everything. It makes lambda a scale factor on 2q count,
        not on total gate count; say which you used when reporting.

        Measurements are never folded: readout is charged once per shot
        regardless of lambda.
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
        G = len(foldable)
        if G == 0:
            return qc.copy()

        # Each fold of one gate adds 2 gates, so to reach lambda*G gates we
        # need round(G*(lambda-1)/2) folds distributed over the gates.
        total_folds = int(round(G * (scale - 1) / 2))
        base_folds, extra = divmod(total_folds, G)

        if method == "random":
            rng = np.random.default_rng(seed)
            chosen = set(rng.choice(G, size=extra, replace=False).tolist()) if extra else set()
        else:
            chosen = set(range(extra))
        fold_count = {foldable[i]: base_folds + (1 if i in chosen else 0)
                      for i in range(G)}

        out = QuantumCircuit(*base.qregs, *base.cregs)
        out.global_phase = base.global_phase
        for i, inst in enumerate(base.data):
            out.append(inst.operation, inst.qubits, inst.clbits)
            for _ in range(fold_count.get(i, 0)):
                out.append(inst.operation.inverse(), inst.qubits, inst.clbits)
                out.append(inst.operation,           inst.qubits, inst.clbits)

        folded = self._reattach_measurements(qc, out)
        folded._em_zne = {"scale": scale, "method": method,
                          "two_qubit_only": two_qubit_only, "seed": seed}
        return folded

    def _fold_global(self, qc: QuantumCircuit, scale: int) -> QuantumCircuit:
        base = qc.remove_final_measurements(inplace=False)
        folded = base.copy()
        for _ in range((scale - 1) // 2):
            folded.compose(base.inverse(), inplace=True)
            folded.compose(base,           inplace=True)
        out = self._reattach_measurements(qc, folded)
        out._em_zne = {"scale": scale, "method": "global",
                       "two_qubit_only": False, "seed": None}
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
        Build the FULL set of circuits one ZNE estimate needs.

        >>> MULTI-TASK. This is the boundary of the circuit-in/circuit-out
        >>> contract. Each entry is a separate Braket task; the mitigated
        >>> number only exists after all of them return and you fit
        >>> P_correct(lambda) -> lambda = 0. Nothing here produces a
        >>> mitigated result on its own.

        Returns [{"scale", "circuit", "profile", "achieved_scale_2q", ...}].
        `achieved_scale_2q` is the measured ratio of 2q gates to the lambda=1
        circuit -- check it tracks lambda before spending shots, because
        fractional folding cannot hit every lambda exactly on a short
        circuit (there are only so many gates to fold).
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
        Readout-error-mitigation CALIBRATION circuits.

        >>> REM ADDS NO GATES TO YOUR CIRCUIT. It does not fit the
        >>> circuit-in/circuit-out contract at all. The mitigation is a
        >>> classical inversion applied to the COUNTS, using a confusion
        >>> matrix measured by the separate circuits this method returns.
        >>> Your submitted QPE circuit is byte-for-byte unchanged.

        model:
          "tensored" -- assume A = (x)_i A_i: one 2x2 confusion matrix per
                        qubit, calibrated by preparing |0...0> and |1...1>.
                        2 circuits, independent of register size.
          "full"     -- calibrate all 2^n basis states, capturing readout
                        CROSSTALK between neighbouring resonators. 2^n
                        circuits: fine to n=5, refuses beyond n=8.

        Running both and differencing them measures the crosstalk directly,
        which is a result in its own right.

        Inversion (in the analysis, not here): do NOT use a raw pseudo-
        inverse -- A^-1 p produces negative quasi-probabilities. Use a
        constrained least-squares projection onto the probability simplex,
        and report the condition number of A plus bootstrap error bars,
        because A^-1 amplifies shot noise. REM trades bias for variance and
        both sides need reporting.
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
            circ._em_rem = {"model": model, "label": label,
                            "prepared_bits": bits, "physical_qubits": phys}
            out.append((label, circ))
        return out

    # =====================================================================
    # Verification -- the compiler is not on your side
    # =====================================================================

    def verify_ddd(self, before: QuantumCircuit, after: QuantumCircuit) -> dict:
        """
        Client-side check that DDD actually added the gates it claims. Pair
        this with a hardware-side check against
        task.metadata()["additionalMetadata"]["rigettiMetadata"]["compiledProgram"]
        -- if the DD gates are missing there, the compiler ate them and the
        run is INVALID, not merely unmitigated. Flag it; do not average it in.
        """
        pb, pa = count_native(before), count_native(after)
        rep = self.ddd_report(after) or {}
        return {
            "sequence":      rep.get("sequence"),
            "n_1q_before":   pb["n_1q"], "n_1q_after": pa["n_1q"],
            "n_2q_before":   pb["n_2q"], "n_2q_after": pa["n_2q"],
            "gates_added":   pa["n_gates"] - pb["n_gates"],
            "pulses_planned": rep.get("n_pulses"),
            "two_qubit_unchanged": pb["n_2q"] == pa["n_2q"],
            "ok": (pa["n_gates"] > pb["n_gates"]) and (pb["n_2q"] == pa["n_2q"]),
        }

    def verify_zne(self, circuit_set: list[dict], tol: float = 0.15) -> dict:
        """
        Check the folded circuits' 2q counts actually track lambda. Folding
        that the compiler undoes (U U^dag U -> U is exactly the kind of
        simplification Quilc does) leaves every lambda executing as lambda=1
        and the extrapolation silently meaningless.
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
                "note": "client-side only; re-check against the compiled "
                        "program returned by the device"}
