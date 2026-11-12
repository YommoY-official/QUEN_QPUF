"""
iqm_qpuf_common.py
==================
Circuit builders for a genuinely sequential two-stage PE-QPUF on IQM
(Garnet / Emerald) using Braket's experimental dynamic circuits
(`measure_ff`, `cc_prx`), submitted as VERBATIM programs.

Why the ancillas: IQM does not return the outcome of a `measure_ff`. The only
way to keep m1 is to copy it at the moment of measurement: a fresh ancilla
a_i gets `cc_prx(pi, 0, key_i)` (an X flip iff the P1_i result was 1). P1 is
also re-read at the end; ancilla == re-read P1 witnesses that the
measurement really happened mid-circuit, and its disagreement rate is the
decay of P1 during stage 2.

Arms
----
  A       deferred twin: same sites, same native gates, NO measure_ff/cc_prx.
  B       sequential: stage-1 QPE, measure_ff every P1 qubit, ancilla copy,
          stage-2 QPE, measure P2, ancillas, P1.
  Breuse  after measure_ff, P1_i is reset by cc_prx on itself and reused as
          P2_i (the "token stored, qubits recycled" variant).

Corrected circuit only (CLAUDE.md s4): precision qubit k controls U^(2^(d-1-k))
(MSB-first), inverse QFT is `QFTGate(d).inverse()` on prec[::-1]. Decode of one
stage: m = sum_k c_k 2^(d-1-k)  (c_k = outcome of precision qubit k). This is
the same per-stage bit reversal validated for the fixed circuit in
docs/plot_ledger/make_notes_figs.py.
"""
import json, os, re
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.transpiler import CouplingMap

DEVICES = {  # price: AWS Braket pricing page (per task, per shot), USD
    "garnet":  dict(arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",  region="eu-north-1",
                    n=20, task_usd=0.30, shot_usd=0.00145),
    "emerald": dict(arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald", region="eu-north-1",
                    n=54, task_usd=0.30, shot_usd=0.00160),
}
RUN_TAG = "iqm-mcm-20261112"
HERE = os.path.dirname(os.path.abspath(__file__))
PI = float(np.pi)


# -- unitaries ---------------------------------------------------------------
def haar_random_unitary(d, rng):
    """Same generator as ionq_mitigation_common (Mezzadri QR with phase fix)."""
    Z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def known_phase_unitary(n_targ, phi):
    """diag(1,...,1,e^{2 pi i phi}): target |1..1> has eigenphase phi exactly."""
    u = np.ones(2 ** n_targ, complex); u[-1] = np.exp(2j * np.pi * phi)
    return np.diag(u)


def encode_unitary(U):
    return {"shape": list(U.shape), "real": U.real.tolist(), "imag": U.imag.tolist()}


# -- device caps -------------------------------------------------------------
def placeholder_caps(device):
    """Square lattice, 1-based like Braket's IQM indexing. NOT the real chip:
    run query_device_caps.py on the DCV and the real file replaces this."""
    n = DEVICES[device]["n"]; w = 5 if n == 20 else 9
    edges = [(i, i + 1) for i in range(1, n + 1) if i % w and i + 1 <= n] + \
            [(i, i + w) for i in range(1, n + 1) if i + w <= n]
    return dict(device=device, nodes=list(range(1, n + 1)), edges=edges, groups=None, real=False)


def load_caps(device):
    p = os.path.join(HERE, f"device_caps_{device}.json")
    if os.path.exists(p):
        c = json.load(open(p)); c["edges"] = [tuple(e) for e in c["edges"]]; c["real"] = True
        return c
    return placeholder_caps(device)


def _adj(caps):
    a = {q: set() for q in caps["nodes"]}
    for x, y in caps["edges"]:
        a[x].add(y); a[y].add(x)
    return a


def _bfs_order(adj, src, banned):
    seen, order, fr = {src}, [], [src]
    while fr:
        nxt = []
        for u in fr:
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v); nxt.append(v)
                    if v not in banned: order.append(v)
        fr = nxt
    return order


def _group_of(caps, q):
    if not caps.get("groups"): return None
    for i, g in enumerate(caps["groups"]):
        if q in g: return i
    return None


# -- logical stages (qiskit, pre-routing) -------------------------------------
def stage_logical(d, nt, U, init=None):
    """qubits 0..d-1 precision, d..d+nt-1 target. init: None | 'haar' (ry/rz seed 99) | 'ones'."""
    qc = QuantumCircuit(d + nt)
    if init == "ones":
        for i in range(nt): qc.x(d + i)
    elif init is not None:
        r = np.random.default_rng(init)       # same draw order as make_notes_figs.sim_counts
        for i in range(nt):
            qc.ry(r.uniform(0, np.pi), d + i); qc.rz(r.uniform(0, 2 * np.pi), d + i)
    qc.h(range(d))
    for k in range(d):
        P = U
        for _ in range(d - 1 - k): P = P @ P           # U^(2^(d-1-k)): MSB-first
        qc.append(UnitaryGate(P).control(1), [k] + list(range(d, d + nt)))
    qc.append(QFTGate(d).inverse(), list(range(d))[::-1])
    return qc


def _route(logical, sites, caps, allowed, seed=7):
    """Transpile `logical` (qubit j starts on physical sites[j]) to native
    {r(=PRx), cz} on the device graph restricted to `allowed`.
    Returns (ops, final_sites). SABRE may permute qubits; final_sites tracks it."""
    allowed = sorted(allowed); ix = {p: i for i, p in enumerate(allowed)}
    cm = CouplingMap([[ix[a], ix[b]] for a, b in caps["edges"] if a in ix and b in ix] +
                     [[ix[b], ix[a]] for a, b in caps["edges"] if a in ix and b in ix])
    tq = transpile(logical, coupling_map=cm, basis_gates=["r", "cz"], optimization_level=1,
                   initial_layout=[ix[s] for s in sites], seed_transpiler=seed, routing_method="sabre")
    fin = tq.layout.final_index_layout(filter_ancillas=False)   # virtual (post-layout) -> final physical
    ops = []
    for inst in tq.data:
        n = inst.operation.name; q = [tq.find_bit(b).index for b in inst.qubits]
        if n == "r":   ops.append(("prx", allowed[q[0]], float(inst.operation.params[0]), float(inst.operation.params[1])))
        elif n == "cz": ops.append(("cz", allowed[q[0]], allowed[q[1]]))
        elif n == "barrier": pass
        else: raise ValueError(f"non-native gate after transpile: {n}")
    return ops, [allowed[fin[j]] for j in range(len(sites))]   # fin is indexed by logical qubit j


# -- program planner ----------------------------------------------------------
def build_program(arm, d, nt, U, init, caps, seed=7, layout="primary"):
    """Try every hub in the region (best-connected first) and keep the program with the
    fewest CZ. A hub is rejected when removing P1/ancilla sites leaves stage 2 unroutable."""
    region = set(max(caps["groups"], key=len)) if caps.get("groups") else set(caps["nodes"])
    rc = dict(caps, nodes=sorted(region), edges=[e for e in caps["edges"] if e[0] in region and e[1] in region])
    adj = _adj(rc); best = None
    for hub in sorted(rc["nodes"], key=lambda q: (-len(adj[q]), q)):
        for sd in (seed, seed + 1, seed + 2):       # routing seed also moves where P1 ends up
            try: p = _build(arm, d, nt, U, init, rc, hub, sd, layout)
            except Exception: continue
            if best is None or gate_counts(p)["cz"] < gate_counts(best)["cz"]: best = p
    if best is None: raise RuntimeError("no hub in the region gives a routable program")
    return best


def _build(arm, d, nt, U, init, caps, hub, seed=7, layout="primary"):
    """layout 'primary': ancilla anywhere in P1_i's group (cc_prx is classically controlled).
    layout 'adjacent': every P1_i-ancilla_i pair is also a coupler (fallback if the device
    rejects non-adjacent feed-forward). P1/ancilla avoid caps['ro_avoid'] (poor readout):
    readout error there would be mistaken for measurement disturbance in the witness."""
    avoid = set(caps.get("ro_avoid") or [])
    """Returns dict(body=[ops], measure=[(role, idx, phys)], arm, d, nt, sites...).
    body ops: ('prx',q,th,ph) ('cz',a,b) ('measure_ff',q,key) ('cc_prx',q,th,ph,key)."""
    assert arm in ("A", "B", "Breuse")
    adj = _adj(caps)
    t_sites = [hub] + _bfs_order(adj, hub, {hub})[:nt - 1]
    p1 = _bfs_order(adj, hub, set(t_sites) | avoid)[:d]
    body, nodes = [], set(caps["nodes"])
    ops, fin = _route(stage_logical(d, nt, U, init), p1 + t_sites, caps, nodes, seed)
    body += ops; p1_f, t_f = fin[:d], fin[d:]

    # ancilla sites: nearest free site in the same group as the P1 qubit it copies.
    # Reserved in EVERY arm so arms share one layout and one stage-2 routing.
    if set(p1_f) & avoid: raise RuntimeError("P1 routed onto a poor-readout qubit")
    taken, anc = set(p1_f) | set(t_f), []
    for s in p1_f:
        cand = [q for q in _bfs_order(adj, s, taken | avoid) if _group_of(caps, q) == _group_of(caps, s)
                and (layout == "primary" or q in adj[s])]
        if not cand: raise RuntimeError("no ancilla site")
        anc.append(cand[0]); taken.add(cand[0])
    if arm == "B":
        for i in range(d):
            body += [("measure_ff", p1_f[i], i), ("cc_prx", anc[i], PI, 0.0, i)]
    if arm == "Breuse":
        for i in range(d):
            body += [("measure_ff", p1_f[i], i), ("cc_prx", anc[i], PI, 0.0, i), ("cc_prx", p1_f[i], PI, 0.0, i)]
        p2_start = list(p1_f)
    else:
        sub = dict(caps, nodes=sorted(nodes - set(anc) - set(p1_f)),
                   edges=[e for e in caps["edges"] if not ({e[0], e[1]} & (set(anc) | set(p1_f)))])
        p2_start = [q for q in _bfs_order(_adj(sub), t_f[0], set(t_f))][:d]
        if len(p2_start) < d: raise RuntimeError("not enough connected sites for stage 2")
    # Stage 2 may not route through P1 (they hold m1 until the final re-read and
    # a SWAP through them would move the state we want to read back in place).
    allowed = nodes - set(anc) - (set() if arm == "Breuse" else set(p1_f))
    ops, fin = _route(stage_logical(d, nt, U), p2_start + t_f, caps, allowed, seed)
    body += ops; p2_f = fin[:d]
    p1_final = p1_f
    meas = [("P2", i, p2_f[i]) for i in range(d)]
    if arm in ("B", "Breuse"): meas += [("anc", i, anc[i]) for i in range(d)]
    if arm != "Breuse": meas += [("P1", i, p1_final[i]) for i in range(d)]
    order = sorted(set(q for _, _, q in meas))
    assert len(order) == len(meas), "two logical qubits ended on one physical qubit"
    return dict(arm=arm, d=d, nt=nt, body=body, measure=meas, order=order,
                p1_stage1=p1_f, anc=anc, t_sites=t_f, p2=p2_f)


# -- Braket emission ----------------------------------------------------------
def to_braket(prog):
    """Verbatim Braket circuit on PHYSICAL qubits, serialised by the SDK itself."""
    from braket.circuits import Circuit
    from braket.experimental_capabilities import EnableExperimentalCapability
    with EnableExperimentalCapability():
        inner = Circuit()
        for o in prog["body"]:
            if o[0] == "prx": inner.prx(o[1], o[2], o[3])
            elif o[0] == "cz": inner.cz(o[1], o[2])
            elif o[0] == "measure_ff": inner.measure_ff(o[1], o[2])
            else: inner.cc_prx(o[1], o[2], o[3], o[4])
        c = Circuit().add_verbatim_box(inner)
        for q in prog["order"]: c.measure(q)         # bit j = j-th smallest measured physical qubit
    return c


def to_qasm(prog):
    from braket.circuits.serialization import (IRType, OpenQASMSerializationProperties, QubitReferenceType)
    return to_braket(prog).to_ir(ir_type=IRType.OPENQASM, serialization_properties=
                                 OpenQASMSerializationProperties(qubit_reference_type=QubitReferenceType.PHYSICAL)).source


_RX = {
    "prx": re.compile(r"prx\(([^,]+),\s*([^)]+)\)\s*\$(\d+);"),
    "cz": re.compile(r"cz\s*\$(\d+),\s*\$(\d+);"),
    "mff": re.compile(r"measure_ff\((\d+)\)\s*\$(\d+);"),
    "cc": re.compile(r"cc_prx\(([^,]+),\s*([^,]+),\s*(\d+)\)\s*\$(\d+);"),
    "m": re.compile(r"b\[(\d+)\]\s*=\s*measure\s*\$(\d+);"),
}


def ops_from_qasm(qasm):
    """Parse the program that will actually be submitted back into ops + measure map."""
    body, meas = [], {}
    for ln in qasm.splitlines():
        ln = ln.strip()
        if m := _RX["prx"].fullmatch(ln): body.append(("prx", int(m[3]), float(m[1]), float(m[2])))
        elif m := _RX["cz"].fullmatch(ln): body.append(("cz", int(m[1]), int(m[2])))
        elif m := _RX["mff"].fullmatch(ln): body.append(("measure_ff", int(m[2]), int(m[1])))
        elif m := _RX["cc"].fullmatch(ln): body.append(("cc_prx", int(m[4]), float(m[1]), float(m[2]), int(m[3])))
        elif m := _RX["m"].fullmatch(ln): meas[int(m[1])] = int(m[2])
        elif ln and not ln.startswith(("OPENQASM", "bit", "qubit", "#pragma", "box", "{", "}")):
            raise ValueError(f"unparsed QASM line: {ln!r}")
    return body, [meas[i] for i in sorted(meas)]


# -- validation helpers: ops -> Aer ------------------------------------------
def simulate_ops(body, measured, shots, seed=1):
    """Aer run of the parsed program: measure_ff -> mid-circuit measure into a key
    bit, cc_prx -> if_test on that bit. Returns raw counts, key char j = b[j]."""
    from qiskit import ClassicalRegister, QuantumRegister
    from qiskit_aer import AerSimulator
    used = sorted({q for o in body for q in (o[1:3] if o[0] == "cz" else o[1:2])} | set(measured))
    ix = {p: i for i, p in enumerate(used)}
    keys = sorted({o[2] for o in body if o[0] == "measure_ff"})
    q = QuantumRegister(len(used)); kreg = ClassicalRegister(max(1, len(keys)), "k"); b = ClassicalRegister(len(measured), "b")
    qc = QuantumCircuit(q, kreg, b)
    for o in body:
        if o[0] == "prx": qc.r(o[2], o[3], q[ix[o[1]]])
        elif o[0] == "cz": qc.cz(q[ix[o[1]]], q[ix[o[2]]])
        elif o[0] == "measure_ff": qc.measure(q[ix[o[1]]], kreg[keys.index(o[2])])
        else:
            with qc.if_test((kreg[keys.index(o[4])], 1)): qc.r(o[2], o[3], q[ix[o[1]]])
    for j, p in enumerate(measured): qc.measure(q[ix[p]], b[j])
    sim = AerSimulator(method="statevector", seed_simulator=seed)
    cnt = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    out = {}
    for k, v in cnt.items():
        bits = k.split()[0][::-1]                       # qiskit: b[n-1]..b[0] -> char j = b[j]
        out[bits] = out.get(bits, 0) + v
    return out


# -- decode (raw counts -> m1, m2) --------------------------------------------
def decode(raw_counts, prog_meta):
    """raw_counts: {bitstring: n}, char j = j-th smallest MEASURED physical qubit.
    prog_meta: dict with 'arm', 'd', 'measure' = [[role, idx, phys], ...], 'order'."""
    d, arm = prog_meta["d"], prog_meta["arm"]
    col = {(r, i): prog_meta["order"].index(p) for r, i, p in prog_meta["measure"]}
    val = lambda s, role: sum(int(s[col[(role, k)]]) << (d - 1 - k) for k in range(d))
    M = np.zeros((2 ** d, 2 ** d)); agree = tot = 0; n_all = 0
    for s, n in raw_counts.items():
        m2 = val(s, "P2")
        if arm == "A": m1 = val(s, "P1")
        else:
            m1 = val(s, "anc")
            if arm == "B": agree += n * (m1 == val(s, "P1"))
        M[m1, m2] += n; n_all += n
    return M / n_all, (agree / n_all if arm == "B" else None)


# -- reporting ----------------------------------------------------------------
def gate_counts(prog):
    b = prog["body"]; free = {}
    for o in b:
        qs = o[1:3] if o[0] == "cz" else o[1:2]
        t = max(free.get(q, 0) for q in qs) + 1
        for q in qs: free[q] = t
    return dict(cz=sum(o[0] == "cz" for o in b), prx=sum(o[0] == "prx" for o in b),
                measure_ff=sum(o[0] == "measure_ff" for o in b), cc_prx=sum(o[0] == "cc_prx" for o in b),
                qubits=len(prog["order"]) + len(prog["t_sites"]), depth=max(free.values()))


def check_dynamic_rules(prog, caps):
    """IQM rules: unique keys; cc_prx after its measure_ff; one controller per
    target; control and target in the same group. Returns (errors, warnings)."""
    err, warn = [], []
    seen_key, ctrl_of, src, keys_of = set(), {}, {}, {}
    for o in prog["body"]:
        if o[0] not in ("prx", "cz", "measure_ff", "cc_prx", "barrier"): err.append(f"non-native op {o[0]} in verbatim box")
        if o[0] == "measure_ff":
            if o[2] in seen_key: err.append(f"duplicate key {o[2]}")
            seen_key.add(o[2]); src[o[2]] = o[1]
        elif o[0] == "cc_prx":
            if o[4] not in seen_key: err.append(f"cc_prx key {o[4]} before its measure_ff")
            else:
                s, t = src[o[4]], o[1]
                keys_of.setdefault(t, set()).add(o[4])
                if len(keys_of[t]) > 1: err.append(f"qubit {t} controlled by more than one key {sorted(keys_of[t])}")
                if ctrl_of.setdefault(t, s) != s: err.append(f"qubit {t} controlled by two qubits")
                gs, gt = _group_of(caps, s), _group_of(caps, t)
                if caps.get("groups") and gs != gt: err.append(f"cc_prx control {s} (group {gs}) / target {t} (group {gt})")
    if not caps.get("groups"): warn.append("qubit groups unknown (no device_caps file): group rule NOT checked")
    fan = {}
    for t, s in ctrl_of.items(): fan.setdefault(s, []).append(t)
    for s, ts in fan.items():
        if len(ts) > 1: warn.append(f"measure_ff on {s} drives {len(ts)} targets {ts}: confirm the device allows one key to drive several cc_prx")
    if not caps["real"]: warn.append("PLACEHOLDER lattice, not the real chip")
    return err, warn


def est_success(prog, fid):
    """Product of median gate / readout fidelities: a rough survival probability, not a QPUF success rate."""
    c = gate_counts(prog)
    return fid["fcz"] ** c["cz"] * fid["f1q"] ** c["prx"] * fid["fro"] ** (len(prog["order"]) + c["measure_ff"])


def readout_cal_programs(qubits):
    """Two circuits on the given physical qubits: prepare |0...0> and |1...1>, measure all.
    The |0> circuit uses prx(0,0) (a no-op native gate) so both verbatim boxes are non-empty
    and have the same shape."""
    out = {}
    for st in ("0", "1"):
        body = [("prx", q, PI if st == "1" else 0.0, 0.0) for q in qubits]
        out[st] = dict(arm="cal", d=0, nt=0, body=body, order=sorted(qubits), measure=[("cal", i, q) for i, q in enumerate(sorted(qubits))],
                       t_sites=[], p1_stage1=[], anc=[], p2=[], cal_state=st)
    return out


def estimate_cost(device, n_tasks, shots):
    D = DEVICES[device]
    return n_tasks * D["task_usd"] + n_tasks * shots * D["shot_usd"]


def task_tags(workload, extra=None):
    t = {"Run": RUN_TAG, "Workload": workload}
    t.update({k: str(v) for k, v in (extra or {}).items()}); return t


def append_job_log(results_dir, record):
    os.makedirs(results_dir, exist_ok=True)
    p = os.path.join(results_dir, "job_log.txt")
    with open(p, "a") as f: f.write(json.dumps(record) + "\n")
    return p
