#!/usr/bin/env python3
"""
query_device_caps.py
====================
RUN THIS FIRST on the DCV (it needs AWS credentials).

Dumps Rigetti Cepheus-1-108Q's live capabilities and caches the ones the
other scripts need into device_caps.json:

  - qubit count and the REAL connectivity graph
      -> so transpile_for_rigetti() routes onto the actual lattice instead
         of a placeholder grid. Gate counts are meaningless without this:
         QPE forces every precision qubit to reach the target, and on a
         degree-4 lattice that is paid for in SWAPs.
  - native gate set
      -> confirms rz/rx/cz is the right transpile basis.
  - OpenQASM action flags
      -> supportsPartialVerbatimBox decides whether ZNE folding can be
         protected from Quilc's optimiser;
         supportsUnassignedMeasurements / requiresAllQubitsMeasurement /
         requiresContiguousQubitIndices decide how measurements must be
         written.
  - calibration data (per-qubit and per-edge fidelities) if published
      -> feed the medians back into F1Q / F2Q in rigetti_qpuf_common.py.

Also prints whether `reset` appears anywhere in the device properties, which
is the check that told us the IonQ two-stage-with-MCM form cannot be reused
here.
"""

import json
import os
import statistics
import sys

from braket.aws import AwsDevice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigetti_qpuf_common import DEVICE_ARN, CONN_CACHE, fetch_device_caps


def main():
    print(f"Querying {DEVICE_ARN} ...\n")
    device = AwsDevice(DEVICE_ARN)
    props  = device.properties

    print(f"name    : {device.name}")
    print(f"status  : {device.status}")
    print(f"qubits  : {props.paradigm.qubitCount}")
    print(f"native  : {getattr(props.paradigm, 'nativeGateSet', None)}")
    print()

    caps = fetch_device_caps(device)

    # ── Connectivity summary ──────────────────────────────────────────────────
    graph = caps["connectivity"]
    if graph:
        degrees = [len(v) for v in graph.values()]
        n_edges = sum(degrees) // 2
        print("=== connectivity ===")
        print(f"  nodes        : {len(graph)}")
        print(f"  edges        : {n_edges}")
        print(f"  degree       : min={min(degrees)} max={max(degrees)} "
              f"mean={statistics.mean(degrees):.2f}")
        print(f"  fullyConnected: {caps['fully_connected']}")
    else:
        print("=== connectivity ===\n  (device reports no connectivityGraph)")
    print()

    # ── OpenQASM action flags ─────────────────────────────────────────────────
    print("=== braket.ir.openqasm.program ===")
    for k, v in caps["openqasm"].items():
        if isinstance(v, list) and len(v) > 12:
            print(f"  {k}: [{len(v)} entries] {v[:12]} ...")
        else:
            print(f"  {k}: {v}")
    print()

    verbatim_ok = caps["openqasm"].get("supportsPartialVerbatimBox")
    print(f"  --> verbatim boxes supported: {verbatim_ok}")
    if not verbatim_ok:
        print("      WARNING: without verbatim, Rigetti's Quilc recompiles the")
        print("      circuit and will CANCEL the C^dag C pairs that ZNE folding")
        print("      inserts — every lambda collapses back to 1 and ZNE is void.")
    print()

    # ── Calibration data ──────────────────────────────────────────────────────
    specs = getattr(getattr(props, "provider", None), "specs", None)
    if specs:
        print("=== calibration (from provider.specs) ===")
        one_q = specs.get("1Q", {})
        two_q = specs.get("2Q", {})

        def _median_of(d, keys):
            for key in keys:
                vals = [m[key] for m in d.values()
                        if isinstance(m, dict) and isinstance(m.get(key), (int, float))]
                if vals:
                    return key, statistics.median(vals), len(vals)
            return None, None, 0

        k1, m1, n1 = _median_of(one_q, ["f1QRB", "f1QRB_std_err", "fActiveReset"])
        if m1 is not None:
            print(f"  1Q  median {k1} = {m1:.5f}  (n={n1})")
            print(f"      -> set F1Q = {m1:.5f} in rigetti_qpuf_common.py")
        k2, m2, n2 = _median_of(two_q, ["fCZ", "fISWAP", "fCPHASE", "fXY"])
        if m2 is not None:
            print(f"  2Q  median {k2} = {m2:.5f}  (n={n2})")
            print(f"      -> set F2Q = {m2:.5f} in rigetti_qpuf_common.py")

        t1 = [m["T1"] for m in one_q.values()
              if isinstance(m, dict) and isinstance(m.get("T1"), (int, float))]
        t2 = [m["T2"] for m in one_q.values()
              if isinstance(m, dict) and isinstance(m.get("T2"), (int, float))]
        if t1:
            print(f"  T1  median = {statistics.median(t1)*1e6:.1f} us")
        if t2:
            print(f"  T2  median = {statistics.median(t2)*1e6:.1f} us")
        caps["calibration_medians"] = {"F1Q": m1, "F2Q": m2,
                                       "T1_s": statistics.median(t1) if t1 else None,
                                       "T2_s": statistics.median(t2) if t2 else None}
        print()

    # ── reset / MCM probe ─────────────────────────────────────────────────────
    dump = props.json() if hasattr(props, "json") else json.dumps(props.dict())
    hits = [tok.strip() for tok in dump.split(",")
            if "reset" in tok.lower() or "midcircuit" in tok.lower()]
    print("=== mentions of 'reset' / 'midcircuit' ===")
    for h in hits[:20]:
        print(f"  {h}")
    if not hits:
        print("  (none — no mid-circuit measurement / reset; the QPUF must use")
        print("   the deferred-measurement two-register form, which is what")
        print("   build_qpuf_two_stage() does.)")
    print()

    with open(CONN_CACHE, "w") as f:
        json.dump(caps, f, indent=2)
    print(f"Cached device capabilities -> {CONN_CACHE}")
    print("submit_test.py / submit_qpuf_mitigation.py will now route against")
    print("the real Cepheus lattice.")


if __name__ == "__main__":
    main()
