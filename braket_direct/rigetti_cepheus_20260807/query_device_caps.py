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
      -> feed the medians back into F1Q / F2Q in rigetti_qpuf_common.py, and
         give ErrorMitigation.select_best_qubits something better than a flat
         default to score chiplets with.

Also prints whether `reset` appears anywhere in the device properties, which
is the check that told us the IonQ two-stage-with-MCM form cannot be reused
here.

Finding the calibration
-----------------------
The cached device_caps.json has real connectivity but specs = {}, so
placement currently runs on topology alone. Calibration comes off the SAME
device ARN the submissions use -- there is no second endpoint to try. The
reservation ARN is not one: it is account-scoped, names a booked window
rather than hardware, and has no .properties.

This script writes the untouched properties blob to device_properties_raw.json
alongside the parsed cache, so calibration published under a key the parser
does not know about is still on disk, and it says plainly when specs comes
back empty instead of recording that silently.

    python query_device_caps.py                 # the device ARN
    python query_device_caps.py --arn <ARN>     # override it

Calibration is often only published while the device is AVAILABLE, so if it
comes back empty, try again inside the reservation window before concluding
Rigetti does not publish it.
"""

import argparse
import json
import os
import statistics
import sys

from braket.aws import AwsDevice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigetti_qpuf_common import DEVICE_ARN, CONN_CACHE, fetch_device_caps

RAW_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "device_properties_raw.json")


def _specs_of(props) -> dict:
    """provider.specs as a plain dict, whatever shape the SDK hands back."""
    provider = getattr(props, "provider", None)
    raw = getattr(provider, "specs", None) if provider is not None else None
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    # Pydantic model rather than a bare dict -- unwrap instead of discarding.
    for attr in ("dict", "model_dump"):
        fn = getattr(raw, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arn", default=DEVICE_ARN, metavar="ARN",
                    help=f"device ARN to query (default: {DEVICE_ARN})")
    args = ap.parse_args()

    # ONE device ARN, the same one the on-demand and reserved submissions use.
    # RES_ARN is NOT an alternative to it: a reservation ARN is account-scoped
    # and names a booked WINDOW, not hardware, so it has no .properties to read
    # calibration from. That is why DirectReservation takes the device and the
    # reservation as two separate arguments -- the reservation only decides
    # which queue a task lands in.
    arn = args.arn
    print(f"Querying {arn} ...\n")
    device = AwsDevice(arn)
    props  = device.properties
    specs  = _specs_of(props)

    print(f"name    : {device.name}")
    print(f"status  : {device.status}")
    print(f"qubits  : {props.paradigm.qubitCount}")
    print(f"native  : {getattr(props.paradigm, 'nativeGateSet', None)}")
    print()

    # Raw properties, verbatim. The parsed caps below keep only what the other
    # scripts consume; if calibration turns up under a key this script does not
    # know about, it is in here rather than lost.
    try:
        raw = props.json() if hasattr(props, "json") else json.dumps(props.dict())
        with open(RAW_DUMP, "w") as f:
            f.write(raw)
        print(f"Raw device properties -> {RAW_DUMP}  ({len(raw):,} chars)")
    except Exception as e:
        print(f"(could not dump raw properties: {type(e).__name__}: {e})")
    print()

    caps = fetch_device_caps(device)
    caps["calibration_source_arn"] = arn
    # fetch_device_caps filters specs down to numeric fields; if it came back
    # empty but the device did publish calibration, keep the unfiltered copy
    # rather than shipping a device_caps.json that silently has none.
    if specs and not caps.get("specs"):
        caps["specs"] = specs

    # -- Connectivity summary --------------------------------------------------
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

    # -- OpenQASM action flags -------------------------------------------------
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
        print("      inserts -- every lambda collapses back to 1 and ZNE is void.")
    print()

    # -- Calibration data ------------------------------------------------------
    if not specs:
        print("=== calibration (from provider.specs) ===")
        print("  NONE. The device resolved but published no per-qubit/per-edge")
        print("  specs, so device_caps.json will again carry specs = {}.")
        print("  Consequences, so this is not discovered later:")
        print("    - F1Q / F2Q in rigetti_qpuf_common.py stay at their guessed")
        print("      defaults, so every 'est F' number is a model, not a measurement.")
        print("    - ErrorMitigation.select_best_qubits scores all on-chiplet CZs")
        print("      identically, so it can avoid intermodule couplers but CANNOT")
        print("      pick the best-calibrated chiplet.")
        print("  Calibration is often only published while the device is AVAILABLE,")
        print(f"  so try again inside the reservation window (status now: {device.status}).")
        print(f"  Check {os.path.basename(RAW_DUMP)} for calibration under another key.")
        print()
    else:
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

    # -- reset / MCM probe -----------------------------------------------------
    dump = props.json() if hasattr(props, "json") else json.dumps(props.dict())
    hits = [tok.strip() for tok in dump.split(",")
            if "reset" in tok.lower() or "midcircuit" in tok.lower()]
    print("=== mentions of 'reset' / 'midcircuit' ===")
    for h in hits[:20]:
        print(f"  {h}")
    if not hits:
        print("  (none -- no mid-circuit measurement / reset; the QPUF must use")
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
