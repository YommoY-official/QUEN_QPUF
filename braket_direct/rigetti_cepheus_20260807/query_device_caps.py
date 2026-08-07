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

import argparse
import json
import os
import statistics
import sys

from braket.aws import AwsDevice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigetti_qpuf_common import DEVICE_ARN, CONN_CACHE, fetch_device_caps

# ARNs to try, in order, when looking for LIVE CALIBRATION.
#
# device_caps.json came back with specs = {} -- real connectivity, no
# per-qubit or per-edge fidelities -- which leaves the placement scorer
# unable to tell a good chiplet from a bad one (it can only avoid intermodule
# couplers, which is topology, not calibration). Rigetti publishes the
# calibration through provider.specs, so the fix is to find the endpoint that
# actually returns it.
#
# The second entry is the non-standard one: Braket device ARNs normally use
# the `braket` service namespace, not `quantum`, and Cepheus is a us-west-1
# device. The SDK does not validate either field -- AwsDevice.get_device_region
# just splits the ARN on ":" and takes field 3 -- so this ARN is accepted
# syntactically and simply opens a us-east-1 session. Whether AWS serves the
# device from there is a server-side question this script answers empirically
# rather than assuming: it tries each ARN and keeps the first that comes back
# with a populated specs block.
CALIBRATION_ARNS = [
    DEVICE_ARN,
    "arn:aws:quantum:us-east-1::device/qpu/rigetti/Cepheus-1-108Q",
]

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


def resolve_device(arns: list[str]):
    """
    Return (device, arn, specs) for the first ARN that yields CALIBRATION,
    falling back to the first that merely resolves.

    Reports every attempt. An ARN that resolves but returns no specs is not a
    failure worth aborting on -- it still carries the connectivity graph the
    routing depends on -- but it is worth saying out loud, because a silent
    specs = {} is exactly how the placement scorer ended up running blind.
    """
    first_ok = None
    for arn in arns:
        try:
            device = AwsDevice(arn)
            props  = device.properties
        except Exception as e:
            print(f"  {arn}\n      -> FAILED ({type(e).__name__}: {str(e)[:160]})")
            continue

        specs = _specs_of(props)
        n1 = len(specs.get("1Q", {}) or {})
        n2 = len(specs.get("2Q", {}) or {})
        if n1 or n2:
            print(f"  {arn}\n      -> OK, calibration present (1Q: {n1}, 2Q: {n2})")
            return device, arn, specs
        print(f"  {arn}\n      -> resolved, but provider.specs is EMPTY")
        if first_ok is None:
            first_ok = (device, arn, specs)

    if first_ok is None:
        print("\nERROR: no ARN resolved. Check credentials, region, and that the")
        print("       device name is current.")
        sys.exit(1)
    return first_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arn", action="append", metavar="ARN",
                    help="device ARN to query (repeatable; overrides the "
                         "built-in candidate list)")
    args = ap.parse_args()

    arns = args.arn or CALIBRATION_ARNS
    print("Resolving device / looking for live calibration ...\n")
    device, arn, specs = resolve_device(arns)
    props = device.properties
    print(f"\nUsing   : {arn}")

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
        print("  Try again inside the reservation window, or with --arn <other>;")
        print("  calibration is often only published while the device is available.")
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
