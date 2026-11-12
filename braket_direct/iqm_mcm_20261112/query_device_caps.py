#!/usr/bin/env python3
"""
query_device_caps.py -- READ-ONLY (no task is created). Run on the DCV.
Writes device_properties_raw_<dev>.json (everything the API returns) and
device_caps_<dev>.json (edges, nodes, native gates, qubit groups if present).
`groups` is taken from any property key containing 'group'; if the device
publishes none it is left null and submit_qpuf_mcm.py says the group rule is
UNCHECKED -- fill it in by hand from the Braket IQM documentation.

    python query_device_caps.py garnet|emerald
"""
import json, sys
from braket.aws import AwsDevice
from iqm_qpuf_common import DEVICES

dev = sys.argv[1]; d = AwsDevice(DEVICES[dev]["arn"])
raw = json.loads(d.properties.json()); json.dump(raw, open(f"device_properties_raw_{dev}.json", "w"), indent=1)
conn = raw["paradigm"]["connectivity"]["connectivityGraph"]
edges = sorted({tuple(sorted((int(a), int(b)))) for a, bs in conn.items() for b in bs})
nodes = sorted({int(a) for a in conn} | {b for e in edges for b in e})


def find(o, key, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if key in k.lower(): yield path + "/" + k, v
            yield from find(v, key, path + "/" + k)


groups = [(p, v) for p, v in find(raw, "group")]
print("native gates:", raw["paradigm"].get("nativeGateSet")); print("group-like keys:", [p for p, _ in groups] or "none")
json.dump(dict(device=dev, nodes=nodes, edges=edges, groups=None, native_gates=raw["paradigm"].get("nativeGateSet"),
               group_candidates=[{"path": p, "value": v} for p, v in groups]), open(f"device_caps_{dev}.json", "w"), indent=1)
print(f"{dev}: {len(nodes)} qubits, {len(edges)} couplers. EDIT device_caps_{dev}.json: set 'groups' to a list of qubit lists.")
