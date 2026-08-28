#!/usr/bin/env python3
"""
query_device_caps.py
=====================
Copied near-as-is from braket_direct/ionq_20260601/query_device_caps.py.

Prints IonQ Forte-Enterprise-1's OpenQASM action flags -- in particular
supportsPartialVerbatimBox, which is the single most important unanswered
question blocking real-hardware ZNE/DDD numbers under this campaign (see
MITIGATION_BRIEF_IONQ.md Section 9.1). Also scans the full properties dump
for any mention of 'reset' / 'midcircuit' capability.

THIS SCRIPT WAS NOT RUN AS PART OF CREATING THIS CAMPAIGN DIRECTORY -- it
requires AWS credentials and a live connection to the device, neither of
which is available in the environment this file was written in. Run it from
the DCV box and save BOTH the raw stdout and a device_caps.json /
device_properties_raw.json into this directory, mirroring
braket_direct/rigetti_cepheus_20260807/'s layout, before trusting any
verbatim-dependent claim in this campaign.
"""

from braket.aws import AwsDevice

DEVICE_ARN = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"
dev = AwsDevice(DEVICE_ARN)
props = dev.properties

print(f"name   : {dev.name}")
print(f"qubits : {props.paradigm.qubitCount}")
print()

for action_name, action in props.action.items():
    print(f"=== action: {action_name} ===")
    for attr in ("supportedOperations",
                 "supportedPragmas",
                 "forbiddenPragmas",
                 "supportedResultTypes",
                 "supportedModifiers",
                 "supportPhysicalQubits",
                 "requiresContiguousQubitIndices",
                 "requiresAllQubitsMeasurement",
                 "supportsPartialVerbatimBox",
                 "supportsUnassignedMeasurements"):
        val = getattr(action, attr, None)
        if val is not None:
            print(f"  {attr}: {val}")
    print()

# Look for any mention of 'reset' or 'mid' anywhere in the props dump
import json

dump = props.json() if hasattr(props, "json") else json.dumps(
    props.dict() if hasattr(props, "dict") else
    props.model_dump()
)
hits = [l for l in dump.split(",") if "reset" in
        l.lower() or "midcircuit" in l.lower()]
print("=== mentions of 'reset' / 'midcircuit' in props == = ")
for h in hits:
    print(f"  {h.strip()}")
if not hits:
    print("  (none — reset/MCM is not advertised by the device)")
