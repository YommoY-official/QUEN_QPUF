from braket.aws import AwsDevice

DEVICE_ARN = "arn:aws:braket:us-east-1::device/qpu/ionq / Forte - Enterprise - 1"
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
