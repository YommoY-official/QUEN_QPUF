import numpy as np
import time
import json
from braket.aws import AwsDevice, AwsSession
from braket.circuits import Circuit
from braket.device_schema import GateModelSimulatorDeviceCapabilities
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.rigetti import RigettiDeviceCapabilities
from braket.device_schema.iqm import IqmDeviceCapabilities

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET   = "your-bucket-name"
S3_PREFIX   = "multi-device-sweep"
SHOTS       = 256

# All gate-model QPUs available on AWS Braket (as of mid-2025)
# Comment out any you don't want to hit (cost / availability)
QPU_ARNS = [
    "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1",
    "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",      # IQM 20q
    "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",  # Rigetti 84q
]

# ── Circuit builders ──────────────────────────────────────────────────────────
def build_qft(n: int) -> Circuit:
    """n-qubit QFT (native Braket gates)."""
    circ = Circuit()
    for i in range(n):
        circ.h(i)
        for j in range(i + 1, n):
            circ.cphaseshift(j, i, np.pi / 2 ** (j - i))
    for i in range(n // 2):
        circ.swap(i, n - 1 - i)
    return circ

def build_ghz(n: int) -> Circuit:
    """n-qubit GHZ state: (|0...0> + |1...1>) / sqrt(2)."""
    circ = Circuit()
    circ.h(0)
    for i in range(n - 1):
        circ.cnot(i, i + 1)
    return circ

# ── Qubit counts: 1, half, max ────────────────────────────────────────────────
def qubit_sizes(max_q: int) -> list[int]:
    sizes = sorted(set([1, max(1, max_q // 2), max_q]))
    return sizes

# ── Main sweep ────────────────────────────────────────────────────────────────
submitted = []   # list of dicts tracking every task

for arn in QPU_ARNS:
    print(f"\n{'─'*60}")
    print(f"Device: {arn.split('/')[-1]}  ({arn})")

    # Query the device for its max qubit count
    try:
        device = AwsDevice(arn)
        caps   = device.properties

        # Pull qubit count — schema differs slightly by vendor
        if hasattr(caps, "paradigm"):
            max_qubits = caps.paradigm.qubitCount
        else:
            print("  Could not determine qubit count, skipping.")
            continue

        status = device.status
        print(f"  Status   : {status}")
        print(f"  Max qubits: {max_qubits}")

        if status != "ONLINE":
            print("  Device offline — skipping (tasks would queue indefinitely)")
            continue

    except Exception as e:
        print(f"  Error querying device: {e}")
        continue

    sizes = qubit_sizes(max_qubits)
    print(f"  Circuit sizes: {sizes}")

    for n in sizes:
        for circuit_type, builder in [("QFT", build_qft), ("GHZ", build_ghz)]:

            # GHZ only needs CNOT chain — fine on all topologies
            # QFT needs all-to-all for large n (warn for grid QPUs)
            if circuit_type == "QFT" and n > 10:
                vendor = arn.split("/qpu/")[1].split("/")[0]
                if vendor in ("rigetti", "iqm"):
                    print(f"  ⚠  {circuit_type} n={n} on {vendor}: "
                          f"grid topology → heavy SWAP overhead. Submitting anyway.")

            circ = builder(n)
            circ.probability()   # ask for marginal probabilities (not just bitstrings)

            try:
                task = device.run(
                    circ,
                    shots=SHOTS,
                    s3_destination_folder=(S3_BUCKET,
                                           f"{S3_PREFIX}/{arn.split('/')[-1]}")
                )
                submitted.append({
                    "device"  : arn.split("/")[-1],
                    "arn"     : arn,
                    "circuit" : circuit_type,
                    "n_qubits": n,
                    "task_id" : task.id,
                    "task_obj": task,         # keep reference for polling
                })
                print(f"  ✓ Submitted {circuit_type:3s} n={n:2d}  →  {task.id}")

            except Exception as e:
                print(f"  ✗ Failed   {circuit_type:3s} n={n:2d}  →  {e}")

print(f"\n\nTotal tasks submitted: {len(submitted)}")

# ── Save task IDs immediately (so you can recover if session dies) ─────────────
manifest = [{k: v for k, v in t.items() if k != "task_obj"} for t in submitted]
with open("task_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Task manifest saved to task_manifest.json")

# ── Poll results ──────────────────────────────────────────────────────────────
# Option A: block and wait for everything (fine for small sweeps)
print("\nWaiting for results...")
results = {}
for entry in submitted:
    task = entry["task_obj"]
    key  = f"{entry['device']}_{entry['circuit']}_n{entry['n_qubits']}"
    try:
        res = task.result()   # blocks until this specific task completes
        counts = res.measurement_counts
        results[key] = counts
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        print(f"  {key:40s}  top-3: {top}")
    except Exception as e:
        print(f"  {key:40s}  ERROR: {e}")
        results[key] = None

# Option B: non-blocking — check state and come back later
# for entry in submitted:
#     print(entry["task_id"], entry["task_obj"].state())

# ── Save results ──────────────────────────────────────────────────────────────
with open("sweep_results.json", "w") as f:
    json.dump({k: dict(v) if v else None for k, v in results.items()}, f, indent=2)
print("\nResults saved to sweep_results.json")