#!/usr/bin/env python3
"""
rigetti_cepheus_QPE.py
======================
Submit a 51-qubit QPE circuit to Rigetti Cepheus-1-108Q during a Braket
reservation window.

Circuit
-------
  Algorithm        : Quantum Phase Estimation (QPE)
  Total qubits     : 51   (50 counting + 1 target)
  Target unitary   : T gate           (eigenvalue e^{2πi · 1/8})
  Eigenstate       : |1⟩  on the target qubit
  Phase φ          : 1/8
  Ideal peak bin   : 2^50 / 8 = 140737488355328
  Shots            : 2000
  Device           : Rigetti Cepheus-1-108Q  (us-west-1)
  Gate set         : H, X, Rz, CNOT          (CPhaseShift decomposed)
  Convention       : qubit 0 of the counting register is the MSB of the
                     measured integer.

Before running, fill in:
  - RESERVATION_ARN : Braket reservation ARN for the booked window
  - S3_BUCKET       : destination bucket for task results
"""

import sys
import numpy as np

from braket.aws import AwsDevice
from braket.circuits import Circuit


# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE_ARN      = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
N_TOTAL         = 51
N_COUNTING      = N_TOTAL - 1          # 50
SHOTS           = 2000
PHASE_TARGET    = 1 / 8                # T gate

RESERVATION_ARN = ""                   # ← fill in before running
S3_BUCKET       = ""                   # ← fill in before running
S3_PREFIX       = "qpu-benchmark/Cepheus-1-108Q"


# ── Gate decompositions ───────────────────────────────────────────────────────
def _cphase(circ: Circuit, control: int, target: int, angle: float):
    """CPhaseShift(angle) decomposed into Rz + CNOT (correct up to global phase)."""
    circ.rz(control, angle / 2)
    circ.cnot(control, target)
    circ.rz(target, -angle / 2)
    circ.cnot(control, target)
    circ.rz(target, angle / 2)


# ── Circuit builder ───────────────────────────────────────────────────────────
def build_qpe(n_counting: int) -> Circuit:
    """
    Textbook QPE for U = T (φ = 1/8), eigenstate |1⟩.

    Layout:
      counting register : qubits 0 .. n_counting-1   (qubit 0 = MSB)
      target qubit      : qubit  n_counting           (initialized to |1⟩)

    Reads the measured bitstring as an integer m with qubit 0 leftmost;
    m / 2^n_counting estimates φ. For φ = 1/8 the ideal peak bin is
    2^n_counting / 8 (exact for n_counting ≥ 3).
    """
    target   = n_counting
    counting = list(range(n_counting))

    circ = Circuit()

    # Eigenstate |1⟩ on target
    circ.x(target)

    # Hadamard on all counting qubits
    for k in counting:
        circ.h(k)

    # Controlled-T^(2^(n_counting-1-k)): qubit 0 (MSB) drives the largest power
    for k in range(n_counting):
        power = 2 ** (n_counting - 1 - k)
        angle = (np.pi / 4) * power % (2 * np.pi)
        _cphase(circ, counting[k], target, angle)

    # Inverse QFT: SWAPs first, then reverse cascade of CR† and H
    for i in range(n_counting // 2):
        circ.swap(counting[i], counting[n_counting - 1 - i])

    for i in range(n_counting - 1, -1, -1):
        for j in range(n_counting - 1, i, -1):
            angle = -np.pi / 2 ** (j - i)
            _cphase(circ, counting[j], counting[i], angle)
        circ.h(counting[i])

    return circ


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not RESERVATION_ARN:
        print("ERROR: RESERVATION_ARN is empty — set it at the top of this file.")
        sys.exit(1)
    if not S3_BUCKET:
        print("ERROR: S3_BUCKET is empty — set it at the top of this file.")
        sys.exit(1)

    print(f"Building QPE circuit: n_total={N_TOTAL}, n_counting={N_COUNTING}, "
          f"shots={SHOTS}, phase={PHASE_TARGET}")
    circ = build_qpe(N_COUNTING)
    print(f"  depth={circ.depth}  gates={len(circ.instructions)}")

    device = AwsDevice(DEVICE_ARN)
    print(f"Submitting to {DEVICE_ARN} ...")

    task = device.run(
        circ,
        shots=SHOTS,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX),
        reservation_arn=RESERVATION_ARN,
    )

    print(f"  Task ID : {task.id}")


if __name__ == "__main__":
    main()
