#!/usr/bin/env python3
"""
submit_test.py
==============
Cost calculator / proxy for submit_QPUF_ntarg.py.

Prompts for N_PREC, N_TARG and N_SHOTS at runtime, builds the same two-stage
QPUF circuit that submit_QPUF_ntarg.py would submit (two disjoint precision
blocks — no `reset`), transpiles it against the IonQ basis, and reports the
gate counts, circuit depth and an estimated QPU runtime for that
configuration. Nothing is submitted to the QPU.

This lets you preview "what would this N_PREC / N_TARG / N_SHOTS cost?"
without touching submit_QPUF_ntarg.py's production constants — they are no
longer imported here.

The gate-time model (ONE_QUBIT_TIME_S etc.) and circuit builders are still
imported from submit_QPUF_ntarg.py so the estimate matches what the real
submission script reports.
"""

import sys

import numpy as np
from qiskit import transpile

from submit_QPUF_ntarg import (
    DEVICE_NAME,
    DEVICE_ARN,
    RES_ARN,
    SEED,
    TARGET_INIT_SEED,
    ONE_QUBIT_TIME_S,
    TWO_QUBIT_TIME_S,
    READOUT_TIME_S,
    STARTUP_TIME_S,
    haar_random_unitary,
    build_full_circuit,
)

DEVICE_QUBIT_BUDGET = 35   # Forte-Enterprise-1 qubit count


def _prompt_int(label: str) -> int:
    """Prompt for a strictly-positive integer, re-prompting on bad input."""
    while True:
        try:
            raw = input(f"{label}: ").strip()
        except EOFError:
            print("\nNo input — aborting.")
            sys.exit(1)
        try:
            val = int(raw)
            if val <= 0:
                raise ValueError("must be positive")
            return val
        except ValueError as e:
            print(f"  Invalid value ({e}); enter a positive integer.")


def prompt_inputs() -> tuple[int, int, int]:
    """Ask for N_PREC, N_TARG, N_SHOTS interactively."""
    print("=" * 72)
    print("QPUF COST CALCULATOR (proxy for submit_QPUF_ntarg.py)")
    print("=" * 72)
    print("Enter the configuration to estimate (no job is submitted):\n")
    n_prec  = _prompt_int("N_PREC  (precision qubits per stage)")
    n_targ  = _prompt_int("N_TARG  (target qubits)")
    n_shots = _prompt_int("N_SHOTS (shots)")
    return n_prec, n_targ, n_shots


def _fmt_time(t_s: float) -> str:
    return f"{t_s:.1f} s  ({t_s/60:.2f} min, {t_s/3600:.3f} h)"


def cost_calculator(n_prec: int, n_targ: int, n_shots: int) -> None:
    """Build + transpile the circuit for the given config and report cost."""
    n_total = n_targ + 2 * n_prec

    print("\n" + "=" * 72)
    print("CONFIGURATION")
    print("=" * 72)
    print(f"DEVICE      : {DEVICE_NAME}")
    print(f"RES_ARN     : {RES_ARN}")
    print(f"N_PREC      : {n_prec}  (x2 disjoint blocks — no reset)")
    print(f"N_TARG      : {n_targ}")
    print(f"N_SHOTS     : {n_shots}")
    print(f"Total qubits: {n_total}  (targ + prec_a + prec_b)")
    if n_total > DEVICE_QUBIT_BUDGET:
        print(f"  WARNING: {n_total} qubits EXCEEDS the {DEVICE_NAME} budget "
              f"of {DEVICE_QUBIT_BUDGET} — this config could not be submitted.")

    # Haar-random U for the target register (seed fixed; the transpiled 2q
    # count is governed by N_TARG, essentially independent of the draw).
    rng = np.random.default_rng(seed=SEED)
    d = 2 ** n_targ
    U = haar_random_unitary(d, rng=rng)

    qc = build_full_circuit(n_prec, n_targ, U, target_init_seed=TARGET_INIT_SEED)
    print(f"Logical ops : {qc.size()}")

    # rz/rx/rxx ≈ IonQ MS-basis; Braket Direct's server-side compiler maps
    # these to native gpi/gpi2/zz at submission. Gate/depth counts here are a
    # proxy for the native cost.
    ionq_basis = ['rz', 'rx', 'rxx', 'measure', 'reset']

    print("\n--- Full circuit transpile (this may take a moment) ---")
    qc_hw  = transpile(qc, basis_gates=ionq_basis, optimization_level=1)
    ops    = qc_hw.count_ops()
    n_1q   = ops.get('rz', 0) + ops.get('rx', 0)
    n_2q   = ops.get('rxx', 0)
    n_meas = ops.get('measure', 0)
    n_gates = qc_hw.size()

    print(f"Transpiled depth   : {qc_hw.depth()}")
    print(f"Transpiled gates   : {n_gates}")
    print(f"  1q gates (rz+rx) : {n_1q}")
    print(f"  2q gates (rxx)   : {n_2q}")
    print(f"  measurements     : {n_meas}")
    print(f"Op breakdown       : {dict(ops)}")

    print("\n--- Totals across all shots ---")
    print(f"Total gates        : {n_gates * n_shots}")
    print(f"Total 2-qubit gates: {n_2q * n_shots}")

    # ── Estimated runtime (IonQ Forte-1 gate-time model) ──────────────────────
    # (a) literal  — readout charged once per shot, as written in the formula
    # (b) per-meas — readout charged per measurement event in the circuit
    per_shot_gates  = n_1q * ONE_QUBIT_TIME_S + n_2q * TWO_QUBIT_TIME_S
    t_literal_s     = STARTUP_TIME_S + n_shots * (per_shot_gates + READOUT_TIME_S)
    t_permeas_s     = STARTUP_TIME_S + n_shots * (per_shot_gates + n_meas * READOUT_TIME_S)

    print("\n--- Estimated runtime ---")
    print(f"Literal readout/shot     : {_fmt_time(t_literal_s)}")
    print(f"Readout x n_measurements : {_fmt_time(t_permeas_s)}")


def main():
    n_prec, n_targ, n_shots = prompt_inputs()
    cost_calculator(n_prec, n_targ, n_shots)


if __name__ == "__main__":
    main()