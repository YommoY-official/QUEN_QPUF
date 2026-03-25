#!/usr/bin/env python3
"""
check_jobs.py
=============
Reads the 2 most recent jobs from jobs_log.jsonl, prints their AWS status,
and – if a job is COMPLETED – runs the full result analysis from the notebook.

Run this on the DCV where AWS credentials and the Braket SDK are available.
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from braket.aws import AwsQuantumTask

# ── Config ─────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), "jobs_log.jsonl")
DELTA    = 2    # cyclic-distance acceptance threshold
N_JOBS   = 2    # number of most-recent jobs to inspect

# ── Helpers (mirrored from notebook) ───────────────────────────────────────

def cyclic_distance(a: int, b: int, n_prec: int) -> int:
    M    = 2 ** n_prec
    diff = abs(a - b)
    return min(diff, M - diff)


def parse_outcome(bitstring: str, n_prec: int):
    """
    Parse a Qiskit counts key produced by qiskit_braket_provider.
    Key format: 'c2_bits c1_bits'  (space-separated, c2 printed first).
    Returns (m1, m2) as integers.
    """
    parts = bitstring.split(' ')
    return int(parts[1], 2), int(parts[0], 2)


def analyse_counts(counts: dict, n_prec: int, delta: int, label: str = ''):
    total    = sum(counts.values())
    accepted = sum(
        cnt for outcome, cnt in counts.items()
        if cyclic_distance(*parse_outcome(outcome, n_prec), n_prec) <= delta
    )
    acc_rate = accepted / total

    prefix = f'[{label}] ' if label else ''
    print(f'{prefix}Total shots        : {total}')
    print(f'{prefix}Accepted (dist ≤ {delta}) : {accepted}')
    print(f'{prefix}Acceptance rate    : {acc_rate:.4f}')

    print(f'\n{prefix}Top 10 outcomes:')
    print(f'  {"bitstring":28s}  m1  m2  dist  count')
    print(f'  {"-"*52}')
    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        m1, m2 = parse_outcome(k, n_prec)
        print(f'  {k!r:28s}  {m1:3d}  {m2:3d}  {cyclic_distance(m1, m2, n_prec):4d}  {v}')

    return total, accepted, acc_rate


def angles_to_unitary(angles: dict) -> np.ndarray:
    phi, theta, lam = angles['phi'], angles['theta'], angles['lam']
    def Rz(a): return np.array([[np.exp(-1j*a/2), 0], [0, np.exp(1j*a/2)]])
    def Ry(a): return np.array([[np.cos(a/2), -np.sin(a/2)], [np.sin(a/2), np.cos(a/2)]])
    return Rz(phi) @ Ry(theta) @ Rz(lam)


def print_eigenvalue_bins(angles: dict, n_prec: int):
    unitary = angles_to_unitary(angles)
    print('Theoretical QPE bins (ideal):')
    for ev in np.linalg.eigvals(unitary):
        phase = np.angle(ev) / (2 * np.pi)
        if phase < 0:
            phase += 1
        print(f'  phase={phase:.4f}  ideal bin={round(phase * 2**n_prec) % 2**n_prec}')


def plot_m1_vs_m2(counts: dict, n_prec: int, n_shots: int, title_suffix: str = ''):
    m1_vals, m2_vals, weights = [], [], []
    for outcome, count in counts.items():
        m1, m2 = parse_outcome(outcome, n_prec)
        m1_vals.append(m1)
        m2_vals.append(m2)
        weights.append(count)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(m1_vals, m2_vals, s=np.array(weights) * 2.0, alpha=0.6)
    ax.plot([0, 2**n_prec - 1], [0, 2**n_prec - 1], 'r--', label='m1 = m2')
    ax.set_xlabel('m1  (Stage 1 QPE)')
    ax.set_ylabel('m2  (Stage 2 QPE)')
    title = f'Phase estimates – 1-qubit Haar unitary\n(n_prec={n_prec}, n_shots={n_shots})'
    if title_suffix:
        title += f'  {title_suffix}'
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plot_m1_vs_m2_{title_suffix.replace(" ", "_").replace("[","").replace("]","")}.png')
    plt.show()


def plot_acceptance_vs_delta(counts: dict, n_prec: int, n_shots: int, delta: int,
                              title_suffix: str = ''):
    total       = sum(counts.values())
    delta_range = list(range(0, 2**n_prec + 1))
    acc_rates   = []

    for d in delta_range:
        acc = sum(
            cnt for outcome, cnt in counts.items()
            if cyclic_distance(*parse_outcome(outcome, n_prec), n_prec) <= d
        )
        acc_rates.append(acc / total)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(delta_range, acc_rates, marker='o', markersize=4)
    ax.axvline(delta, color='r', linestyle='--', label=f'current delta={delta}')
    ax.set_xlabel('delta  (cyclic distance threshold)')
    ax.set_ylabel('acceptance rate')
    ax.set_title(f'Acceptance rate vs delta  (n_prec={n_prec}, n_shots={n_shots})  {title_suffix}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plot_acc_vs_delta_{title_suffix.replace(" ", "_").replace("[","").replace("]","")}.png')
    plt.show()

    print('\ndelta → acceptance rate')
    for d, r in zip(delta_range, acc_rates):
        print(f'  delta={d:3d}: {r:.4f}')


def retrieve_counts_via_braket_provider(job_id: str, device_name: str) -> dict:
    """Use qiskit_braket_provider to get Qiskit-format counts dict."""
    try:
        from qiskit_braket_provider import BraketProvider
    except ImportError:
        print('ERROR: qiskit-braket-provider not installed (pip install qiskit-braket-provider)')
        return None

    provider    = BraketProvider()
    backend     = provider.get_backend(device_name)
    job_hw      = backend.retrieve_job(job_id)
    counts      = job_hw.result().get_counts()
    return counts


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # Load job log
    if not os.path.exists(LOG_FILE):
        print(f'ERROR: Job log not found: {LOG_FILE}')
        print('Run submit_1qubit_qpu.py first.')
        sys.exit(1)

    with open(LOG_FILE) as f:
        job_records = [json.loads(line) for line in f if line.strip()]

    if not job_records:
        print('ERROR: jobs_log.jsonl is empty.')
        sys.exit(1)

    # Take the N_JOBS most recent
    recent = job_records[-N_JOBS:]
    print(f'Checking {len(recent)} most recent job(s) from {LOG_FILE}\n')
    print('=' * 70)

    results_summary = []

    for i, rec in enumerate(recent):
        job_id      = rec['job_id']
        device_name = rec['device']
        n_prec      = rec['n_prec']
        n_shots     = rec['n_shots']
        angles      = rec['angles']
        submitted   = rec['datetime']

        label = f"Job {len(job_records) - len(recent) + i + 1} of {len(job_records)}"
        print(f'\n--- {label} ---')
        print(f'  Submitted : {submitted}')
        print(f'  Device    : {device_name}')
        print(f'  n_prec    : {n_prec}    n_shots : {n_shots}')
        print(f'  Job ID    : {job_id}')
        print(f'  Angles    : phi={angles["phi"]:.6f}  '
              f'theta={angles["theta"]:.6f}  lam={angles["lam"]:.6f}')

        # ── Check status via AwsQuantumTask ────────────────────────────────
        try:
            task   = AwsQuantumTask(arn=job_id)
            status = task.state()
        except Exception as e:
            print(f'\n  ERROR retrieving task status: {e}')
            results_summary.append((label, 'ERROR', None))
            continue

        print(f'\n  Status : {status}')

        if status != 'COMPLETED':
            print(f'\n  Job is not completed (status={status}). '
                  f'Results are not yet available.')
            results_summary.append((label, status, None))
            continue

        # ── Retrieve and analyse results ────────────────────────────────────
        print('\n  Job is COMPLETED — retrieving results …')
        counts = retrieve_counts_via_braket_provider(job_id, device_name)

        if counts is None:
            results_summary.append((label, 'COMPLETED (retrieval failed)', None))
            continue

        n_retrieved = sum(counts.values())
        print(f'  Retrieved {n_retrieved} shots with {len(counts)} unique outcomes.\n')

        tag = f'[{device_name} hardware | {label}]'

        # Analysis
        total_hw, accepted_hw, acc_rate_hw = analyse_counts(
            counts, n_prec, DELTA, label=f'{device_name} hardware'
        )
        print()
        print_eigenvalue_bins(angles, n_prec)

        # Plots
        plot_m1_vs_m2(counts, n_prec, n_shots, title_suffix=tag)
        plot_acceptance_vs_delta(counts, n_prec, n_shots, DELTA, title_suffix=tag)

        results_summary.append((label, 'COMPLETED', acc_rate_hw))

    # ── Summary ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('SUMMARY')
    print(f'  {"Job":30s}  {"Status":20s}  {"Acceptance rate":>16s}')
    print(f'  {"-"*68}')
    for lbl, st, acc in results_summary:
        acc_str = f'{acc:.4f}' if acc is not None else 'N/A'
        print(f'  {lbl:30s}  {st:20s}  {acc_str:>16s}')


if __name__ == '__main__':
    main()