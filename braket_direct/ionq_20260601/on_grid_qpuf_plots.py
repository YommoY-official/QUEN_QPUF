#!/usr/bin/env python3
"""
on_grid_qpuf_plots.py
======================
Plot/decode helpers for on_grid_qpuf_pipeline.py, lifted from
`braket_direct_analysis.ipynb`'s cell 5 `plot_m1_vs_m2` (notebook itself is
untouched). NOISELESS-ONLY module: no AWS/Braket imports, no device calls.

`decode_bits` is NOT the notebook's `parse_outcome` fallback -- that
fallback (`int(bitstring[n_prec:], 2), int(bitstring[:n_prec], 2)`) does
not reverse each half, and this repo has been bitten by exactly this class
of bug before (see CLAUDE.md known-bug item 4 and
`decode_mcm_qpuf_result.py`'s convention-A-vs-B saga). Verified wrong for
`build_full_circuit`'s register layout by direct test in
`on_grid_qpuf_pipeline.py`'s `_verify_decode_known_answer` (it decodes the
known eigenstate answer (7,7) as (14,14) = bit-reversed). `decode_bits`
below is the corrected version and is the single source of truth for this
decode -- written once, not duplicated/guessed at a second time.
"""

import numpy as np
import matplotlib.pyplot as plt


def decode_bits(bitstring: str, n_prec: int) -> tuple[int, int]:
    """
    Decode one qiskit-convention measurement bitstring (single flat
    ClassicalRegister("c", 2*n_prec), as build_full_circuit emits) into
    (m1, m2), MSB-first per build_qpe_stage's convention.

    Qiskit/Aer order counts-dict keys with the LEFTMOST char = highest
    classical bit index (c[2*n_prec-1]), rightmost char = c[0] -- confirmed
    empirically (x(q0); measure(q0,c0) alone yields "...0001") and
    reconfirmed directly on AerSimulator for this exact register layout.
    build_full_circuit maps c[0..n_prec-1] = stage 1 (c[0] = MSB of m1) and
    c[n_prec..2n_prec-1] = stage 2 (c[n_prec] = MSB of m2). Index algebra:
    c[k] = b[2*n_prec-1-k], so the TRAILING n_prec chars of b, REVERSED,
    give c[0..n_prec-1] MSB-first = m1; the LEADING n_prec chars, reversed,
    give m2.
    """
    L = 2 * n_prec
    s1 = bitstring[n_prec:L]
    s2 = bitstring[0:n_prec]
    m1 = int(s1[::-1], 2)
    m2 = int(s2[::-1], 2)
    return m1, m2


def plot_m1_vs_m2(counts: dict, n_prec: int, eigen_bins, n_shots: int,
                  title_suffix: str = '', ax=None):
    """
    Scatter of (m1, m2) with dotted green lines at every ideal eigenbin --
    same style as the notebook's plot_m1_vs_m2, but decoding via the
    verified `decode_bits` above (not the notebook's unreversed
    `parse_outcome` fallback) and taking `eigen_bins` directly from the
    caller's already-known m_j list instead of re-eigendecomposing a
    diagonal matrix via np.linalg.eig on data whose eigenvalues are already
    known exactly.
    """
    m1_vals, m2_vals, weights = [], [], []
    for outcome, cnt in counts.items():
        m1, m2 = decode_bits(outcome, n_prec)
        m1_vals.append(m1); m2_vals.append(m2); weights.append(cnt)

    M = 2 ** n_prec
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    s = np.array(weights, dtype=float) * (200.0 / max(weights))
    ax.scatter(m1_vals, m2_vals, s=s, alpha=0.6)
    ax.plot([0, M - 1], [0, M - 1], 'r--', alpha=0.5, label='m1 = m2')
    for b in sorted(set(eigen_bins)):
        ax.axvline(b, color='green', linestyle=':', alpha=0.4)
        ax.axhline(b, color='green', linestyle=':', alpha=0.4)
    ax.set_xlim(-0.5, M - 0.5); ax.set_ylim(-0.5, M - 0.5)
    ax.set_xlabel('m1  (Stage 1 QPE)')
    ax.set_ylabel('m2  (Stage 2 QPE)')
    title = f'm1 vs m2 (n_prec={n_prec}, shots={n_shots})'
    if title_suffix:
        title += f'  -  {title_suffix}'
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    return ax
