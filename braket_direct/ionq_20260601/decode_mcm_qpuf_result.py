#!/usr/bin/env python3
"""
decode_mcm_qpuf_result.py
==========================
Pure analysis of the one real completed two-stage MCM QPE-QPUF hardware
result (gate_limit_probe_results/0ede858e-d8ba-44a5-bdbc-14154b5aeeaa.json,
n_prec=3, n_targ=3, 100 shots, seed=10, target_init_seed=99). No hardware
access, no AWS calls -- this only reads the already-retrieved JSON.

THE CONVENTION CONFLICT (see CLAUDE.md known-bug item 4)
----------------------------------------------------------
The data file's own `_counts_convention` note says the counts came straight
from `AwsQuantumTask(arn).result().measurement_counts`, with NO
`qiskit_braket_provider` in the path. That is the RAW Braket convention:
one character per MEASURED qubit, ordered by ASCENDING physical qubit
index. With `measured_qubits=[3,4,5,6,7,8]` (q0-q2 = target, unmeasured):
    char 0 -> q3, char 1 -> q4, ..., char 5 -> q8.
Combined with `build_full_circuit`'s layout (q3-q5 = stage-1 precision ->
c[0..2] = m1, q6-q8 = stage-2 precision -> c[3..5] = m2) and the MSB-first
convention `build_qpe_stage` uses internally (precision qubit 0 carries the
highest power of U and is therefore the MSB of that stage's integer, see
that function's docstring), the decode is:

    m1 = int(b[0:3], 2)      m2 = int(b[3:6], 2)          ("convention B")

A separate physicist's pre-registration note assumed a *qiskit-style*
string (leftmost char = highest classical bit, `b = c5 c4 c3 c2 c1 c0`),
giving the exact reverse:

    m1 = int(b[5]+b[4]+b[3], 2)   m2 = int(b[2]+b[1]+b[0], 2)  ("convention A")

`qiskit_braket_provider._result_from_circuit_task` applies `{k[::-1]: v}`
when converting Braket counts to qiskit counts precisely because these two
conventions are exact reverses of each other -- and this run did NOT go
through that provider, so the physicist's note does not apply to this file.

DECISIVE, PROVENANCE-INDEPENDENT CHECK
---------------------------------------
`ideal_joint()` below re-derives the exact noiseless (m1, m2) joint
distribution from scratch (branch-exact statevector simulation of the two
QPE stages, not sampling) using the SAME seeds this hardware run used. It
reproduces the physicist's quoted marginal `[0.0625, 0.2479, 0.0188,
0.5696, 0.0328, 0.0313, 0.0316, 0.0055]` and `P(m1==m2) = 0.8791` to
machine precision, and confirms the ideal joint is exactly symmetric
(m1 <-> m2) -- an independent cross-check on the ideal baseline itself,
not just the decoding.

Decoding the hardware counts both ways and comparing to that ideal:
  - convention B's m1 marginal peaks at bin 3 -- exactly where the ideal
    predicts essentially all the weight (0.57) for BOTH stages. Stage 1 is
    unaffected by the MCM disturbance under investigation (it happens
    between the two stages), so a clean match there is expected and is not
    a coincidence you'd get from the wrong bit order.
  - convention A is related to B by relabeling each stage's integer under
    3-bit reversal and swapping which stage is "m1" (A_m1 = bitrev3(B_m2),
    A_m2 = bitrev3(B_m1) -- verified numerically). Under A, NEITHER
    marginal peaks at bin 3; the bin-3 weight that's really stage 1's
    signal gets scattered under a QFT integer relabeling that has no
    reason to preserve peak location.
  - convention B is therefore adopted as correct; convention A is reported
    as the rejected alternative for the record.

This script does not draw physics conclusions (no p_flip estimate, no
"supports/refutes" language) -- it produces the numbers and a plot for the
physicist/verifier to interpret.
"""

import argparse
import itertools
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(
    HERE, "gate_limit_probe_results", "0ede858e-d8ba-44a5-bdbc-14154b5aeeaa.json"
)
PLOTS_DIR = os.path.join(HERE, "gate_limit_probe_results", "plots")

N_PREC = 3
N_TARG = 3
SEED = 10
TARGET_INIT_SEED = 99
D = 2 ** N_PREC  # =8, both the QFT modulus and the circular-distance modulus


# =========================================================================
# Ideal (noiseless) joint distribution -- branch-exact, not sampled
# =========================================================================

def ideal_joint(n_prec: int = N_PREC, n_targ: int = N_TARG,
                seed: int = SEED, target_init_seed: int = TARGET_INIT_SEED,
                U: np.ndarray | None = None, eigenstate: bool = False
                ) -> np.ndarray:
    """
    Exact noiseless P(m1, m2) for build_full_circuit's two-stage QPUF.

    Sampling this with a stochastic simulator would only be approximate, so
    instead we simulate stage 1 to a Statevector, branch EXACTLY over the
    8 possible stage-1 outcomes (projection + renormalization on the
    precision-a subspace, done directly on the amplitude array -- Aer/qiskit
    have no built-in "give me the exact MCM-branch probabilities" call),
    then evolve each branch's residual state through stage 2 and read off
    its exact conditional distribution over m2.

    U=None (default) draws the original Haar U from `seed` -- this is
    load-bearing: the hardware-decode call site below (`ideal_joint()` with
    no args) must keep reproducing P(m1==m2)=0.8791 to machine precision.
    Passing U explicitly (e.g. an on-grid unitary) bypasses the Haar draw
    entirely; `seed` is then unused for U (target_init_seed still seeds the
    product-state init unless eigenstate=True).
    """
    import sys
    sys.path.insert(0, HERE)
    from submit_QPUF_ntarg import haar_random_unitary, build_qpe_stage
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Statevector

    if U is None:
        rng = np.random.default_rng(seed=seed)
        U = haar_random_unitary(2 ** n_targ, rng=rng)

    n_total = n_targ + 2 * n_prec
    targ_idx = list(range(0, n_targ))
    prec_a_idx = list(range(n_targ, n_targ + n_prec))
    prec_b_idx = list(range(n_targ + n_prec, n_targ + 2 * n_prec))

    q = QuantumRegister(n_total, "q")
    qc1 = QuantumCircuit(q)
    if eigenstate:
        for i in targ_idx:
            qc1.x(q[i])
    else:
        init_rng = np.random.default_rng(seed=target_init_seed)
        for i in targ_idx:
            qc1.ry(init_rng.uniform(0, np.pi), q[i])
            qc1.rz(init_rng.uniform(0, 2 * np.pi), q[i])
    qpe1 = build_qpe_stage(n_prec, U)
    qc1.append(qpe1, prec_a_idx + targ_idx)
    sv1 = Statevector.from_instruction(qc1)
    qpe2 = build_qpe_stage(n_prec, U)

    joint = np.zeros((2 ** n_prec, 2 ** n_prec))
    idx = np.arange(sv1.dim)
    for m1 in range(2 ** n_prec):
        # prec_a_idx[0] is the MSB of m1 (build_qpe_stage's MSB-first
        # convention), so unpack m1's bits MSB-first onto that qubit order.
        bits = [(m1 >> (n_prec - 1 - j)) & 1 for j in range(n_prec)]
        mask = np.ones_like(idx, dtype=bool)
        for j, qb in enumerate(prec_a_idx):
            mask &= ((idx >> qb) & 1) == bits[j]
        amp = np.where(mask, sv1.data, 0.0)
        prob_m1 = float(np.sum(np.abs(amp) ** 2))
        if prob_m1 < 1e-14:
            continue
        from qiskit.quantum_info import Statevector as SV
        proj = SV(amp / np.sqrt(prob_m1))
        sv2 = proj.evolve(qpe2, qargs=prec_b_idx + targ_idx)
        # probabilities(qargs=...) is little-endian in qargs order (qargs[0]
        # = LSB of the returned array's index), so reversing prec_b_idx
        # makes the array index equal m2 directly under the MSB-first
        # physics convention -- avoids a separate bit-reversal step.
        probs_m2 = sv2.probabilities(qargs=prec_b_idx[::-1])
        joint[m1, :] = prob_m1 * probs_m2

    return joint


# =========================================================================
# Hardware counts -> (m1, m2) joint, both candidate conventions
# =========================================================================

def decode_counts(counts: dict, convention: str) -> np.ndarray:
    """
    Re-key raw Braket measurement_counts (one char per measured qubit,
    ascending physical index; measured_qubits=[3,4,5,6,7,8]) into an 8x8
    integer count table over (m1, m2).

    convention:
      "raw_braket" (adopted) -- char0..char5 = q3..q8 directly; each 3-char
          half is MSB-first because build_qpe_stage assigns qubit 0 of a
          precision register the highest power of U.
      "physicist_note" (rejected alternative, kept for the appendix) --
          treats the string as if it had been through
          qiskit_braket_provider's character reversal, which this run's
          provenance shows it was NOT.
    """
    joint = np.zeros((D, D), dtype=int)
    for bits, n in counts.items():
        if convention == "raw_braket":
            m1 = int(bits[0:3], 2)
            m2 = int(bits[3:6], 2)
        elif convention == "physicist_note":
            m1 = int(bits[5] + bits[4] + bits[3], 2)
            m2 = int(bits[2] + bits[1] + bits[0], 2)
        else:
            raise ValueError(convention)
        joint[m1, m2] += n
    return joint


# =========================================================================
# Statistics
# =========================================================================

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 95% CI on a binomial proportion k/n."""
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def circular_distance_pmf(joint: np.ndarray, d: int = D) -> np.ndarray:
    """
    dist_d(m1,m2) = min(|m1-m2| mod d, d - (|m1-m2| mod d)) -- the paper's
    statistic, NOT the linear |m2-m1| the repo elsewhere reports. Returns
    counts indexed by distance 0..d//2.
    """
    n = joint.sum()
    # dtype matches the input: hardware call sites pass integer counts
    # (exact accumulation), but two_stage_on_grid_baseline.py also calls
    # this on ideal_joint's float probability matrix -- a fixed dtype=int
    # here would silently truncate every fractional addition to 0 before
    # it ever reaches 1, zeroing out the whole histogram.
    out = np.zeros(d // 2 + 1, dtype=joint.dtype)
    for m1 in range(d):
        for m2 in range(d):
            diff = abs(m1 - m2) % d
            dist = min(diff, d - diff)
            out[dist] += joint[m1, m2]
    assert np.isclose(out.sum(), n)
    return out


def verification_rate(dist_counts: np.ndarray, n: int, delta: int) -> float:
    """v_R(Delta) = P(dist_d(m1,m2) <= Delta) -- cumulative, per the paper."""
    return float(dist_counts[: delta + 1].sum() / n)


def guess_floor(delta: int, d: int = D) -> float:
    """Random-guess floor for v_R(Delta): (2*Delta+1)/d, for Delta < d/2."""
    return (2 * delta + 1) / d


# =========================================================================
# Reporting
# =========================================================================

def print_table(joint: np.ndarray, title: str):
    n = joint.sum()
    print(f"\n{title}  (raw counts, n={n})")
    header = "m1\\m2 " + "".join(f"{j:>5}" for j in range(D))
    print(header)
    for i in range(D):
        row = "".join(f"{joint[i, j]:>5}" for j in range(D))
        print(f"{i:>5} {row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    counts = data["measurement_counts"]
    n_shots = sum(counts.values())
    n_unique = len(counts)

    ideal = ideal_joint()
    ideal_diag = float(np.trace(ideal))
    ideal_symmetric_err = float(np.max(np.abs(ideal - ideal.T)))

    joint_B = decode_counts(counts, "raw_braket")
    joint_A = decode_counts(counts, "physicist_note")

    marg1_B = joint_B.sum(axis=1) / n_shots
    marg2_B = joint_B.sum(axis=0) / n_shots
    marg1_A = joint_A.sum(axis=1) / n_shots
    marg2_A = joint_A.sum(axis=0) / n_shots
    ideal_marg = ideal.sum(axis=1)

    print("=" * 78)
    print("CONVENTION DETERMINATION")
    print("=" * 78)
    print(f"Ideal marginal (re-derived, branch-exact statevector sim): "
          f"{np.round(ideal_marg, 4).tolist()}")
    print(f"Ideal joint symmetry |J - J^T|_max = {ideal_symmetric_err:.2e}  "
          f"(should be ~0)")
    print(f"Ideal P(m1==m2) = {ideal_diag:.4f}  "
          f"(physicist's pre-registered value: 0.8791)")
    print()
    print(f"convention B (raw Braket, adopted)   marg1={np.round(marg1_B, 3).tolist()}")
    print(f"                                      marg2={np.round(marg2_B, 3).tolist()}")
    print(f"  argmax(marg1)={int(np.argmax(marg1_B))}  argmax(marg2)={int(np.argmax(marg2_B))}"
          f"  (ideal argmax = {int(np.argmax(ideal_marg))})")
    print(f"convention A (physicist note, rejected) marg1={np.round(marg1_A, 3).tolist()}")
    print(f"                                         marg2={np.round(marg2_A, 3).tolist()}")
    print(f"  argmax(marg1)={int(np.argmax(marg1_A))}  argmax(marg2)={int(np.argmax(marg2_A))}"
          f"  (ideal argmax = {int(np.argmax(ideal_marg))})")
    print()
    print("-> Only convention B's stage-1 marginal (unaffected by the MCM "
          "disturbance, since\n   it is measured BEFORE the mid-circuit "
          "measurement occurs) lands its argmax on the\n   ideal-predicted "
          "bin. Convention B is adopted; convention A is the rejected\n"
          "   alternative, reported below for the record only.")

    print_table(joint_B, "PRIMARY: (m1, m2) joint, convention B (raw Braket, adopted)")
    print_table(joint_A, "APPENDIX: (m1, m2) joint, convention A (physicist note, REJECTED)")

    # ---- P(m1 == m2) ----------------------------------------------------
    k_diag = int(np.trace(joint_B))
    p_diag = k_diag / n_shots
    lo, hi = clopper_pearson(k_diag, n_shots)
    print("\n" + "=" * 78)
    print("P(m1 == m2)")
    print("=" * 78)
    print(f"hardware: {k_diag}/{n_shots} = {p_diag:.4f}   "
          f"95% Clopper-Pearson [{lo:.4f}, {hi:.4f}]")
    print(f"ideal baseline (this instance, NOT 1.0): {ideal_diag:.4f}")

    # ---- circular distance (paper's statistic) ---------------------------
    dist_counts = circular_distance_pmf(joint_B)
    dist_pmf = dist_counts / n_shots
    print("\n" + "=" * 78)
    print("Circular mod-8 distance dist_d(m1,m2) -- the paper's statistic")
    print("=" * 78)
    for dd in range(len(dist_counts)):
        print(f"  dist={dd}: count={dist_counts[dd]:>3}  pmf={dist_pmf[dd]:.4f}")
    print()
    for delta in (0, 1, 2):
        vr = verification_rate(dist_counts, n_shots, delta)
        floor = guess_floor(delta)
        print(f"  v_R(Delta<={delta}) = {vr:.4f}   random-guess floor = {floor:.4f}"
              f"   ({'above' if vr > floor else 'at/below'} floor)")

    lin_mean = float(sum(abs(m2 - m1) * joint_B[m1, m2]
                         for m1 in range(D) for m2 in range(D)) / n_shots)
    print(f"\n  (for reference only, NOT the paper's statistic) "
          f"mean linear |m2-m1| = {lin_mean:.4f}")

    # ---- unique outcomes --------------------------------------------------
    print("\n" + "=" * 78)
    print("Unique outcomes")
    print("=" * 78)
    print(f"  {n_unique} unique 6-bit outcomes observed in the file "
          f"(earlier relay said 44 -- miscount per the file's own "
          f"_provenance note; {n_unique} is what's actually in the data).")

    # ---- plot ---------------------------------------------------------------
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.subplots_adjust(top=0.80, wspace=0.3)

    p_hw = joint_B / n_shots
    vmax = max(p_hw.max(), ideal.max())

    for ax, mat, title in (
        (axes[0], p_hw, f"Observed (hardware, n={n_shots}, convention B)"),
        (axes[1], ideal, "Ideal (noiseless, branch-exact sim, same seeds)"),
    ):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
        ax.set_xlabel("m2")
        ax.set_ylabel("m1")
        ax.set_xticks(range(D))
        ax.set_yticks(range(D))
        ax.set_title(title, fontsize=10)
        # Mark the m1==m2 diagonal explicitly -- that's the QPUF
        # verification condition, not just a visual coincidence.
        ax.plot([0, D - 1], [0, D - 1], color="red", lw=1.2, ls="--", alpha=0.8)
        for i in range(D):
            for j in range(D):
                if mat[i, j] > 0.02:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if mat[i, j] > vmax * 0.6 else "black")

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="probability")
    fig.suptitle(
        f"Two-stage MCM QPE-QPUF, n_prec={N_PREC} n_targ={N_TARG}, "
        f"IonQ Forte-Enterprise-1, task 0ede858e (100 shots)\n"
        f"observed vs. ideal (m1,m2) joint  --  "
        f"P(m1==m2): hw={p_diag:.3f} vs ideal={ideal_diag:.4f}  "
        f"(red dashed = m1==m2 diagonal)",
        fontsize=10.5, y=0.98,
    )
    out_png = os.path.join(PLOTS_DIR, "mcm_qpuf_joint_observed_vs_ideal.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
