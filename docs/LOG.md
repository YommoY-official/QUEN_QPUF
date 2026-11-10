# QUEN_QPUF Change/Discussion Log

Chronological, append-only. Each entry: date, agent/session, what changed, why, files, verification.

---

## 2026-08-28 — Worker2 (coder/verifier/physicist chain)

**What**: Fixed the IonQ inverse-QFT endianness bug (CLAUDE.md known-bug list
item 4) — the controlled-power loop builds the precision register MSB-first
but qiskit's `QFTGate` is little-endian, so the inverse QFT must be appended
to `prec[::-1]`, not `prec`. This mirrors the fix already verified on the
Rigetti side (`rigetti_qpuf_common.py:166`).

**Why**: Bug was confirmed present and unpatched on the IonQ side via
read-only exploration by the director session; left unfixed it silently
smears QPE's sharp phase across many bins instead of resolving it into one.

**Files**:
- `braket_direct/ionq_20260601/submit_QPUF_ntarg.py` — `build_qpe_stage`,
  `qc.append(iqft, prec)` → `qc.append(iqft, prec[::-1])`.
- `noise_mitigation/ionq_noise_mitig.py` — `build_qpe_stage`,
  `qc.append(QFTGate(n_prec).inverse(), prec)` → `... , prec[::-1])`.
- `noise_mitigation/ionq_qpuf_two_stage.py` — not touched, inherits the fix
  via import of `build_qpuf_two_stage` from `ionq_noise_mitig.py`.

**Verification**: Known-answer test on noiseless `Statevector`
(φ=1/8, n_prec=5, exact eigenstate): pre-fix p(ideal bin)=0.18 (smeared
across bins matching MITIGATION_BRIEF.md §5.1's Rigetti numbers); post-fix
p(ideal bin)=1.0000 exact, run against the actual imported patched
functions. Full CLAUDE.md known-bug regression checklist (a)-(d) re-checked,
all pass. No compensating double-reversal found elsewhere in the codebase.

**Physics impact (negative finding)**: All 13 previously-retrieved real
Forte-Enterprise-1 results (9 in `noise_mitigation/job_results/`, 4 in
`noise_mitigation/job_results_2stage/`) were taken under the unpatched bug
and are unreliable for phase-resolution or two-stage agreement claims — the
bug, not physical noise, is the dominant contributor to observed deviation
from ideal (83-91% of the agreement shortfall at the two-stage sites
tested). The currently quoted p_flip ≈ 0.2-0.3 estimate was computed against
a buggy simulator baseline; against the corrected baseline the same hardware
data implies p_flip ≈ 0.43-0.60 instead — a 3-4x change, and this is a
direct hit on the paper's central finding under investigation. Separately
(not fixed by this patch): `build_qpuf_two_stage` in `ionq_noise_mitig.py`
has no mid-circuit measurement at all (deferred-measurement identity), so
`job_results_2stage/` cannot test the MCM eigenvector-switching hypothesis
even in principle — only `build_full_circuit` in `submit_QPUF_ntarg.py` (the
9 never-retrieved orphaned jobs) has a genuine MCM between stages.
Recommendation: retake data under the patched code using the MCM-variant
circuits, scored at Δ≥2 mod-d agreement per Ghosh et al. Theorem 4.

No hardware was submitted or retrieved as part of this task.

---

## 2026-09-08 — Coder/Verifier (on-grid noiseless two-stage QPE-QPUF baseline)

**What**: Derived and verified why the noiseless two-stage QPE-QPUF ideal
P(m1==m2) is not 1.0 for Haar-random controlled-U (TODO.md's "typical
0.673" finding from 2026-08-28), then constructed and verified a new
on-grid U variant for which it is exactly 1.0, as a clean noiseless
baseline for future hardware MCM-disturbance testing.

**Derivation**: Noiselessly, the two-stage joint distribution factorizes
as a mixture over eigenvectors of U:

P(m1, m2) = Σ_j |c_j|² P_j(m1) P_j(m2)

where c_j is the target state's amplitude on U's j-th eigenvector and
P_j(·) is the single-stage QPE outcome distribution conditioned on
starting in eigenvector j. In this picture, the two-stage protocol is
equivalent to "pick eigenvector j with probability |c_j|², then run two
*independent* QPE draws from P_j" — the stage-1 mid-circuit measurement
does project the target onto (a reweighting toward) an eigenvector, but
each stage's own QPE outcome is still a draw from P_j, which is a delta
function only if eigenphase j sits exactly on the n_prec-bit grid.
Consequently: (1) it is off-grid Haar eigenphases, not target-state
superposition, that cause m2 ≠ m1 even with zero noise and even for a
target initialized exactly on an eigenstate; (2) increasing n_prec does
**not** fix this for generic Haar U — a representative half-a-bin-off-grid
eigenstate has two-stage collision probability C_n = (1/3)(1 + 2·4^-n),
which *decreases* toward 1/3 as n_prec grows rather than approaching 1.

**On-grid construction**: If every eigenphase of U lies exactly on the
n_prec-bit grid, P(m1==m2) = 1 exactly for *any* target state, not only an
eigenstate — this removes the off-grid-leakage ambiguity as a confound for
any future MCM-disturbance measurement. Built as `on_grid_unitary()`: U
diagonal in the computational basis, U[j,j] = exp(2πi · ((Σ_{i: bit i of j
set} k_i) mod 2^n_prec) / 2^n_prec) for integer k_i ∈ [0, 2^n_prec) per
target qubit (k not all zero and not all equal — the all-equal case
degenerates to a global phase, making every basis state a degenerate
eigenstate and destroying the eigenstate structure the MCM-disturbance
test needs, even though it would still give P=1). Equivalent to
U = ⊗_i P(2π k_i / 2^n_prec). A second, `rotated_on_grid_unitary()`
variant conjugates the same diagonal D by a fresh Haar-random V
(U = V D V†) to keep eigenphases on-grid while making the eigenbasis
generic instead of coincident with the measurement basis — needed because
a diagonal U's eigenbasis *is* the computational basis, which would make
the coherent-rotation MCM-disturbance effect (this project's central
hardware investigation) unobservable/degenerate.

**Files**:
- `braket_direct/ionq_20260601/submit_QPUF_ntarg.py` — added
  `on_grid_unitary()`, `rotated_on_grid_unitary()`, and an
  `eigenstate: bool = False` kwarg on `build_full_circuit` (X-gate
  |1...1⟩ target instead of the seeded RY/RZ product-state init). Also
  fixed the docstring mislabel flagged in TODO.md §3 ("Haar-random initial
  state" → random product state).
- `braket_direct/ionq_20260601/decode_mcm_qpuf_result.py` —
  `ideal_joint()` generalized with optional `U`/`eigenstate` kwargs
  (default Haar-random behavior verified unchanged); `circular_distance_pmf`
  dtype/assert generalized to accept float probability input
  (verified behavior-preserving on integer inputs).
- New `braket_direct/ionq_20260601/two_stage_on_grid_baseline.py`
  (noiseless-only, verified zero AWS/Braket imports via grep) — builds the
  on-grid instances, reports P(m1==m2)/marginals, runs a sampled MCM
  check, reports transpiled gate counts; `--regression` and `--plot` flags.

**Verification** (independently re-derived by the Verifier, not taken from
the Coder's report):
- All 8 combinations of {diagonal U, rotated U} × {eigenstate target
  |1...1⟩, RY/RZ product target} × {primary instance k=(1,2,4), n_prec=4,
  n_targ=3; secondary seeded random-k instance} give P(m1==m2) = 1.0 to
  machine precision (1 − P(m1==m2) between 0.0 and 1.1e-16 across all 8).
- Primary instance, eigenstate target, diagonal U: exact delta at
  m1=m2=7 = (1+2+4) mod 16, confirmed by an independent numpy
  recomputation of the eigenphase-binned population (not via the circuit),
  matching `ideal_joint`'s marginal to 6.4e-16 max abs difference.
- Sampled check (Statevector-based sampler; qiskit_aer not installed in
  this environment, so a standalone test was run first to confirm the
  sampler actually discriminates MCM-collapse-honored [agreement→1.0]
  from not-honored [agreement→~0.5] before trusting it): exactly
  4000/4000 shots agreeing (m1==m2) for both eigenstate and product
  targets on the primary diagonal-U instance — zero disagreements.
- Regression preserved: default `ideal_joint()` (Haar U, seed=10,
  target_init_seed=99, n_prec=n_targ=3) still gives
  P(m1==m2) = 0.8790997468647305 (0.8791 to 4 d.p.), joint symmetric to
  1.4e-16. `decode_mcm_qpuf_result.py` run standalone reproduces its prior
  output exactly (ideal marginal
  [0.0625, 0.2479, 0.0188, 0.5696, 0.0328, 0.0313, 0.0316, 0.0055],
  hardware 25/100 = 0.2500).
- CLAUDE.md known-bug regression list items (1)-(4) all re-confirmed
  still-patched / not reintroduced on the modified builder.

**Gate-count finding (open constraint, not yet resolved)**: transpiled
gate counts (n_targ=3, IonQ native basis) against the empirically-confirmed
2,000-gate-per-circuit IonQ Forte-1 hard limit (`probe_gate_limit.py`,
2026-08-28 session): n_prec=3 diagonal U 395 gates (under limit), rotated
U 4,546 gates (2.3× **over**); n_prec=4 diagonal U 561 gates (under),
rotated U 6,104 gates (3× **over**). The rotated-U variant as currently
constructed (dense Haar V synthesized via generic `UnitaryGate` /
`QuantumCircuit.unitary` decomposition) is not directly hardware-feasible
at these register sizes on Forte-1 within the known gate limit. This is
recorded as an open design constraint for the Physicist/Director to
resolve, not a settled outcome.

**Plot**: `braket_direct/ionq_20260601/gate_limit_probe_results/plots/on_grid_two_stage_ideal_joint.png`
— 3-panel heatmap of P(m1,m2): (i) on-grid k=(1,2,4), n_prec=4 diagonal U,
eigenstate target — single delta spike at (7,7), P(m1==m2)=1.0000; (ii)
same instance, product-state target — mass spread across 8 diagonal bins
only (one per computational-basis component), P(m1==m2)=1.0000; (iii) the
Haar seed=10, n_prec=3 regression instance, shown for contrast — visible
off-diagonal leakage, P(m1==m2)=0.8791. Caption states the MSB-first bit
convention (m = Σ_k c[k]·2^(n_prec-1-k), c[0] = MSB).

**Correction to record**: `qiskit_QPE_experiment.ipynb`'s "Notes /
Interpretation" markdown cell (the cell stating "Both estimates therefore
track the same eigenvalue → `m1 ≈ m2` on every shot") is contradicted by
the mixture derivation above — for the Haar-random U that notebook
actually uses, off-grid eigenphases make exact m1≈m2 agreement on every
shot false in general (that notebook's own delta-sweep cell already shows
the acceptance rate below 1.0 at delta=0, consistent with this). The
notebook itself was **not edited** as part of this task; this is recorded
here as a documentation correction to apply in a future pass.

No hardware was submitted or retrieved as part of this task.

**Addendum (follow-up)**: Investigated an apparent rotated-U vs. Haar
gate-count discrepancy — resolved as not a bug. Haar seed=10 (4,562 gates)
and the rotated on-grid variant (4,546 gates) are within 0.35% of each
other under the identical transpile path at n_prec=3, n_targ=3, as expected
for two generic dense 8×8 unitaries. The real finding: `probe_gate_limit.py`'s
docstring-quoted gate-count figures (1,915/2,583/5,263 gates, etc.) are
**stale** relative to the currently installed `qiskit==2.0.0` in this
environment — re-running that script's own `build_and_transpile()` right
now, unmodified, gives 4,562 gates at n_prec=3 (not 1,915) and 12,342 at
n_prec=8 (not 5,263); a historically-logged real hardware submission at
n_prec=10 (6,659 gates at submission time) also no longer reproduces
(15,494 gates now). **(Superseded — see Addendum 3 below: this paragraph's
framing turned out to be backwards. The docstring's 1,915-gate figure was
independently reproduced bit-for-bit under the braket-arm environment; it
was this session's local `qiskit==2.0.0` that was the outlier, not the
docstring.)** This looks like `qiskit`-version-dependent drift in
`UnitaryGate.control()` synthesis, not a defect introduced by this task's
changes. Per CLAUDE.md's ground-truth principle, this is flagged here as an
open discrepancy for a possible `probe_gate_limit.py` documentation
correction pass — not resolved in this task, and importantly: the
hardware-confirmed 2,000-gate-per-circuit IonQ limit itself is unaffected
(that was an empirical result from real device submissions, not a
locally-computed estimate), only the *local pre-submission gate-count
projections* used to pick candidate (n_prec, n_targ) values before
submitting are now unreliable in this environment.

Added `product_rotated_on_grid_unitary()`:
`U = ⊗_i (v_i · P(2πk_i/2^n_prec) · v_i†)`, a tensor product of per-qubit
terms (v_i independent Haar 2×2 unitaries) rather than a dense V·D·V†.
On-grid eigenphases confirmed to ~1.6e-15; P(m1==m2)=1.0000 confirmed for
both the ordinary RY/RZ product-state target and the V|1...1⟩ eigenstate
target (the latter verified via a standalone branch-exact computation,
since |1...1⟩ itself is not an eigenvector of this variant — only the
diagonal variant has that property). Gate counts (n_targ=3, local
qiskit==2.0.0 figures — superseded, see Addendum 3 below for reconciled
braket-arm numbers): 1,676 / 2,392 / 3,134 at n_prec=3/4/5 — cheaper than
the dense-rotated variant (4,546/6,104/7,654) but still crosses the
2,000-gate limit at n_prec≥4,
because `build_qpe_stage` synthesizes every controlled-U^p as one
monolithic `UnitaryGate.control()` regardless of the matrix's latent
tensor-product structure; realizing the full expected savings (n_targ
independent controlled-single-qubit gates per power) would need an
explicit per-qubit decomposition at the circuit-construction level, not
attempted here.

Added a soft degeneracy warning: `_check_subset_sums_distinct` (via
`warnings.warn`, not a hard error) fires when the 2^n_targ subset sums of a
k-vector aren't all distinct mod 2^n_prec — collisions are legal
(P(m1==m2)=1 still holds; they just mean some eigenspaces are
2+-dimensional). Confirmed it fires on a colliding example and stays silent
on the primary k=(1,2,4) instance. Wired into `on_grid_unitary`,
`rotated_on_grid_unitary`, and `product_rotated_on_grid_unitary`; the
existing hard not-all-zero/not-all-equal `ValueError` checks are unchanged.

**Addendum 2 (follow-up)**: Added an additive factorized controlled-U
QPE-stage builder, `build_qpe_stage_product(n_prec, factors)`, for
product-form U = ⊗_i u_i (the diagonal on-grid variant and the
product-rotated variant — not the dense-V rotated variant). Instead of
routing every controlled-U^(2^p) through one monolithic 8×8
`UnitaryGate.control()` (the pre-existing dense path, left completely
untouched), it applies n_targ independent controlled-single-qubit gates
per precision qubit, computing each u_i^(2^p) exactly (closed-form integer
modular arithmetic for diagonal factors; exact eigendecomposition for the
product-rotated case).

Verified independently by the Verifier (not just the Coder's self-report):
`Operator` equivalence with the dense path confirmed up to global phase
across multiple seeds/k-vectors, both on-grid variants, and two fully
generic (non-on-grid) Haar product-form unitaries built independently of
the on-grid machinery — so the equivalence is not an artifact of one
specific instance. P(m1==m2) = 1.0 exact (literal float, not just rounded)
via the factorized path for both target types and both product-form
variants. Sampled MCM check: 4000/4000 shots agreeing for both target
types (via StatevectorSampler, since qiskit_aer remains uninstalled in
this environment — the AerSimulator branch of the code was not exercised
in this environment, only its fallback). CLAUDE.md known-bug items (1)-(4)
confirmed on the new stage builder, including a per-target-qubit MSB-first
check with *different* k_i per qubit and only a subset of target qubits
initialized (a stronger test than an all-qubits-set sweep, since it can
catch a per-qubit ordering bug a symmetric-sum test would miss). The
pre-existing `build_qpe_stage` function has zero changed lines (confirmed
via git diff), and the dense-U default path (`factors=None`) is confirmed
byte-identical to before — `--regression` still gives 0.8791, and
`decode_mcm_qpuf_result.py`'s standalone output is unchanged.

Gate counts (n_targ=3, LOCAL/PROVISIONAL — qiskit 2.0.0 in this
environment, pending reconciliation against the braket-arm environment's
qiskit 2.3.0, per the Director's note that a separate DCV Worker run is in
progress there): diagonal-factorized 176/279/363 gates at n_prec=3/4/5
(roughly half the dense diagonal path's 395/561/709); product-rotated-
factorized 350/546/716 (versus the dense product-rotated path's
1,676/2,392/3,134) **(superseded — see Addendum 3 below for reconciled
braket-arm numbers)** — this pulls the product-rotated variant back under
the 2,000-gate IonQ limit at n_prec=4 and 5, where the dense path was over
it. These counts are independently reproduced by the Verifier and match
the Coder's figures exactly, but remain provisional pending the
qiskit-version reconciliation noted in the prior addendum (now resolved —
see Addendum 3 below).

**Addendum 3 (follow-up)**: Gate-count reconciliation, run in the
`braket-arm` conda environment (`qiskit==2.3.0`, `qiskit-aer==0.17.2`,
`amazon-braket-sdk==1.108.1`) per the Director's request. **Anchor check
passed exactly**: `probe_gate_limit.py`'s unmodified `build_and_transpile()`
for the Haar seed=10, n_targ=3 circuit reproduces 1,915 gates at n_prec=3
and 2,583 at n_prec=4 — bit-for-bit matching the AWS-confirmed figures in
that script's docstring. This confirms braket-arm is the trusted
gate-count environment; the earlier qiskit-2.0.0-local numbers reported in
the addenda above were purely an interpreter/version mismatch, not a bug
in this project's code. **Going forward, gate counts for
hardware-submission decisions must be taken from the braket-arm
environment, not the local Mac's qiskit 2.0.0.**

Reconciled gate-count table (n_targ=3, ks=(1,2,4), eigenstate target,
seed=1; total gates / 2-qubit gates; UNDER/OVER relative to the IonQ
Forte-1 2,000-gate hard limit):

| variant | n_prec=3 | n_prec=4 | n_prec=5 |
|---|---|---|---|
| diagonal on-grid, dense U | 448/118 (under) | 640/166 (under) | 808/210 (under) |
| diagonal on-grid, factorized | 179/30 (under) | 308/54 (under) | 410/74 (under) |
| product-rotated, dense U | 1,844/320 (under) | 2,478/432 (**over**) | 3,068/536 (**over**) |
| product-rotated, factorized | 177/30 (under) | 301/54 (under) | 403/74 (under) |
| dense-rotated V·D·V†, dense U | 1,896/324 (under) | 2,560/440 (**over**) | 3,208/552 (**over**) |

Both non-factorized generic-eigenbasis variants (product-rotated dense,
dense-rotated V·D·V†) exceed the 2,000-gate limit at n_prec≥4 — this is
confirmed as a real, environment-independent finding (present under both
qiskit 2.0.0 and 2.3.0), not an artifact of the earlier version mismatch.
The factorized-path variants stay comfortably under budget at all three
n_prec tested. Of the generic-eigenbasis (non-computational-basis) options,
product-rotated-factorized is the only one that both (a) avoids the
diagonal variant's degenerate computational-basis eigenbasis and (b) stays
under the gate limit through n_prec=5 — a candidate for the eventual
hardware design, though the final choice is the Physicist's/Director's
call, not stated here as a decision.

Also re-ran the sampled MCM check under braket-arm's `AerSimulator`
(qiskit_aer is installed there, unlike the local Mac environment used
earlier) at n_prec=4, n_targ=3, 4,000 shots: P(m1==m2) = 1.0000 for both
target types, both the dense and factorized paths — matching and
cross-confirming the earlier `StatevectorSampler`-based result.

---

## 2026-09-08 — Coder/Verifier (Step 1 — computational-eigenbasis on-grid QPE-QPUF pipeline, noiseless)

**Context**: Follow-on to the "On-grid noiseless two-stage QPE-QPUF
baseline" entry above (same day, earlier); that entry's `on_grid_unitary`
was per-target-qubit (tensor-product structure, U = ⊗_i P(2πk_i/2^n_prec)).
Yon redirected the investigation to a distinct case: **eigenbasis =
computational basis**, but built as one independent random phase per
computational basis state rather than per qubit — no tensor-product
structure. This is Step 1 of a new ~5-step plan
(`there-is-a-thread-logical-clarke.md`): noiseless sim → noisy sim → IonQ
hardware → mitigation sim → mitigation hardware. Only Step 1 (noiseless) is
done; steps 2-5 are not started.

**What was built**:
- `random_diagonal_on_grid_unitary(n_targ, n_prec, rng, return_m_j=False)`
  in `braket_direct/ionq_20260601/submit_QPUF_ntarg.py` (additive; the
  earlier per-qubit `on_grid_unitary` is untouched) — U =
  diag(exp(2πi·m_j/2^n_prec)), one independent random integer
  m_j ∈ [0, 2^n_prec) per computational basis state j. Hard `ValueError` if
  all m_j are equal (degenerate global phase); soft `warnings.warn` if the
  m_j are not all mutually distinct (collisions are legal — P(m1==m2)=1
  still holds, an eigenspace just becomes 2+-dimensional). This generalizes
  the earlier subset-sum-distinctness warning pattern to the fully generic
  per-basis-state case. Dense-U path only, through the unmodified
  `build_qpe_stage`/`build_full_circuit`. Target state is the existing
  seeded RY/RZ product state (not an eigenstate), so the response is a
  superposition over eigenvectors — giving several points along the
  m1=m2 diagonal at different bins, the "circle plot" style Yon requested
  (an eigenstate target would give only one point).
- New driver `braket_direct/ionq_20260601/on_grid_qpuf_pipeline.py`
  (`--stage noiseless` implemented; `noisy`/`submit` raise
  `NotImplementedError`, reserved for later steps).
- New `braket_direct/ionq_20260601/on_grid_qpuf_plots.py`: a verified
  bitstring decoder (`decode_bits`) and the circle-plot function
  (`plot_m1_vs_m2`), lifted in style from `braket_direct_analysis.ipynb`
  cell 5 (that notebook itself untouched).
- Results in `braket_direct/ionq_20260601/on_grid_results/2026-09-08/`.

**Bug found in `braket_direct_analysis.ipynb`, NOT fixed there**: while
building `decode_bits`, both the Coder and the Verifier independently
confirmed (fresh from-scratch re-derivation and fresh circuit execution
each, not re-running the other's saved output) that
`braket_direct_analysis.ipynb` cell 3's `parse_outcome` function has a
fallback branch for a single flat classical register —
`return int(bitstring[n_prec:], 2), int(bitstring[:n_prec], 2)` — that does
**not** correctly decode `build_full_circuit`'s register layout. Concretely:
running the known-answer circuit (on-grid U with k=(1,2,4), eigenstate
target, expected m1=m2=7 on every shot — established in the entry above)
through real `AerSimulator` gives a raw bitstring that the notebook's
fallback decodes as (14,14) — the 4-bit reversal of 7 — instead of the
correct (7,7). Root cause: qiskit's flat-classical-register convention
(`c[0]` is the rightmost bitstring character, `c[L-1]` the leftmost)
combined with this project's established MSB-first physics convention
(precision qubit 0 / c[0] of each stage carries the highest power of U)
requires a bit-reversal within each stage's n_prec-bit block that the
notebook's fallback omits. The new `decode_bits` in
`on_grid_qpuf_plots.py` implements this correctly and was validated against
the known-answer test (100% of shots correctly decoded).
`braket_direct_analysis.ipynb` was **not edited/fixed** as part of this
task — this is an open item for whoever next touches that notebook's
analysis path. Any hardware analysis that used `parse_outcome`'s
single-flat-register fallback branch on a single-register circuit should be
treated with suspicion until fixed there too. (The notebook's separate
multi-register `'c2 c1'` branch, used historically for a different circuit
topology, is not implicated by this finding.)

**Verified numbers** (instance: seed=9, n_targ=3, n_prec=4):
- m_j = [6, 13, 15, 4, 1, 9, 10, 12] — all 8 distinct, no degeneracy warning
  for this instance. The hard/soft guards were separately confirmed to fire
  correctly on constructed test cases (all-equal m_j raises; colliding-but-
  not-all-equal m_j warns).
- Exact joint (`ideal_joint`, branch-exact statevector): P(m1==m2) = 1 to
  machine precision (1 − trace ≈ 1.1e-16, max off-diagonal ≈ 7e-31).
- Sampled via real `AerSimulator` (genuine MCM): 100% on-diagonal at both
  1,000 and 10,000 shots, checked against raw counts, not just the printed
  fraction.
- m1 marginal cross-check: an RY/RZ product-state target was independently
  built from scratch as a raw numpy statevector (not reusing
  `ideal_joint`'s internals); |amplitude|² per basis state merged by shared
  m_j matches the exact `ideal_joint` marginal to machine precision
  (2.2e-16) and the 10,000-shot observed frequencies to within ~1σ of
  binomial noise across all 8 populated bins:
  {1: 0.1597, 4: 0.0855, 6: 0.0764, 9: 0.1659, 10: 0.1722, 12: 0.1788,
  13: 0.0793, 15: 0.0823} (exact-joint values; observed frequencies all
  within ~1σ).
- Gate count (braket-arm environment, qiskit 2.3.0, the project's trusted
  gate-count environment per Addendum 3 above): this fully-generic-diagonal
  U transpiles to 710 gates (542 1q + 168 2q) at n_targ=3, n_prec=4 —
  noticeably more than the earlier structured per-qubit
  `on_grid_unitary(ks=(1,2,4))`'s 640 gates at the same size (reconciled
  table above). The Verifier confirmed this "generic diagonal costs more
  than structured/tensor-product diagonal" effect holds across three
  further seeds (710, 734, 672), not a fluke of one seed — apparently
  qiskit's diagonal-unitary synthesis exploits the structured per-qubit
  phase pattern for extra gate cancellation that a fully i.i.d.-random
  diagonal doesn't have. Both figures are comfortably under the
  2,000-gate IonQ Forte-1 limit.

**Plot**:
`braket_direct/ionq_20260601/on_grid_results/2026-09-08/on_grid_computational_eigenbasis_circle_plot.png`
— two panels (1,000 shots, 10,000 shots), all points exactly on the m1=m2
diagonal at the 8 expected eigenbins, no off-diagonal points.

No hardware was submitted or retrieved as part of this task. Later steps
of the plan (noisy simulation, IonQ hardware, mitigation sim, mitigation
hardware) have not been started.

---

## 2026-09-08 — Coder/Verifier (Step 2 — noisy simulation, computational-eigenbasis on-grid QPE-QPUF)

**Context**: Step 2 of 5 in `there-is-a-thread-logical-clarke.md` (see the
Step 1 entry directly above for the instance definition: seed=9 →
`random_diagonal_on_grid_unitary(3,4,rng=seed 9)`, m_j =
[6, 13, 15, 4, 1, 9, 10, 12], ideal bins {1,4,6,9,10,12,13,15}, n_targ=3,
n_prec=4, product-state target, genuine MCM; noiseless P(m1==m2)=1.0
exact, braket-arm hardware-submission gate count 710 at opt_level=1).

**What was tested and why**: added `--stage noisy` to
`on_grid_qpuf_pipeline.py`, reusing `build_noise_model`/`_run` from
`braket_direct/20260828_ionq_error_mitigation/mitig_analysis.py`
(depolarizing error on `rx`/`rxx` plus an all-qubit readout error, IonQ
native basis rz/rx/rxx/id, `optimization_level=0`). Opt_level=0 is
deliberate — at opt_level=1 the transpiler cancels/fuses some of the gates
the noise model attaches error to. Same seed-9 circuit as Step 1,
`build_full_circuit` unmodified. Purpose: get a noise-model-conditioned
prediction for the eventual IonQ hardware run (Step 3) before spending any
hardware budget.

**Pre-flight checks** (both independently confirmed by the Verifier before
trusting the noisy arms below):
- The noiseless limit of the noisy code path itself
  (`build_noise_model(1.0, 1.0, 0.0)`) reproduces P(m1==m2)=1.0 exactly —
  confirms the noisy execution path isn't silently altering the circuit or
  decode logic when error rates are zeroed.
- The Step 1 `decode_bits` decoder works unchanged on the noisy backend's
  output, confirmed via a known-answer probe run through the same noisy
  execution path.

**Results table** (Clopper–Pearson 95% CIs; the Verifier independently
reproduced 2 of the 5 arms and found them statistically consistent — an
apparent single-seed discrepancy on the f2q_0.995 arm was resolved cleanly
once the Verifier ran 20 independent seeds and confirmed the claimed value
sits well within the expected sampling distribution, not a systematic
issue):

| arm | n | P(m1==m2) [95% CI] | P(m1∈ideal) | P(m2∈ideal) | P(dist≤1 mod 16) |
|---|---|---|---|---|---|
| nominal (F1Q=0.9998, F2Q=0.9882, p_ro=0.006), 1,000 shots | 1,000 | 0.2880 [0.2601, 0.3172] | 0.7440 [0.7158, 0.7708] | 0.7540 [0.7261, 0.7804] | 0.3880 [0.3577, 0.4190] |
| nominal, 10,000 shots | 10,000 | 0.2783 [0.2695, 0.2872] | 0.7405 [0.7318, 0.7491] | 0.7394 [0.7307, 0.7480] | 0.4014 [0.3918, 0.4111] |
| F2Q=0.985 (sensitivity), 10,000 shots | 10,000 | 0.2078 [0.1999, 0.2159] | 0.7039 | 0.6991 | 0.3346 |
| F2Q=0.995 (sensitivity), 10,000 shots | 10,000 | 0.5307 [0.5209, 0.5405] | 0.8518 | 0.8600 | 0.6197 |
| readout-only (F1Q=F2Q=1.0, p_ro=0.006), 10,000 shots | 10,000 | 0.9522 [0.9478, 0.9563] | 0.9840 | 0.9836 | 0.9642 |

**Gate count**: the noisy-transpile path (opt_level=0, required so the
noise model has gates to attach to) gives 1,740 1q + 168 2q = 1,908 total
gates, versus Step 1's braket-arm hardware-submission reference of 710
(opt_level=1). The Verifier confirmed the 2q count (168, the `rxx`
entangler count) is identical between both optimization levels — only
single-qubit gate fusion/cancellation differs between opt_level=0 and 1, so
this is expected transpiler behavior, not a structural difference. The two
counts serve different purposes (noise-model simulation vs. hardware
submission sizing) and are not meant to be compared directly as an
apples-to-apples figure.

**Physics findings** (stated as findings; interpretation left to the
Physicist/Director):
1. **F2Q sensitivity is steep**: P(m1==m2) swings from 0.21 (F2Q=0.985) to
   ~0.53 (F2Q=0.995) — a >2.5x range from a 1-percentage-point change in
   two-qubit gate fidelity. Confirmed real by independent reproduction, not
   a single-seed artifact: the Verifier ran 20 seeds and confirmed the
   spread is consistent with expected binomial sampling noise around a
   real, steep dependence. Mechanistically plausible given ~168 `rxx`
   gates compounding a per-gate fidelity multiplicatively.
2. **Gate error, not readout error, dominates the signal loss**: the
   readout-only arm (P(m1==m2)=0.9522) stays close to the noiseless ideal,
   while the nominal arm (which adds gate depolarizing error) drops to
   0.2783. Confirmed real, mechanistically consistent with depolarizing
   error hitting 1,740+168 gates versus readout error hitting only 8
   measured qubits once each.

**Caveat**: the noise model used here (independent depolarizing error on
`rx`/`rxx` plus independent readout error) is a Markovian proxy, not a
validated model of Forte-Enterprise-1's actual physics. It cannot
discriminate between this hypothesis and, e.g., a coherent-rotation noise
hypothesis (the MCM-eigenvector-switching mechanism this project is
separately investigating — see the 2026-08-28 Worker2 entry above). This
simulation confirms the code executes and samples correctly under the
stated depolarizing hypothesis; it is not itself evidence for or against
that hypothesis being what is physically happening on real hardware.

**Hardware prediction** (model-conditioned, not a claim about what
hardware will actually show, given the Markovian-proxy caveat above): if
the noise model's calibration parameters used here hold, the nominal
1,000-shot arm predicts **P(m1==m2) ≈ 0.29 [0.26, 0.32] at 1,000 shots
under the nominal noise model (F1Q=0.9998, F2Q=0.9882, p_readout=0.006)**.

**Plots**:
- `braket_direct/ionq_20260601/on_grid_results/2026-09-08/on_grid_noisy_nominal_circle_plot.png`
  — noiseless (Step 1) vs. noisy-nominal circle plots, 1,000 and 10,000
  shots, 2×2 layout.
- `braket_direct/ionq_20260601/on_grid_results/2026-09-08/on_grid_noisy_sensitivity_circle_plot.png`
  — F2Q=0.985/0.9882/0.995 and readout-only circle plots, all at 10,000
  shots, 2×2 layout.

This is Step 2 of the 5-step plan. Step 3 (a genuine IonQ Forte-1 hardware
run) requires Yon's commit of the current work and his explicit,
per-submission go-ahead naming the device, circuit, shot count, and cost —
not started. No hardware was submitted or retrieved as part of this task.

---

## 2026-09-08 — Coder/Verifier (Step 3 prep — hardware-submission scaffolding + MCM program-order finding)

**Headline finding — open blocker for MCM-attribution claims**: while
verifying Step 3's submission scaffolding, a real bug was found and
independently confirmed twice (two separate verification passes, plus
reproduced in a trivial standalone example): **Qiskit's `transpile()`
reorders circuit instructions by DAG data-dependency, not program order.**
For this project's two-stage MCM circuit (`build_full_circuit` in
`submit_QPUF_ntarg.py`), stage 2's gates have no classical-conditioning
dependency on the stage-1 measurement outcome, so a single global
`transpile()` call legally (from Qiskit's perspective) moved ALL of stage
2's gates to before any measurement in the exported program — the
originally-exported submission QASM had every one of its 710 gates,
including stage 2's controlled-U operations, appearing textually **before**
the first measure statement. The mid-circuit measurement's program-order
position was destroyed.

**Reproduction**: a trivial 2-qubit counter-example (`h(0); measure(0,c0);
x(1); measure(1,c1)`, where `x(1)` has zero dependency on q0's measurement)
shows the same reordering at every `optimization_level` 0-3, both in
Qiskit's internal circuit representation and in the exported OpenQASM3
text.

**Why this wasn't caught by any prior noiseless validation, including
today's**: this circuit has no classical feedforward — nothing downstream
depends on the stage-1 classical measurement bits. By the
deferred-measurement theorem, noiseless statistics are identical whether
the measurement genuinely happens mid-circuit or is effectively deferred
to the end. Every noiseless check this project has run to date (including
Steps 1-2 above) is consistent with **both** "genuine MCM happened as
designed" **and** "no MCM happened at all — the target evolved unitarily
through both stages before a single end-of-circuit measurement." Noiseless
simulation structurally cannot discriminate between these two hypotheses
for this circuit design.

**Scope**: the exact same `transpile(qc,
basis_gates=['rz','rx','rxx','measure','reset'], optimization_level=1)`
call already used in `submit_QPUF_ntarg.py`'s `main()` and in
`two_stage_on_grid_baseline.py` — i.e., the same call pattern used for
every prior real IonQ hardware submission this project has made — is
subject to the same reordering. This means this finding is an **open
blocker specifically for any MCM-attribution claim** (any claim that a
hardware result reflects measurement-induced disturbance specifically
because the measurement happened mid-sequence), for both future
submissions and any reinterpretation of past hardware results framed
around genuine mid-circuit measurement — including the central
p_flip/eigenvector-switching MCM-disturbance investigation named in
CLAUDE.md §0. It does **not** affect claims that don't depend on
measurement timing — e.g. raw noise-level measurement or error-mitigation
benchmarking results that don't hinge on the measurement being genuinely
mid-circuit are unaffected by this finding.

**Also unresolved (explicitly out of this project's ability to verify
locally)**: whether IonQ/Braket's server-side compiler (which re-targets
even non-verbatim OpenQASM3 to native GPI/GPI2/ZZ gates) executes a
submitted program in its literal textual/QASM order, or does further
reordering of its own once it receives the (now program-order-correct)
QASM. This needs IonQ/AWS documentation or an AWS-Agent investigation — it
was not and could not be resolved by this session's local-only
verification.

**The fix implemented (this prep's actual deliverable)**:
`on_grid_qpuf_pipeline.py`'s `--stage submit --dry-run` path now
transpiles the circuit in two separate pieces — (target-prep +
stage-1-QPE + stage-1-measure) and (stage-2-QPE + stage-2-measure) — each
with the same settings as the original 710-gate reference
(`basis_gates=['rz','rx','rxx','measure','reset']`,
`optimization_level=1`), then assembles them in explicit program order
into a fresh flat-register circuit with **no further global `transpile()`
call** — avoiding the DAG-reordering bug by construction rather than by
relying on transpiler behavior. Independently verified twice: the exported
QASM's stage-1 measure statements now sit immediately after the last
stage-1 gate and before the first stage-2 gate (confirmed by direct
parsing of the QASM text, not just the Python circuit object), with strict
qubit-locality on each side of the boundary (everything before the
stage-1 measures touches only target+prec_a qubits; everything between the
two measure blocks touches only target+prec_b qubits).

Barriers were deliberately **not** used to additionally pin the ordering —
the device's cached `supportedPragmas`/`supportedOperations`
(`device_properties_raw.json`) don't list barrier support, and without
positive confirmation both that Braket's local parser accepts them and
that IonQ's device honors them, they were left out; program-order assembly
is relied on instead.

Gate count: 711 (543 1q + 168 2q) — one more than the original 710,
attributed to losing a single cross-seam single-qubit-gate fusion
opportunity at the stage-1/stage-2 boundary now that the pieces are
transpiled independently. Still comfortably under the 2,000-gate IonQ
limit.

**Verification of the fix** (both LocalSimulator re-import checks):
re-importing the regenerated exported QASM text via Braket SDK's
`Circuit.from_ir` and running on Braket's `LocalSimulator()` (fully local)
at 4,000 shots gives 100% on-diagonal, matching the exact ideal joint's
per-bin marginal (bins {1,4,6,9,10,12,13,15}) within shot noise. This check
is necessary but **not sufficient** to prove genuine mid-circuit execution
on real hardware, for the reason given above — this caveat is written into
the manifest itself:

> "This confirms program-order preservation is NECESSARY for a correct
> submission but it is NOT SUFFICIENT to prove the physical device
> executes the measurement in that position — noiseless statistics are
> IDENTICAL whether the mid-circuit measurement is genuinely performed
> mid-circuit or the measurement is effectively deferred by the
> server-side compiler, since there is no classical feedforward in this
> circuit for either ordering to affect. Whether IonQ/Braket's server-side
> compiler preserves genuine mid-circuit execution (vs. an internal
> deferred-measurement rewrite) is a property of that compiler, UNVERIFIED
> here, and only a real hardware run (or documented compiler behavior)
> could settle it."

**Decode rule** (unchanged from before the fix, restated for
completeness): raw Braket convention B,
`measured_physical_qubits=[3,4,5,6,7,8,9,10]` (q0-2=target unmeasured,
q3-6=stage-1→c[0..3], q7-10=stage-2→c[4..7]); with `bits`=raw counts-key
string read left to right: `m1=int(bits[0:4],2)`, `m2=int(bits[4:8],2)`.

**Files** (new/regenerated, under
`braket_direct/ionq_20260601/on_grid_results/2026-09-08/submit/`):
`on_grid_seed9_submission.qasm`, `on_grid_seed9_manifest.json`,
`on_grid_seed9_ideal_joint.json`. Modified:
`braket_direct/ionq_20260601/on_grid_qpuf_pipeline.py`.

**Status**: this is Step 3 PREP only. No hardware submission has occurred
or is authorized — that requires Yon's explicit commit and per-submission
go-ahead (device, circuit, shots, cost), not yet given. The `--stage
submit` (real, non-dry-run) path exists but is gated behind an
`--i-have-yons-go-ahead` flag plus an interactive confirmation, and was
confirmed (independently, twice) to exit safely before any AWS object is
constructed when that flag is absent.

---

## 2026-09-08 — Coder/Verifier (Step 4 — error mitigation on the noisy
simulator, seed-9 on-grid instance)

**Context**: Step 4 of 5 in `there-is-a-thread-logical-clarke.md`. Same
seed-9 instance as Steps 1-3 above: n_targ=3, n_prec=4,
m_j=[6,13,15,4,1,9,10,12], ideal bins {1,4,6,9,10,12,13,15}, product-state
target, genuine two-stage MCM. Builds on Step 3's fix for the
`transpile()` DAG-reordering bug (piecewise transpile-then-assemble in
explicit program order, no further global `transpile()` call).

**What Step 4 required beyond a normal REM/DDD/ZNE port — three more
instances of the Step-3 reordering-bug class, found and designed around**:

1. `error_mitig_ionq.py`'s `ErrorMitigation.fold()` (used for ZNE)
   internally strips measurements via `remove_final_measurements()`, which
   Qiskit judges per-qubit. Applied to the full assembled two-stage
   circuit, this would strip both measurement events and re-defer them to
   the end — exactly undoing Step 3's fix. **Fix**: apply fold/DDD to each
   of the two circuit pieces separately (each piece has exactly one
   terminal measurement, the shape `fold()`/DDD correctly handle), then
   reassemble via Step 3's `_assemble_in_program_order`-style logic.
2. `mitig_analysis.py`'s `_run()` internally calls
   `transpile(circ, sim, optimization_level=0)` even when the circuit is
   already in the simulator's exact target basis — this was verified live
   to still reorder instructions. **Fix**: skip that transpile for the
   final simulation call; run the already piecewise-assembled,
   already-correct-basis circuit directly on `AerSimulator`.
3. A third bug, found and fixed during implementation (not anticipated
   going in): `ErrorMitigation.fold()`'s `_reattach_measurements`
   renumbers clbits starting at 0 via
   `enumerate(measured_physical_qubits(original))` — correct for a
   single-stage circuit, but wrong when applied to "piece 2" in isolation,
   whose real clbits should be the upper half of the register, not
   0-indexed. Verified independently by the Verifier: reproduced the bug
   standalone (a circuit measuring into clbits 4-7 got renumbered to 0-3 by
   unmodified `fold()`), then confirmed the fix (each piece gets its own
   local register during fold/DDD; reassembly explicitly remaps piece 2's
   local clbits to the register's upper half) works correctly end-to-end.

All three fixes live entirely in `on_grid_qpuf_pipeline.py`;
`error_mitig_ionq.py` and `mitig_analysis.py` (shared with the
Rigetti/Cepheus and single-stage IonQ campaigns) were **not modified**.

**Additional pre-simulation check**: every arm's final assembled circuit
was checked, before simulating, to confirm the stage-1 measurement still
sits strictly between the two stages' gates — the same class of check
Step 3 established. The Verifier independently re-derived this check from
first principles (their own first attempt at an independent check had a
false-positive flaw, which they caught and fixed themselves — noted here
as evidence of real independent verification, not a rubber stamp) and
confirmed all 7 arms pass.

**Decode cross-check**: the on-grid pipeline's already-verified
`decode_bits` decoder (Step 1) was cross-validated against
`mitig_analysis.py`'s own `counts_to_m` decoder (needed for REM's
confusion-matrix machinery) — 256/256 observed bitstrings agreed exactly,
independently reproduced by the Verifier on a fresh sample (241/241
agreement).

**Noise models**: nominal — F1Q=0.9998, F2Q=0.9904, p_readout=0.0036
(today's live IonQ device-query values). Comparison — same F1Q/p_readout,
F2Q=0.9882 (Step 2's originally-assumed value), run for the raw arm only,
to compare against Step 2's headline number.

**Results — point estimates** (10,000 shots each, seeded; verified exactly
reproducible by the Verifier — bit-for-bit identical on a same-seed rerun,
confirming full pipeline determinism):

| arm | P(m1==m2) | P(m1∈ideal) | P(m2∈ideal) | P(dist≤1 mod16) |
|---|---|---|---|---|
| raw (F2Q=0.9904, nominal) | 0.3410 | 0.7803 | 0.7763 | 0.4567 |
| raw (F2Q=0.9882, comparison) | 0.2814 | 0.7459 | 0.7371 | 0.4051 |
| REM | 0.3493 | 0.7858 | 0.7812 | 0.4639 |
| DDD-off (detuned baseline, no pulses) | 0.0776 | 0.5029 | 0.5104 | 0.2023 |
| DDD-on (no injected error) | 0.3303 | 0.7700 | 0.7681 | 0.4485 |
| DDD-on (with injected coherent detuning) | 0.1549 | 0.5411 | 0.5438 | 0.2707 |

**Which arms help — stated plainly, per the Director's request**:

- **REM: helps, modestly, with high confidence.** Paired bootstrap delta
  is significant and positive on all 4 observables (e.g. P(m1==m2):
  +0.0083, 95% CI [0.0075, 0.0091] — independently reproduced by the
  Verifier). Confusion-matrix condition number 1.06 (excellent — very
  little variance cost for the correction).

- **DDD: only helps against the specific coherent-noise mechanism it
  targets — and against that mechanism, it helps a lot; it does NOT help
  against ordinary depolarizing noise, and may slightly hurt.** The DDD
  baseline comparison is DDD-on-detuned vs. DDD-off-detuned (both under
  injected coherent idle-error detuning) — this isolates "does DDD
  counteract detuning" rather than conflating it with "does detuning hurt
  in the first place" (this baseline choice matches `mitig_analysis.py`'s
  own established precedent — its `validate()` harness's "none" baseline
  is itself detuned, confirmed by the Verifier). Under that comparison,
  DDD roughly doubles P(m1==m2) (0.0776 → 0.1549), a significant positive
  delta on all 4 observables (P(m1==m2): +0.0772, 95% CI [0.0678,
  0.0857]). But comparing DDD-on-with-no-injected-error (0.3303) against
  plain raw (0.3410) shows DDD is very slightly *worse* under ordinary
  Markovian noise with no coherent component — consistent with the
  expected result that DD pulses add gates (and hence depolarizing
  exposure) without anything coherent to cancel, so they can't help and
  may cost a little. This is recorded as a real, informative negative
  result for "does DDD help under plain depolarizing noise" — not a
  failure of the implementation.

- **ZNE: extrapolation is directionally informative but not reliable as a
  precise point estimate.** Linear-model point estimates:
  P(m1==m2)=0.5177, P(m1∈ideal)=0.8899, P(dist≤1)=0.6155. Caveat (verified
  present in the machine-readable results, not softened anywhere in the
  chain): the P(m1==m2) extrapolation has model_spread≈0.29 across
  richardson/linear/exponential fits — the models disagree substantially
  with each other, which is itself evidence the noise isn't cleanly in
  the regime extrapolation assumes at this circuit depth (per
  `mitig_analysis.py`'s own stated interpretation: model disagreement *is*
  the finding, not just uncertainty to average over). The P(dist≤1)
  exponential model is flagged explicitly unphysical (extrapolated value
  1.0514, outside [0,1]) — an invalid fit, correctly excluded/flagged
  rather than reported as a real result. The coarser P(m1∈ideal)
  observable has much better model agreement (spread≈0.029), suggesting
  ZNE's reliability depends heavily on which observable is being
  extrapolated.

- **ZNE+REM: tracks ZNE closely with a small additional positive shift
  from REM**, same model-spread/unphysical-flag pattern as plain ZNE —
  REM's benefit compounds through the extrapolation but does not fix
  ZNE's underlying model-disagreement problem.

**Gate counts** (opt_level=0, `rz/rx/rxx/id` basis — this is the
*simulation* transpile setting, not Step 3's opt_level=1
*hardware-submission-sizing* setting; the two are not directly
comparable): raw/zne_scale_1: 1,908 (under the 2,000-gate limit);
zne_scale_1.5: 2,862 (over); zne_scale_2: 3,816 (over); ddd_off_detuned:
2,042 (over); ddd_on_plain: 2,484 (over); ddd_on_detuned: 2,612 (over).
The raw arm's 1,908-gate count exactly matches Step 2's own previously-
reported opt_level=0 count for this same circuit — a cross-validation
that the new piecewise-transpile approach reproduces the same physical
circuit Step 2 already used. This particular cross-check is the Coder's
claim, not independently re-opened by the Verifier (it was outside that
pass's 9-item checklist). Most mitigation arms exceed the 2,000-gate
hardware limit — informational only for now (this step is simulation
only), but relevant if any of these mitigation techniques are ever
considered for a real hardware submission at this circuit size.

**Open provenance question, flagged by the Verifier, not resolved here**:
`ionq_mitigation_common.py`'s F2Q default was separately updated (by other
concurrent work, not this task) to 0.9882, labeled there as "today's"
calibration-derived value — while this task's own pipeline labels 0.9882
as "Step 2's original value, kept for comparison" and 0.9904 as "today's
live device-query value." These two labels for 0.9882 conflict and should
be reconciled by the Physicist/Director; not resolved as part of this
task.

**Plot**:
`braket_direct/ionq_20260601/on_grid_results/2026-09-08/mitigate/on_grid_mitigation_circle_plots.png`
— 6 panels: raw, REM-input, ZNE scale=2, DDD-off-detuned, DDD-on-plain,
DDD-on-detuned.

**Files**: Modified —
`braket_direct/ionq_20260601/on_grid_qpuf_pipeline.py` (added
`--stage mitigate`). New —
`braket_direct/ionq_20260601/on_grid_results/2026-09-08/mitigate/mitigate_results.json`,
`.../on_grid_mitigation_circle_plots.png`.

**Status**: this is Step 4 of 5. Step 5 (mitigation on real IonQ
hardware) requires Yon's commit and per-submission go-ahead (device,
circuit, shots, cost) and has not started; it is also separately gated
behind resolving the still-open MCM-deferral/server-side-compiler
question raised in the Step 3 entry above (whether IonQ/Braket's
server-side compiler preserves genuine mid-circuit execution). No
hardware was submitted or retrieved as part of this task.

---

## 2026-08-28 — Coder (REM/DDD/ZNE campaign scaffolding for IonQ)

**What**: Created a new error-mitigation benchmark campaign directory,
`braket_direct/20260828_ionq_error_mitigation/`, mirroring the structure of
`braket_direct/rigetti_cepheus_20260807/` (REM, DDD, ZNE on QPE) but adapted
for IonQ Forte-Enterprise-1's all-to-all connectivity and **on-demand**
submission (bare `AwsDevice(DEVICE_ARN).run(...)`, no Braket Direct
reservation). File-creation/code-writing only.

**Why**: The Rigetti campaign has a REM/DDD/ZNE benchmarking pipeline with
substantial numerical hygiene (stratified ZNE folding, real-duration idle-
window detection for DDD, simplex-projected REM inversion, bootstrap CIs,
physicality checks on extrapolated values). No equivalent existed for IonQ;
this task built one from scratch rather than by analogy, since IonQ's
device model differs from Rigetti's in ways that matter (all-to-all vs
fixed lattice, on-demand vs reservation, gate-count-based vs depth-based
runtime, and — the load-bearing difference — verbatim/compiler-cancellation
status is confirmed on Rigetti and entirely unconfirmed on IonQ).

**Files created**:
- `ionq_mitigation_common.py` — circuit builders (`build_qpe_stage` with the
  inverse-QFT endianness fix already in place, `build_qpe_circuit`,
  `build_qpuf_two_stage`, `known_phase_unitary`, `ideal_qpe_distribution`),
  Haar sampling (QR + Mezzadri phase-fix), IonQ timing/fidelity constants
  (copied from `noise_mitigation/ionq_noise_mitig.py`, flagged as
  unvalidated estimates, not calibration), OpenQASM 3 export (rxx→xx
  rewrite, single flat `QuantumRegister`), device-caps loading.
- `error_mitig_ionq.py` — `ErrorMitigation` class: DDD (same pulse algebra
  as Rigetti's `error_mitig.py`, `RZ_TIME_S` introduced as a named,
  explicitly-commented open question rather than silently adopting either
  device's convention), ZNE folding/`zne_circuit_set` (fractional λ, no
  verbatim-integer restriction), REM calibration circuits, verification.
  All chiplet/placement/routing code deliberately omitted (IonQ is
  all-to-all; that code would be dead weight standing in for a constraint
  that does not exist on this device).
- `mitig_analysis.py` — near-verbatim copy of the Rigetti estimator
  (bitstring decoding, REM inversion, ZNE extrapolation, bootstrap CIs);
  changed: rz/rx/rxx noise-model basis, `validate()` defaults swapped to
  IonQ's stated F1Q=0.9998/F2Q=0.985.
- `submit_ionq_mitigation.py` — interactive on-demand submission (bare
  `device.run`, no reservation), `MAX_TOTAL_GATES` guard explicitly
  commented as a self-imposed budget guess, not a cited AWS/IonQ limit.
- `checkRetrieve.py` — adapted from `noise_mitigation/checkRetrieve.py`
  (log-driven, on-demand-appropriate QUEUED→RUNNING→COMPLETED handling).
- `cancelQueue.py` — adapted from
  `braket_direct/ionq_20260601/cancelEditQueue.py` with all reservation
  (`RES_ARN`) references stripped.
- `query_device_caps.py` — copied from `braket_direct/ionq_20260601/`
  as-is; **not run** (no AWS credentials/DCV access in this session).
- `MITIGATION_BRIEF_IONQ.md` — full brief mirroring
  `MITIGATION_BRIEF.md`'s structure. Section 9 ("Open questions") is the
  physicist's text reproduced **verbatim, unedited**, per explicit
  instruction not to paraphrase or soften it — it documents that (1)
  whether IonQ's server-side compiler cancels ZNE folds / DD pulses is
  completely unconfirmed (no `device_caps.json` for IonQ exists anywhere in
  this repo), and (2) DDD has never been run on IonQ here and its timing
  model contains an unresolved `rz`-timing contradiction between the
  Rigetti and IonQ cost models. An editorial addendum immediately below
  Section 9 notes that item (b) referenced there ("inverse-QFT endianness
  bug … unpatched") was fixed and verified later the same day (this log's
  entry directly above), so a future reader is not misled.

**Verification**: All seven Python files pass `py_compile`. Imported and
exercised end-to-end in the `braket-arm` conda environment (qiskit 2.3.0,
qiskit-aer 0.17.2, amazon-braket-sdk present) with NO network/AWS calls:
- Known-answer regression test — `build_qpe_circuit` + `ideal_qpe_distribution`
  at φ=1/8, n_prec=5, exact eigenstate → bin 4, p=1.0000000000000 exactly,
  confirming the inverse-QFT endianness fix carried over correctly into
  this new module.
- `ErrorMitigation.verify_sequences()` — XX/YY/XY4/II identity-up-to-phase
  True, XYXYX False (net bit flip), matching the Rigetti-side regression.
- `apply_ddd` + `verify_ddd`, `zne_circuit_set` + `verify_zne`,
  `rem_calibration_circuits` all ran without error on a transpiled 6-qubit
  QPE circuit; OpenQASM export confirmed no `rxx` token leaks through
  (rewritten to `xx`).
- `mitig_analysis.py --validate --n-prec 3 --shots 2000 --n-boot 30` ran
  end-to-end on the local `qiskit-aer` simulator (no hardware) and produced
  output in the expected shape (bootstrapped deltas, physicality flags on
  the ZNE fit, etc.) — the actual numbers are a smoke test only, not a
  claim about IonQ's real behavior, consistent with the brief's Section 7
  explicitly stating no validation numbers have been recorded yet.

**Scope discipline**: nothing was submitted to real hardware, no
`query_device_caps.py` run against a live device, and no file outside
`braket_direct/20260828_ionq_error_mitigation/` and `docs/LOG.md` was
modified. `braket_direct/ionq_20260601/` and `noise_mitigation/` were read
as templates only.

---

## 2026-08-28 — Worker2 (scope correction, same-day)

**What**: Removed `build_qpuf_two_stage` from
`braket_direct/20260828_ionq_error_mitigation/ionq_mitigation_common.py`.
It was unused by every campaign script (`submit_ionq_mitigation.py`,
`mitig_analysis.py`, `error_mitig_ionq.py` all only call
`build_qpe_circuit`), but its presence as dead code contradicted an
explicit scope confirmation from the Director: this campaign is
single-stage QPE only, matching the Cepheus campaign — the two-stage QPUF
protocol (and its no-MCM design gap / p_flip re-baselining) is a separate
thread and out of scope here. Also fixed a stale reference to the removed
function in `MITIGATION_BRIEF_IONQ.md`'s file-inventory table (Section 2).

**Verification**: `py_compile` clean after the edit; grep confirms no
remaining references to `build_qpuf_two_stage` anywhere in the campaign
directory.

---

## 2026-08-28 — Worker1 (verifier/coder/physicist/verifier chain, Cepheus)

**What**: (1) Verified an in-progress, uncommitted placement-grouping fix in
`qpe_mitigation_analysis.ipynb` (7-tuple family key, appends
`measured_qubits` to the existing 6-tuple config key). (2) Extended the
REM/ZNE analysis from one manually-selected config to all 37 distinct run
families present in the 177-task 2026-08-07 hardware collection, via a new
`run_all_families.py`. (3) Physicist assessed the expected REM-flat /
ZNE-degrades-with-n_prec hypothesis against the real 37-family data and
reassessed `MITIGATION_BRIEF.md` §7. (4) Final verifier pass on the full
deliverable, which caught and required a fix for a table-generation bug.

**Why**: Only one config (n_prec=8, n_targ=1, haar) had been analyzed;
the Director wanted REM/ZNE benefit characterized across the full sweep
(n_prec ∈ {2,3,4,5,6,8,10,15}, n_targ ∈ {1..5}) before any paper-facing
claims. The placement-grouping fix was needed first because several configs
were run on multiple physical qubit placements with very different readout
fidelity, and the old key would have pooled them.

**Files**:
- `braket_direct/rigetti_cepheus_20260807/qpe_mitigation_analysis.ipynb` —
  uncommitted diff, verified correct, **not committed** (not asked to).
  Contains one unrelated bundled change (`K=12`→`K=100` in the Fig. 1
  histogram bin count) — flagged, not touched, still present.
- `braket_direct/rigetti_cepheus_20260807/run_all_families.py` — new script,
  reuses `mitig_analysis.py`'s REM/ZNE/bootstrap functions; owns only the
  grouping/orchestration loop (verified byte-identical `family_key` to the
  notebook's).
- `braket_direct/rigetti_cepheus_20260807/job_results_qpe/analysis_summary_*.json`
  — 38 files (1 pre-existing + 37 new), collision-safe filenames
  (`..._q{sha1(full 7-tuple key)[:10]}.json`) after fixing a real collision
  bug (old scheme omitted zne_scales/placement; two n_prec=8/n_targ=1/haar
  families already collided on it).
- `braket_direct/rigetti_cepheus_20260807/plots/mitigation_benefit_table.{csv,json}`,
  `plots/mitigation_benefit_all_families.png` — consolidated ΔP_correct vs
  n_prec/n_targ. Regenerated after a fix: the `valid` flags originally
  checked only the linear ZNE fit's range instead of the joint
  `physical` flag `mitig_analysis.extrapolate()` already computes across
  all fitted models, overstating validity in 11/26 (ZNE) and 8/24
  (ZNE+REM) rows. Fixed in `run_all_families.py`; per-family JSONs were
  never wrong, only the rollup.

**Verification**: Placement-fix correctness traced to source
(`rigetti_qpuf_common.py`, `checkRetrieve.py`, `mitig_analysis.py`'s
`counts_to_m`) — order-sensitive keying is necessary because
`measured_qubits` order encodes the clbit→physical-qubit map and
`pool_counts()` sums raw bitstrings positionally; regression check
reproduced the existing n_prec=8/haar export to full float precision.
`family_key` diffed character-for-character between notebook and script —
identical. Consolidated table re-checked post-fix: 0/26 and 0/24 flag
mismatches (down from 11/26, 8/24).

**Physics findings (negative/corrective — see full report to Director for
detail)**: The REM-flat / ZNE-degrades-with-n_prec working hypothesis is
**refuted**. REM's effect is proportional to signal strength and changes
sign (24/33 families have negative ΔREM) — expected physics once most
n_prec≥8 families are shown to sit at the uniform noise floor
(`P_correct` ≈ `1/2^n_prec`), not evidence REM fails. ZNE's `model_spread`
metric is not monotone in n_prec and is unreliable at 3 λ-points (exact
3-parameter exponential interpolation, spreads up to 1226 on a [0,1]
quantity) and trivially near-zero once the circuit is at the noise floor.
The dominant real effect is **physical placement**, not n_prec: two
n_prec=4 families, same circuit/seed, 14 minutes apart, differ only in
placement and show a 6.4× swing in raw P_correct (one placement included a
CZ edge with failed calibration). `MITIGATION_BRIEF.md` §7's
gate-fidelity-only framing should be revised to a placement-quality +
duration/T2 criterion; a concrete revision was drafted and handed to the
Director.

**Correction to task premise**: the original task brief's claim that
`device_caps.json`'s `specs` field is "still an unflattened placeholder...
not wired into placement scoring" was independently checked and found
**false** — `rigetti_qpuf_common.py:normalize_specs()` flattens it
in-memory on every `load_device_caps()` call (not persisted back to the
cache file, which is why inspecting the JSON on disk looks unflattened),
and `error_mitig.py` actively uses the flattened per-qubit/per-edge
fidelities in placement-scoring log-likelihood — confirmed by direct
runtime execution (`edge_fidelity`, `qubit_fidelity`, `readout_fidelity`
calls matched the flattened specs exactly). DDD-gap phrasing similarly
corrected: DDD *is* implemented in `submit_qpe.py` (`--ddd` flag); no task
this round was submitted with it set to anything but `none`.

No hardware was submitted or retrieved as part of this task.

---

## 2026-08-28 — Worker1 (MITIGATION_BRIEF.md §7 revision, Yon-approved)

**What**: Wrote the physicist's §7 revision (from the entry above) into
`MITIGATION_BRIEF.md`, verbatim as drafted. The old F2Q-vs-model_spread
table is kept but retitled "simulation prediction — superseded below" with
a one-line note on the real F2Q=0.988842 measurement; the new
placement-quality + circuit-duration/T2 criterion is added directly below
it as the operative viability assessment, replacing the old table as the
thing a reader should actually use.

**Why**: Yon reviewed the drafted revision (relayed via the Director) and
asked for it to be applied. Not committed — no commit requested.

**Files**:
- `braket_direct/rigetti_cepheus_20260807/MITIGATION_BRIEF.md` — §7,
  "ZNE viability is gate-fidelity-driven, not depth-driven" subsection
  (previously lines ~253-262) restructured as described above.

**Verification**: N/A — this is a documentation edit reproducing an
already-verified physicist assessment (see the entry above for the
underlying data checks: 6.4× placement-driven P_correct swing at fixed
n_prec, model_spread degeneracy at 3 λ-points up to 1226, ZNE-applicable
in only 6/26 families under the two-gate criterion). Not a new numerical
claim. §8's stale "device caps not yet fetched" caveat (also flagged by
the physicist as needing retirement) was intentionally left untouched —
out of scope for this specific, narrowly-requested edit.

No hardware was submitted or retrieved as part of this task.

---

## 2026-09-08 — DCV Worker (read-only) + Physicist (MCM feasibility audit: stored-program check, live device query, deferred-vs-genuine equivalence analysis)

**Context**: Triggered by the Step 3 prep entry above's open blocker (transpiler
DAG-reordering can silently defer a mid-circuit measurement to the end of the
circuit). This entry answers three questions raised by that finding: (1) did
this project's one completed real hardware run actually carry a mid-circuit
measurement in its submitted program, or was it already deferred; (2) what
does Braket/IonQ's own device schema and documentation say about MCM support
on Forte, independent of this project's prior assumptions; (3) if a program
is deferred rather than genuinely mid-circuit, is that physically
distinguishable at all, with or without noise. All of A-D below is read-only
investigation — no code was changed and no hardware was submitted.

**A. Confirmation from the stored hardware program (DCV Worker, read-only)**

The one completed two-stage MCM task
(`gate_limit_probe_results/0ede858e-d8ba-44a5-bdbc-14154b5aeeaa.json`,
n_prec=3, n_targ=3) retains its submitted OpenQASM under `action.source`:
1,924 lines = 1,915 gate lines then 6 consecutive measure lines (lines
1918-1923). Zero gates on any qubit appear after the first measure of
q[3..5]. So the transpile-reordering finding from Step 3 is not merely a
present-day artifact: the ONE real hardware run this project has ever
completed had all measurements at the end. No prior IonQ submission has
exercised a mid-circuit measurement even textually. The n_prec=4/5 attempts
were rejected at `CreateQuantumTask` before a program existed, so there is
nothing to check there.

**B. Live device query, Forte-Enterprise-1, 2026-09-08 (DCV Worker, read-only,
amazon-braket-sdk 1.113.0)**

No key in the device property schema states MCM support either way.
`supportedOperations`: x y z h s si t ti v vi rx ry rz cnot swap xx yy zz (no
reset, no barrier). `supportedModifiers` empty. `maximumClassicalArrays` 1.
`supportsUnassignedMeasurements` True, `requiresAllQubitsMeasurement` False.
`supportedPragmas`: the four result types plus verbatim. `nativeGateSet`
GPI/GPI2/ZZ. `supportsPartialVerbatimBox` False. The device DOES publish a
calibrated active reset duration of 150 us (`provider.timing.reset`,
`standardized.activeResetDuration`) even though reset is not in
`supportedOperations`. Forte-1 is OFFLINE with an identical capability schema
and a 50 us reset timing. Calibration drift vs. the 2026-08-28 cached
snapshot, six values, all routine: F2Q 0.9882→0.9904, SPAM 0.994→0.9964,
`updatedAt` 2026-08-27→2026-09-06. Saved untracked on the DCV box at
`braket_direct/ionq_20260601/device_properties_forte_enterprise_1_2026-09-08.json`.
Only two IonQ devices exist under this account.

**C. Documentation findings (physicist + AWS research, with sources)**

- Braket's documented mid-circuit-measurement / dynamic-circuit capability is
  IQM-only (`measure_ff`, `cc_prx`, verbatim-only, experimental). IonQ is not
  listed. Source:
  https://docs.aws.amazon.com/braket/latest/developerguide/braket-access-ionq-forte.html
- Braket's OpenQASM supported-features list includes `measure` and `barrier`,
  and states barriers "prevent gate reordering and optimizations across
  barrier boundaries. They also enforce strict temporal ordering during
  execution." It does not list `reset`; classical control (`if`) is
  LocalSimulator-only. Source:
  https://docs.aws.amazon.com/braket/latest/developerguide/braket-openqasm-supported-features.html
  — Caveat: IonQ's own `supportedOperations` (item B above) does not
  enumerate `barrier`, so acceptance on this device is unverified and would
  need a pilot before relying on it.
- IonQ's own stack states MCM/qubit-reuse/reset run as OpenQASM 3 but "today
  they execute on the simulator (mid-circuit-measurement QPU support is
  rolling out); other targets are rejected server-side." Sources:
  https://github.com/qiskit-community/qiskit-ionq and
  https://docs.ionq.com/llms-full.txt
- NOT FOUND: any statement from AWS or IonQ that measurements on IonQ
  hardware are executed in program order, or deferred, or performed at the
  end. This absence of documentation is itself the finding — there is no
  authoritative source either confirming or denying program-order execution.
- Consequence to record explicitly: CLAUDE.md Section 0's parenthetical "MCM
  + reset confirmed" for Forte-Enterprise-1 is NOT supported by any
  documentation or device capability flag found in this investigation.
  Recorded here as an unverified claim needing Yon's attention — CLAUDE.md
  itself was not and is not edited as part of this task; see TODO.md.

**D. Physicist analysis: is a deferred program physically equivalent to
genuine MCM?**

**D1. Noiselessly: EXACTLY identical, not approximately.** Since no later
operation touches the stage-1 register and there is no feedforward, every
later unitary commutes with the projector, so P(m1,m2) is unchanged — this
holds for the joint, both marginals, and the target state conditioned on m1.
Only the target's UNCONDITIONAL description at the intermediate time differs
(pure-and-entangled vs. proper mixture), distinguishable only by an operation
acting jointly on prec_a and the target, which this circuit does not
contain. Therefore no statistic computable from the returned bitstrings can
separate the two experiments: they define the same probability measure, so
every test has power exactly equal to its size. This is a proof of
impossibility, not an absence of ideas. For accuracy: this deferral argument
is WEAKER than and separate from Ghosh et al.'s Kraus-commutativity argument
(their Eqs. 8-9, 16-17), which is what gives P(m1=m2)=1 for an on-grid U — do
not conflate the two.

**D2. Under noise, NOT equivalent, and the differences run in both
directions, so neither version bounds the other.** Five sub-mechanisms, each
labeled by its evidentiary status:

- (a) Detection dead time [ESTIMATED]. T2 ~1s on Forte; detection duration is
  not published, estimated 100-200 us, plus the published 150 us reset gives
  ~250-400 us of target idling. Resulting dephasing ~1.5e-4, an order of
  magnitude below a single two-qubit gate error of 9.6e-3. This CANNOT
  explain a flip probability of 0.2-0.3. Deferred is marginally optimistic
  here, immaterially so.
- (b) Detection-light scattering onto spectator ions [MECHANISM, real but
  untested in this project's data]. 370nm resonant fluorescence in a static
  shared chain depumps neighbours — the standard trapped-ion MCM crosstalk
  mechanism, which QCCD machines avoid by physically transporting ions away
  during readout, which Forte cannot do. This channel is ENTIRELY ABSENT
  from a deferred program. It is exactly the channel this project's central
  claim is about, so the deferred runs contain zero evidence about it.
- (c) Idle entangled precision register during stage 2 [REAL, pessimistic for
  the deferred version, order-of-magnitude ESTIMATE]. After the inverse QFT,
  prec_a holds computational-basis labels, so Z-type dephasing is diagonal in
  the measured basis, commutes with the projector, and changes nothing —
  only X/Y errors matter, flipping bits of m1 and creating spurious
  disagreement. For hyperfine ions T1 is effectively infinite, so the source
  is gate-induced spectator crosstalk from stage-2 beams. Estimate: 1e-4 to
  1e-3 per neighbouring two-qubit gate, times order 1e2-1e3 stage-2
  two-qubit gates, times 15 idle prec_a qubits, gives a plausible range from
  a few percent to complete randomization of m1. **THIS IS A SERIOUS
  CONFOUND**: it is an alternative explanation for the reported p_flip
  0.2-0.3 that has nothing to do with measurement. Genuine MCM does not have
  this channel, because m1 is already classical by the time stage 2 runs.
- (d) Qubit count [PESSIMISTIC for deferred]. Deferred needs
  2·n_prec + n_targ = 31 ions where reuse needs 16. Longer chain means a
  denser lower-frequency motional spectrum, more heating, worse MS gates,
  more addressing crosstalk.
- (e) Duration [using the repo's own gate-time model]. With 130us 1Q / 970us
  2Q, the stored 1,915-gate program runs ~0.5-0.9s, i.e. at or beyond T2. Any
  result at that size is noise-floor data dominated by (c) and (e), not by
  measurement placement.

**D3. Consequences.** The QPUF fingerprint remains valid in the weak sense
that P(agree) is still a device-and-U-specific statistic, but it is a
fingerprint of a DIFFERENT instrument. Faithfulness to Ghosh et al.: NO.
Ghosh et al. state the entire quantum instrument is the QPUF (Sec. III G 2,
Def. 5), and the token protocol requires the post-measurement state to
physically exist, be handed over, held, and returned, while m0 sits in the
verifier's classical memory (Sec. III G 3). A deferred program never
produces a classical m0 mid-protocol and never produces the token state. It
reproduces the OUTPUT STATISTICS of the protocol without INSTANTIATING the
protocol. It is therefore NOT a test of the quantum-memory channel that
CLAUDE.md names as the central finding; it is a memory test of ancilla
idling instead. **Every existing "MCM disturbance → p_flip 0.2-0.3"
statement derived from these runs must be relabelled as unattributed
circuit noise, with channel (b) untested and (c)/(e) unexcluded.** This is
recorded explicitly as a negative result per CLAUDE.md §2 ("negative results
are deliverables") — this is a correction to the project's central claim,
not a side note.

**D4. Decisive experiment — design sketch only, NOT approved and NOT
scheduled.** Paired, interleaved, same day, same U seed, and crucially the
SAME qubit count in both arms (two disjoint precision registers in both, so
only measurement position varies and confound (d) is controlled). Small
n_prec, 3 or 4, otherwise confound (e) swamps the effect. Arm A:
order-preserving text (genuine attempt at mid-circuit). Arm B: deliberately
deferred. Arm C: control, target in an eigenstate of the on-grid U so the
noiseless answer is a fixed known bitstring with m1=m2 deterministic.
Primary observable P(m1=m2); the real discriminator is secondary: the
Hamming-distance distribution of m1 XOR m2 and the per-bit-position
disagreement rate. Ancilla bit flips give weight-1-dominated disagreement
roughly uniform across bit positions; eigenvector switching gives
disagreements concentrated on patterns set by the k_j structure of U;
depolarizing-like MCM damage gives near-uniform m2 at fixed m1. About 2,000
shots per arm detects a 0.05 difference in agreement at roughly 3 sigma.
**Pre-registered gate condition**: retrieve the executed program from task
metadata and confirm arm A retained its mid-circuit measure. If that cannot
be confirmed, the experiment concludes NOTHING about MCM and must be
reported as "contrast not achievable on this submission path" — NOT as a
null result. Conflating "couldn't test it" with "tested it and found no
effect" would itself be a documentation error.

**E. Also recorded in this pass**

- **F2Q labelling reconciliation** (closes the open item flagged in the Step
  4 entry above): `ionq_mitigation_common.py` line 109 has held F2Q=0.9882
  since the August campaign, and the cached snapshot agrees (`updatedAt`
  2026-08-27). The live query in item B above returned 0.9904 (`updatedAt`
  2026-09-06). So 0.9904 is current, 0.9882 is stale. The on-grid pipeline's
  labelling in the Step 4 entry was correct. The shared constant in
  `ionq_mitigation_common.py` was NOT changed — the Rigetti and single-stage
  campaigns depend on it, and moving the default would silently shift their
  baselines.
- **Gate-counting environment**: the `braket-arm` conda environment
  (`/Users/yonnoylee/miniforge3/envs/braket-arm`, qiskit 2.3.0) is the only
  trusted gate counter, having reproduced the AWS-confirmed 1,915 and 2,583
  anchors exactly (Addendum 3, on-grid baseline entry above); the Mac's
  default qiskit 2.0.0 over-counts by roughly 2.4x. Any gate count used for
  a submission decision must come from braket-arm.

**Status**: read-only investigation. No code was changed, no hardware was
submitted or retrieved as part of this task. See `TODO.md` for the
carryover items this raises, including the flag for Yon's attention on
CLAUDE.md Section 0's "MCM + reset confirmed" parenthetical.
