# TODO — Session Recap and Carryover (as of 2026-09-08)

This document **supersedes** the 2026-08-28/31 recap — it is a rewrite, not
an appendix to it. Genuinely still-open items from that recap are carried
forward below; items it listed as open that are now resolved are marked
resolved with a pointer to the `docs/LOG.md` entry that resolved them.

Today's (2026-09-08) session had two threads on the IonQ side: (1) a 5-step
on-grid QPE-QPUF pipeline (noiseless sim → noisy sim → hardware-submission
prep → mitigation sim; step 5, mitigation on real hardware, not started),
and (2) a read-only feasibility audit of whether this project's mid-circuit
measurement (MCM) claims are actually supported by any hardware program,
device schema, or documentation — which produced this session's most
consequential finding. See `docs/LOG.md`'s six 2026-09-08 entries (on-grid
baseline + 3 addenda, Step 1, Step 2, Step 3 prep, Step 4, and the MCM
feasibility audit) for full detail; this document does not repeat their
content, only their carryover consequences.

---

## 0. Critical carryover — read this first

1. **No prior IonQ run has exercised a mid-circuit measurement — not even
   textually.** The one completed real Forte-1 hardware task
   (`gate_limit_probe_results/0ede858e-...json`, n_prec=3/n_targ=3) retains
   its submitted OpenQASM, and it shows 1,915 gate lines followed by 6
   consecutive measure lines — every measurement at the end. The existing
   **p_flip ≈ 0.2–0.3 figures are now UNATTRIBUTED CIRCUIT NOISE, not
   evidence of MCM disturbance**, pending the decisive experiment in item 6
   below. (`docs/LOG.md`, "MCM feasibility audit" entry, §A.)

2. **The order-preserving export (Step 3's fix) is a PREREQUISITE for any
   future MCM claim, and even it is NOT SUFFICIENT.** Program-order
   assembly (piecewise transpile, no further global `transpile()`) stops
   Qiskit's DAG-reordering bug from destroying MCM ordering pre-submission,
   but whether IonQ/Braket's server-side compiler preserves genuine
   mid-circuit execution once it receives that program is still
   **unverified** — and today's audit adds a formal reason this matters:
   noiseless statistics are provably identical whether the measurement is
   genuine or deferred to the end (`docs/LOG.md` MCM audit entry, §D1), so
   no noiseless check, however careful, can ever settle this on its own.

3. **Notebook decode bug, not fixed, taints a specific code path.**
   `braket_direct_analysis.ipynb` cell 3's `parse_outcome` single-flat-
   register fallback branch omits a required per-stage bit reversal and
   mis-decodes m1/m2 (known-answer test: gives (14,14) instead of the
   correct (7,7)). Any past hardware analysis that ran a single-flat-
   register circuit through that fallback branch should be treated with
   suspicion until it's fixed. (`docs/LOG.md`, Step 1 entry.) The notebook
   was deliberately **not** edited, per instruction to leave code
   untouched outside the specific task at hand.

4. **Gate counts must come only from the `braket-arm` conda environment.**
   The local Mac's default `qiskit==2.0.0` over-counts transpiled gates by
   roughly 2.4x relative to `braket-arm`'s `qiskit==2.3.0`, which is the
   only environment that has reproduced the AWS-confirmed 1,915/2,583-gate
   anchors exactly. Any gate count feeding a submission decision — for the
   on-grid pipeline or anything else — must be taken from `braket-arm`.
   (`docs/LOG.md`, on-grid baseline entry, Addendum 3; reaffirmed in the
   MCM feasibility audit entry, §E.)

5. **CLAUDE.md §0's "MCM + reset confirmed" parenthetical for
   Forte-Enterprise-1 is UNVERIFIED by any documentation or device
   capability flag found in this investigation.** A live device query
   (amazon-braket-sdk 1.113.0, 2026-09-08) found no schema key stating MCM
   support either way, and Braket's own documentation lists dynamic-circuit
   / mid-circuit-measurement support as IQM-only, not IonQ. IonQ's own
   stack states MCM "today executes on the simulator... other targets are
   rejected server-side," with rollout to QPU support in progress. No
   source, on either side, states or denies that IonQ hardware executes
   measurements in program order. **This is flagged here for Yon's
   attention and is explicitly NOT edited into CLAUDE.md itself** — per
   this session's instruction, CLAUDE.md is log-only for this discrepancy.
   (`docs/LOG.md`, MCM feasibility audit entry, §B–C.)

6. **The paired A/B/C decisive experiment is the RECOMMENDED next hardware
   step, ahead of a raw Step-3-style submission — design sketch only, NOT
   approved, NOT scheduled.** A plain Step-3-style submission, even with
   program-order-preserving export, cannot by itself attribute any observed
   effect to MCM: idle-precision-register spectator crosstalk during stage
   2 (confound (c), order-of-magnitude estimated at "a few percent to
   complete randomization of m1") is a live rival explanation for the
   reported p_flip that a single-arm submission cannot rule out. The
   decisive design instead pairs an order-preserving arm, a deliberately
   deferred arm, and an eigenstate-control arm, same day, same U seed, same
   qubit count, and reads out the Hamming-distance/per-bit disagreement
   pattern (not just P(m1=m2)) as the actual discriminator between
   mechanisms. (`docs/LOG.md`, MCM feasibility audit entry, §D2(c), §D4.)

7. **Steps 3 (real hardware submission) and 5 (mitigation on real
   hardware) both remain gated on Yon's explicit, per-submission
   go-ahead.** This project has not submitted anything to real hardware
   today. Per CLAUDE.md §2, this holds no matter how small or "just a
   test" a circuit is.

8. **Nothing was committed today.** See §4 below for the full file list
   touched across today's session, for Yon's review before any commit
   decision.

---

## 1. What happened this session, what was wrong, what should change

### On-grid pipeline (Steps 1–4)

Built and noiselessly/noisily verified a 5-step on-grid QPE-QPUF pipeline
(`on_grid_qpuf_pipeline.py`) using a computational-eigenbasis on-grid U
(`random_diagonal_on_grid_unitary`), which gives an exact P(m1==m2)=1.0
noiseless baseline — removing the off-grid-eigenphase baseline ambiguity
that was the single largest physics-methodology finding of the prior
(2026-08-28) session. Noisy-simulation results (Step 2) and error-mitigation
results (Step 4) are recorded in `docs/LOG.md`; none of this has touched
real hardware yet.

Three more instances of the Step-3 "transpiler silently defers/reorders
measurement" bug class were found and designed around during Step 4
(fold/DDD applied per-piece rather than to the assembled circuit; a
`transpile()` call inside `mitig_analysis.py`'s `_run()` skipped for
already-correct-basis circuits; a clbit-renumbering bug in
`_reattach_measurements` when applied to an isolated circuit piece). All
three fixes live in `on_grid_qpuf_pipeline.py` only — `error_mitig_ionq.py`
and `mitig_analysis.py`, shared with the Rigetti/Cepheus and single-stage
IonQ campaigns, were **not modified** by this work (see §4's discrepancy
flag below, though — git status shows these files as locally modified from
some other, undocumented source).

### MCM feasibility audit (new this session)

See §0 items 1, 2, 5, 6 above — this is the session's headline finding and
is not repeated here.

### Wrong assumption already on record, now further undercut

The 2026-08-28 session's correction that "any prior p_flip estimate
assuming a 1.0 baseline is not valid" is now superseded by a more
fundamental problem: regardless of baseline correctness, the underlying
hardware data was never taken with a genuine mid-circuit measurement in the
first place (§0 item 1). Both corrections stand; the second does not
retroactively validate or invalidate the first, they are independent
defects in the same body of prior p_flip claims.

### Open question, carried forward unresolved

Whether to write up the exploratory n_prec=3/n_targ=3 gate-limit-probe
m1-vs-m2 result as a preliminary/negative finding, or hold off — recorded
in the prior recap as a decision point for Yon, not resolved there. Given
today's finding that this specific data point never had a genuine MCM to
begin with, if it is written up, it must be framed accordingly (unattributed
circuit noise on a deferred-measurement program), not as an MCM result.

---

## 2. Physics / conceptual questions and tasks

- **p_flip re-baselining — superseded, not resolved.** The prior session's
  "recompute p_flip against a per-instance simulated baseline" task is
  moot for any data taken to date: per §0 item 1, none of it reflects a
  genuine MCM regardless of baseline. The on-grid pipeline (Steps 1–4)
  sidesteps the baseline-ambiguity problem going forward by construction
  (P(m1==m2)=1.0 exact for on-grid U), but has not yet been run on real
  hardware.
- **Isolated MCM-disturbance test — status updated.** The on-grid U
  construction is implemented and noiselessly verified (P(m1==m2)=1.0 to
  machine precision, all tested cases). Step 3 implemented a program-order-
  preserving export as a partial mitigation for the deferral bug, but per
  today's audit (§D1) this is necessary, not sufficient — noiseless checks
  structurally cannot prove genuine mid-circuit execution occurred. The
  actual test design that could settle it is the D4 paired-arm experiment
  (§0 item 6) — not approved or scheduled.
- **Can MCM-specific disturbance be separated from ordinary depth/gate-error
  scrambling at current depths? — substantially addressed analytically,
  still open experimentally.** Today's audit's §D2 lays out five ranked,
  evidence-labeled sub-mechanisms (detection dead time [estimated,
  negligible]; detection-light scattering onto spectators [real mechanism,
  untested by any of this project's data]; idle-entangled-precision-register
  crosstalk [real, a serious confound, order-of-magnitude estimated at a
  few percent to full randomization]; qubit count; circuit duration at/past
  T2). This is a substantially more precise answer than the prior session's
  general "circuit was too deep" finding, but it is analytical — no
  experiment has yet discriminated between these mechanisms. The D4 design
  is the proposed way to do so.
- Decide the disposition of the 9 orphaned, never-retrieved real Forte
  hardware jobs from the original May/June submission window
  (`braket_direct/ionq_20260601/job_results/job_log.txt`). Unchanged, still
  open.
- Revisit the Rigetti Cepheus placement-driven finding for the paper
  narrative (`MITIGATION_BRIEF.md` §7 already revised, applied not
  committed) — check the rest of the paper narrative for the same stale
  gate-fidelity-only assumption. Unchanged, still open.
- Rigetti Cepheus DDD data: zero DDD tasks collected to date despite
  `--ddd` being implemented in `submit_qpe.py`. Unchanged, still open.
- IonQ Debias comparison currently scoped to a single family (D1) because
  `ionq_noise_mitig.py`'s two-stage circuit path only builds random-Haar-
  target circuits. Unchanged, still open.
- Two still-open decisions on the IonQ REM/DDD/ZNE mitigation campaign
  (`braket_direct/20260828_ionq_error_mitigation/`): (1) family D2
  (n_prec=4, n_targ=3, haar) is statistically too weak at 2,000 shots
  (2.1σ from the uniform floor) — pre-register as a floor-confirmation null
  result (recommended), bump to 5,000 shots (+$240), or drop; (2) whether
  to extend the Debias arm beyond D1. Unchanged, still open.

---

## 3. Coding tasks

- Fix `braket_direct/rigetti_cepheus_20260807/run_all_families.py` lines
  152/261: `int()` truncation of fractional ZNE scales silently collapses
  λ=1.5 into the λ=1 bucket. **Status: unconfirmed.** The prior recap
  flagged this as unfixed pending a status confirmation from Worker1; no
  `docs/LOG.md` entry from today or before addresses it. Needs an explicit
  check before any fractional-λ Cepheus run.
- Fix `braket_direct/20260828_ionq_error_mitigation/submit_ionq_mitigation.py:74`:
  `ZNE_SCALES = [1, 1.5, 2, 2.5, 3]` has an inconsistent literal int `1`
  among floats, same bug class as above, currently masked by a downstream
  re-cast. **Status: unconfirmed** — no LOG.md entry documents this being
  fixed. (Git status shows this file as locally modified; see the
  discrepancy flag in §4 — the modification's content is not described in
  any LOG.md entry, so it cannot be assumed to be this fix.)
- Verify the `MAX_GATES_PER_CIRCUIT = 2000` guard is applied consistently
  to every code path that can submit to IonQ. This is now more concrete
  than when first flagged: `on_grid_qpuf_pipeline.py`'s `--stage submit`
  path (built in Step 3) is exactly the kind of "future two-stage/probe
  script" the original item anticipated, and no LOG.md entry confirms this
  guard was checked against it. Still open.
- Design and implement a circuit change that makes MCM deferral impossible
  (classically-conditioned operation on the stage-1 result, or a
  reset-based ancilla-reuse redesign). **Status, updated**: Step 3's
  program-order-preserving assembly is a partial mitigation, applied and
  verified for local-simulator re-import. It is not a full resolution —
  per today's audit (§D1), no purely-noiseless fix can ever be sufficient,
  because deferred and genuine MCM are provably indistinguishable
  noiselessly. Full resolution requires either documented IonQ/Braket
  compiler behavior (not found — see §0 item 5) or the D4 differential
  hardware experiment.
- Investigate and clean up the stray `.claude/docs/` nested directory
  (separate from the real `docs/` at repo root). Unchanged, still open.
- Retrieve or formally close out the 9 orphaned IonQ hardware jobs (a
  `checkRetrieve.py` run against
  `braket_direct/ionq_20260601/job_results/job_log.txt`, or an explicit
  decision to abandon them). Unchanged, still open.
- Optional, low-cost (~$8–16), not yet actioned: two confirmation
  submissions at n_targ=2 (n_prec=11 expected accepted at 1,910 gates,
  n_prec=12 expected rejected at 2,134 gates) as a 4th/5th gate-count
  confirmation point. Unchanged, still open, purely confirmatory.
- Fix `braket_direct_analysis.ipynb` cell 3's `parse_outcome` single-flat-
  register fallback branch (see §0 item 3). Unchanged, still open, not
  fixed in the notebook per explicit instruction.
- **Resolved — Haar-random target docstring mislabel.** The two-stage
  circuit's target-state generator was described as "Haar-random" but
  builds a random product state. **Fixed 2026-09-08** in
  `submit_QPUF_ntarg.py`'s docstring. (`docs/LOG.md`, on-grid baseline
  entry.)
- **Resolved — F2Q labelling conflict.** The Step 4 entry flagged
  conflicting labels for F2Q=0.9882 ("today's calibration value" vs.
  "Step 2's stale original value"). Today's live device query resolved
  this: 0.9904 is current (`updatedAt` 2026-09-06), 0.9882 is stale
  (`updatedAt` 2026-08-27). The on-grid pipeline's labelling was correct.
  `ionq_mitigation_common.py`'s shared F2Q=0.9882 constant was
  **deliberately left unchanged** — Rigetti and single-stage IonQ
  campaigns depend on it and moving the default would silently shift their
  baselines. (`docs/LOG.md`, MCM feasibility audit entry, §E.)
- Outstanding commits, unchanged and now larger: Worker1's Cepheus
  deliverables (placement-grouping bugfix, `run_all_families.py`, 38
  `analysis_summary_*.json` files, plots, `MITIGATION_BRIEF.md` §7
  revision) plus everything from today's on-grid pipeline and MCM
  feasibility audit are applied locally but not committed. See §4.

---

## 4. Files touched today (2026-09-08), for review before any commit

Reconstructed from `docs/LOG.md`'s six 2026-09-08 entries, cross-checked
against the (truncated) git status snapshot for completeness.

**New**:
- `braket_direct/ionq_20260601/two_stage_on_grid_baseline.py`
- `braket_direct/ionq_20260601/on_grid_qpuf_pipeline.py`
- `braket_direct/ionq_20260601/on_grid_qpuf_plots.py`
- `braket_direct/ionq_20260601/on_grid_results/2026-09-08/` (all contents:
  Step 1/2/4 plots and JSON, `submit/on_grid_seed9_submission.qasm`,
  `submit/on_grid_seed9_manifest.json`, `submit/on_grid_seed9_ideal_joint.json`,
  `mitigate/mitigate_results.json`, `mitigate/on_grid_mitigation_circle_plots.png`)
- `braket_direct/ionq_20260601/gate_limit_probe_results/plots/on_grid_two_stage_ideal_joint.png`
- `braket_direct/ionq_20260601/device_properties_forte_enterprise_1_2026-09-08.json`
  (saved untracked on the DCV box — not necessarily present in every local
  checkout of this repo)
- `docs/LOG.md`'s six 2026-09-08 entries (five from the coder/verifier
  chain, one from this writer pass documenting the MCM feasibility audit)
- This file (`TODO.md`, full rewrite)

**Modified**:
- `braket_direct/ionq_20260601/submit_QPUF_ntarg.py` (added
  `on_grid_unitary`, `rotated_on_grid_unitary`,
  `product_rotated_on_grid_unitary`, `random_diagonal_on_grid_unitary`,
  `build_qpe_stage_product`, `eigenstate` kwarg; fixed the docstring
  mislabel)
- `braket_direct/ionq_20260601/decode_mcm_qpuf_result.py` (`ideal_joint`
  generalized; `circular_distance_pmf` dtype/assert generalized)

**Explicitly NOT modified today** (per `docs/LOG.md`'s Step 4 entry, stated
directly): `error_mitig_ionq.py`, `mitig_analysis.py`.

**Unreconciled discrepancy — flagged, not resolved by this writer pass**:
the git status snapshot available to this session shows
`braket_direct/20260828_ionq_error_mitigation/error_mitig_ionq.py`,
`ionq_mitigation_common.py`, `mitig_analysis.py`, `submit_ionq_mitigation.py`,
and `braket_direct/ionq_20260601/probe_gate_limit.py` as locally modified,
but no `docs/LOG.md` entry from today describes edits to any of these five
files — Step 4's entry explicitly states the opposite for two of them. This
could be a prior, undocumented session's uncommitted diff, or an omission
in today's reporting. Either way, **do not assume the content of these
diffs from this document** — the Director/Coder should check `git diff` on
these five files specifically before any commit.

No hardware was submitted or retrieved as part of today's session. No files
were committed.
