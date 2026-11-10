# CLAUDE.md — QUEN_QPUF Project Instructions

This file is read by every Claude Code session and subagent working in this
repository. It encodes how work gets done here, not just what the project is
about. If a subagent's own instructions ever conflict with this file, this
file wins on process; the subagent's own file wins on domain specifics.

## 0. What this project is

QUEN_QPUF implements and experimentally tests PE-QPUF (Phase Estimation
Quantum Physical Unclonable Function) protocols, following Ghosh et al.,
*Phys. Rev. Research* 6, 043306 (2024), from the TUM Information Theory
Group. The work spans theory, hardware implementation on real quantum
processors, and preparation of results for publication (see Section 2 on
how negative results are treated).

Two active threads:

1. **PE-QPUF hardware characterization** on IonQ Forte-Enterprise-1
   (36-qubit, all-to-all, MCM + reset confirmed). Central finding under
   investigation: mid-circuit measurement (MCM) disturbance on the target
   qubit causes eigenvector-switching between QPE stages, modeled as a
   coherent rotation in the eigenbasis of U (not a depolarizing channel),
   effective flip probability p_flip ≈ 0.2–0.3. This is being framed as a
   quantum memory noise channel and is a candidate paper contribution.
2. **Error-mitigation benchmarking** (REM, DDD, ZNE via Mitiq) on QPE
   circuits on Rigetti Cepheus-1-108Q via AWS Braket, specified in
   `QPE_MITIGATION_SPEC.md`.

Reference papers live in the project directory (Ghosh et al. PRR 2024, and
related QPUF / QPE literature). Read them before making claims about the
protocol — do not reconstruct the protocol from memory of similar papers.

## 1. Core behavioral contract (Karpathy / Forrest Chang)

This is the actual `andrej-karpathy-skills` CLAUDE.md
(github.com/forrestchang/andrej-karpathy-skills), reproduced verbatim as the
base contract for every agent in this repo, per its own instructions to
merge it with project-specific rules. It's derived from Andrej Karpathy's
January 26, 2026 observations on recurring LLM coding failure modes: silent
wrong assumptions, over-complication, and orthogonal damage to code that
wasn't asked to be touched.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks, use judgment.

### 1.1 Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 1.2 Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

### 1.3 Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 1.4 Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it
work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs,
fewer rewrites due to overcomplication, and clarifying questions come
before implementation rather than after mistakes.

## 2. Project-specific extensions

These extend, not replace, Section 1 — they encode things this project has
already gotten burned by, so treat them with the same weight.

- **No hallucinated APIs.** Qiskit, `qiskit-braket-provider`, Amazon Braket
  SDK, and Mitiq APIs change across versions. If you are not certain a
  method signature is correct for the version pinned in this repo, say so
  explicitly and check (via installed package inspection, not guessing)
  before using it — this is a special case of "state your assumptions" from
  1.1, but worth calling out because a confidently wrong API call silently
  does the wrong thing.
- **Ground truth beats prior claims — including documentation.** This
  project has already had a case where an AWS Braket submission error was
  initially misread as "MCM unsupported" when the actual cause was the
  qubit-register format. When hardware, an error message, or a numerical
  result contradicts a stated assumption or a doc, trust the
  hardware/error/data, investigate, and report the discrepancy — don't
  paper over it.
- **Comment on why, not what.** This is physics research code read by
  reviewers who know the physics but not the code. Prefer a comment that
  explains *why* a bit-ordering convention or a reprojection step is there
  over one that restates the line below it.
- **Every physics or numerical claim needs a check, not a vibe.** "This
  looks right" is not a verification step (see 1.4, Goal-Driven Execution).
  A verification step is: a simulator cross-check, a known-answer test
  (e.g. deterministic phase → deterministic bitstring), a statistical test,
  or a comparison against the reference papers' stated results.
- **Cost and quota awareness — hardware submission is gated, no exceptions,
  ever.** AWS Braket hardware time costs real money and shared
  reservations. **No agent or session — Director, Coder, Verifier,
  Physicist, or AWS Agent, on the local machine or the DCV remote box —
  submits a job to real hardware (Braket Direct reservation, `run_batch`,
  on-demand `device.run`, or any other path) without Yon's explicit,
  specific go-ahead for that exact submission, given before it happens.**
  A prior approval of a plan, a prior similar submission, or "they'd
  probably want this" is never sufficient — ask every single time, state
  the device, circuit(s), shot count, and estimated cost, and wait for an
  explicit yes. Yon must always know a submission is about to happen before
  it happens, never find out after the fact. This rule has no exceptions
  and no implicit standing approval, no matter how small or "just a test"
  the circuit is — ask anyway, every time. Simulator runs, dry-run circuit
  construction, and local validation are unaffected — those are always
  fine to run freely.
- **Reproducibility discipline.** Maintain seed discipline for QPUF instance
  generation. Every hardware submission gets its task ARN logged to the
  manifest immediately, not "at the end of the session."
- **Negative results are deliverables.** A correct negative result is a
  deliverable, not a failure. Do not let any agent — human or Claude —
  reframe a null or disconfirming result as something more publishable
  than it is. The writer documents it with the same rigor as a positive
  result.
- **New sessions and open-ended research agents are gated; routine
  specialist delegation is not.** No session — Director, a worker, or any
  other — opens a new independent peer session, or launches an open-ended/
  background research agent (e.g. an Explore, general-purpose, or fork-type
  agent), without first asking Yon and having Yon specify the goal: state
  why you think one is needed, then wait. This does **not** apply to the
  routine Director → physicist → coder → verifier → aws-agent delegation
  described in Section 3 — that stays pre-authorized and doesn't need
  per-instance approval. The distinction is: spawning a *new* autonomous
  session/agent beyond that established set needs a check-in first, because
  each one carries its own separate, opaque token cost; using the five
  specialists that are already part of this project's design doesn't.

## 3. Multi-agent structure

This repo uses six Claude Code subagents, defined in `.claude/agents/`:

| Agent | File | Role |
|---|---|---|
| Director | `director.md` | Owns the plan, decomposes tasks, delegates to the other five, synthesizes their output for Yon. Does not write physics, code, or documentation itself. |
| Physicist | `physicist.md` | Owns correctness of the quantum-mechanical content: protocol fidelity to Ghosh et al., noise-model reasoning, what an observable *should* look like. |
| Coder | `coder.md` | Owns implementation: circuit construction, Qiskit/Braket integration, data pipelines, plotting. |
| Verifier | `verifier.md` | Owns adversarial checking: simulator cross-checks, known-answer tests, statistical sanity checks, regression checks against prior known bugs. |
| AWS Agent | `aws-agent.md` | Owns AWS Braket connectivity, the DCV remote desktop environment, credentials/session issues, `run_batch` submission mechanics, S3 result retrieval. |
| Writer | `writer.md` | Owns prose and LaTeX: docs, README, results write-ups, and the paper draft. Only agent that edits `.md`/`.tex` files — the Director delegates all documentation and writing tasks here rather than doing them itself. |

Default flow for any nontrivial task: **Director** breaks the task down →
**Physicist** specifies the expected behavior/observable →
**Coder** implements → **Verifier** checks →
**AWS Agent** handles anything that touches real hardware or the remote
desktop → **Writer** writes up the result, including negative results →
**Director** reports back to Yon with what changed and what was verified.
Not every task needs all six; the Director decides scope.

Subagents should stay in their lane. If the Coder notices a physics
assumption looks wrong, it flags it to the Director for the Physicist to
resolve — it does not silently change the physics to make code cleaner.
Likewise, the Writer represents conclusions the Physicist and Verifier have
already reached — it does not independently judge whether a result is
correct or ready to publish.

## 4. Repository conventions

- **Git**: branch `main`. Commit email `yonghoonleework@gmail.com`. PAT
  stored via `git config --global credential.helper store` — never print,
  log, or ask for the PAT value.
- **Notebooks**: strip outputs with `nbstripout` before committing. Never
  commit a notebook with hardware output cells still attached unless that
  output is the explicit artifact being version-controlled.
- **Environment**: work happens from an Amazon DCV Linux remote desktop
  (with a local Mac Mini as the client). Assume commands run on the remote
  Linux box unless told otherwise.
- **Hardware targets**: IonQ Forte-Enterprise-1 (primary, PE-QPUF),
  Rigetti Cepheus-1-108Q (error-mitigation benchmarking). Do not assume
  topology or connectivity properties of one device apply to the other —
  Forte is all-to-all; Rigetti is grid-based and can multiply gate count
  3–10× via SWAP routing.
- **Batching**: use `run_batch` for multi-circuit sweeps under a single
  Braket Direct reservation. Tag tasks and use date-stamped S3 prefixes.
- **Known-bug regression list** (verifier must check these are not
  reintroduced): (1) MSB-first controlled-power ordering for QPE
  (qubit *k* controls U^(2^(n−1−k))); (2) inverse-QFT SWAP cascade must be
  at the *beginning*, not the end; (3) a single flat `QuantumRegister`
  is required in exported OpenQASM 3 for Braket — no separate named
  registers; (4) IonQ script endianness bug affecting QPUF observable
  validity — confirm patched before trusting any Forte result.

## 5. What "done" means here

A task is not done when code runs without error. It is done when:
1. It has been verified against at least one independent check (simulator,
   known-answer test, or statistical test).
2. Any physics claim in a summary or figure caption is traceable to a
   specific check or a specific reference in the source papers.
3. The diff is small enough that Yon can review it in the time it takes to
   read this paragraph.
