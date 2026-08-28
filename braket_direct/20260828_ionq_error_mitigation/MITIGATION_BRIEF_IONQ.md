# Error-Mitigation Implementation Brief -- IonQ Forte-Enterprise-1

Self-contained context for discussing this implementation with someone (or
something) that has not seen the codebase. Structured to mirror
`braket_direct/rigetti_cepheus_20260807/MITIGATION_BRIEF.md`, but this is a
**brand-new campaign**: nothing below reports a hardware result, and Section
7 says so explicitly rather than borrowing the Rigetti numbers.

---

## 1. What we are building

Benchmarking three error-mitigation techniques -- **REM**, **DDD**, **ZNE**
-- on **Quantum Phase Estimation** running on **IonQ Forte-Enterprise-1** via
**AWS Braket**, submitted **ON-DEMAND** (no Braket Direct reservation for
this campaign -- contrast with the Rigetti campaign, which runs under a
booked reservation window). It is a characterization project: the
deliverable is how each technique's benefit scales with precision register
size and depth, plus an honest account of where each fails -- including the
possibility that a technique's apparent benefit or null result on this
device is an artifact of server-side recompilation rather than a physical
effect (Section 9.1).

**Environment:** same stack as the Rigetti campaign -- Python 3.13.13,
qiskit 2.4.1, amazon-braket-sdk 1.117.2, qiskit-aer 0.17.2, numpy 2.4.4,
scipy 1.17.1. Mitiq **not** installed (same two reasons as the Rigetti
brief's Section 4: it will not install cleanly under Python 3.13, and its
qiskit frontend round-trips through cirq/QASM2, which would be lossy here
too).

**Device facts that shape everything:**
- Native gate set (Braket-level) `rz` / `rx` / `rxx`; IonQ's own account
  (`submit_QPUF_ntarg.py:312-314`, `submit_test.py:104`) is that this is
  re-synthesised server-side into native `gpi`/`gpi2`/`zz`.
- **All-to-all connectivity.** Every qubit pair has a direct 2q gate. There
  is no coupling map, no SWAP routing, no chiplet partition, and no
  placement decision to make -- the entire placement/routing layer that
  dominates `rigetti_cepheus_20260807/error_mitig.py` does not exist on the
  IonQ side of this repo and is not reproduced here.
- **Mid-circuit measurement is available** on Forte-Enterprise-1 (behind
  `EnableExperimentalCapability()`), unlike Rigetti. This campaign does
  **not** use it: the primary circuit is single-stage QPE with one terminal
  measurement, chosen for the same reason the Rigetti campaign chose it --
  a directly computable ground truth. See Section 9.1's blind-spot note for
  why MCM and mitigation-technique benchmarking are best kept separate for
  now.
- **Whether IonQ's server-side compiler preserves our exact gate sequence is
  UNCONFIRMED.** This is the single biggest difference from the Rigetti
  brief, which could state its verbatim requirement as a device fact. See
  Section 9.1 -- it is the load-bearing open question of this entire
  campaign and is reproduced there **verbatim**, as written by the
  physicist, per this campaign's own process requirements.

**Circuit:** single-stage QPE, identical construction to
`rigetti_qpuf_common.build_qpe_circuit` / `submit_qpe.py`: precision qubit
*k* controls `U^(2^(n_prec-1-k))` (MSB-first), then inverse QFT, then one
terminal measurement of the precision register. Qubits used = `n_targ +
n_prec`. Primary benchmark instance is an on-grid phase: `U` diagonal with
an exact `|1...1>` eigenstate carrying phase `phi = 1/8`, so noiseless QPE
puts **all** weight in bin `phi * 2^n_prec` and any spread is measured
error.

---

## 2. Code layout

| file | role |
|---|---|
| `ionq_mitigation_common.py` | circuit builders (Haar sampling, `build_qpe_stage`/`build_qpe_circuit` -- single-stage QPE only, matching the Cepheus campaign; the two-stage QPUF protocol is a separate thread, out of scope here), IonQ timing/fidelity constants, OpenQASM3 export, device-caps loading |
| `error_mitig_ionq.py` | `ErrorMitigation` class -- DDD, ZNE folding, REM calibration circuits, verification. No placement/routing (Section 1: all-to-all) |
| `mitig_analysis.py` | estimator half -- P_correct, REM inversion, ZNE extrapolation, bootstrap CIs, noisy-sim validation |
| `submit_ionq_mitigation.py` | interactive, on-demand submission (bare `device.run`, no reservation) |
| `checkRetrieve.py`, `cancelQueue.py`, `query_device_caps.py` | retrieval / queue-management / device-capability probes |

---

## 3. The contract split (same as the Rigetti campaign)

The original intent was "every mitigation is a function: circuit in, circuit
out." Only half the techniques fit that shape, and forcing the other half
into it would be a lie -- identical reasoning to
`rigetti_cepheus_20260807/error_mitig.py` Section 3 of that brief, so it is
not re-derived here:

| technique | circuit -> circuit? | extra hardware tasks |
|---|---|---|
| DDD | **yes** | 0 |
| ZNE | one circuit *per lambda* | N-1, plus an extrapolation fit |
| REM | **no -- adds zero gates** | 2 calibration circuits + classical inversion |

REM leaves the submitted circuit byte-for-byte unchanged; the mitigation is
an inversion applied to the *counts*. ZNE's `fold()` is genuinely
circuit->circuit, but a single folded circuit is not a mitigated result.

Consequence for structure: `error_mitig_ionq.py` builds circuits,
`mitig_analysis.py` consumes counts -- same split as Rigetti, minus the
placement column (there is no placement stage here at all, not merely one
that is "always on").

---

## 4. Implementation decisions and rationale

**No chiplet/placement/routing layer, by design, not by omission.**
Rigetti's `ErrorMitigation` carries `CHIPLET_SIZE`, `chiplets()`,
`select_best_qubits()`, an ISA-calibration-driven edge/qubit-fidelity
scorer, and a SABRE-routing candidate search, because Cepheus-1-108Q is a
fixed lattice where *which* qubits a circuit lands on changes the SWAP
count by a large factor. Forte-Enterprise-1 is all-to-all: every
controlled-`U^k` and every inverse-QFT rotation is already a direct 2q gate
on whatever logical qubits the circuit names. Porting that layer over would
be dead code standing in for a constraint that does not exist on this
device, so `error_mitig_ionq.py` drops it entirely.

**DD pulse math is reused unchanged.** `X = rx(pi)`, `Y = rz(-pi/2) rx(pi)
rz(pi/2)` is basis-agnostic and directly expressible in `IONQ_BASIS =
['rz', 'rx', 'rxx', 'measure']`. Whether the sequence *survives* IonQ's
re-synthesis into `gpi`/`gpi2` unchanged is a separate, open question --
Section 9.2.

**The `rz` timing convention is NOT copied from Rigetti.** Rigetti's
`error_mitig.py` hard-codes `rz = 0` (virtual-Z frame update, zero physical
duration). IonQ's own existing cost estimator
(`noise_mitigation/ionq_noise_mitig.py`) charges every `rz` the full
`ONE_QUBIT_TIME_S` instead. The two cannot both be right, and which one is
correct changes a DDD insertion-window decision qualitatively (an XY4
sequence fits inside a 970 us window with `rz=0` and does not fit at all
with `rz` charged). `error_mitig_ionq.py` defines `RZ_TIME_S` as a named,
explicitly-commented constant defaulting to "charge it" (matching this
project's existing IonQ cost model, for self-consistency), with a comment
pointing at Section 9.2 and stating the choice is not resolved. See
"Specific implementation guidance" that produced this design, preserved in
the commit history of this campaign's creation.

**ZNE folding supports fractional lambda by default**, unlike the Rigetti
QPE submission script's `fold_global` (odd-integer-only). There is no
verbatim requirement gating this here the way there is on Rigetti (because
none is confirmed necessary *or* sufficient -- Section 9.1), so there is no
reason to restrict to the coarser odd-integer ladder; `ErrorMitigation.fold`
with `method="random"` (stratified 1q/2q pool folding, same algorithm as
Rigetti's `error_mitig.py`) is used directly.

**Composition order (fixed):** transpile to native gates (once) -> DDD
insertion -> [hardware] -> REM -> P_correct -> ZNE extrapolation. Never
re-transpile at `optimization_level > 0` after DDD/folding -- see Section
9.1, item 3 of "what would settle it", for why this is not merely a style
preference: this project's own IonQ pipeline already transpiles at
`optimization_level=1` elsewhere, and a second such pass after folding would
cancel the very thing being tested.

---

## 5. Bugs and regressions carried over from the Rigetti campaign

### 5.1 Inverse-QFT endianness -- already fixed on this campaign's code path
The Rigetti brief documents this as its most serious bug (its Section 5.1);
CLAUDE.md's known-bug list carries it as item 4, specifically calling out
the IonQ scripts as the ones that had NOT been patched as of the brief's
writing. **This campaign's `ionq_mitigation_common.build_qpe_stage` is
written with the fix already in place** (`prec[::-1]` on the inverse-QFT
append), inheriting the patch applied earlier the same day to
`noise_mitigation/ionq_noise_mitig.py` and
`braket_direct/ionq_20260601/submit_QPUF_ntarg.py` (see `docs/LOG.md`,
2026-08-28 entry, and the editorial note appended to Section 9 below). Any
future edit to `build_qpe_stage` in this directory must preserve the
`prec[::-1]` reversal -- removing it silently smears a sharp phase across
many bins rather than merely relabelling output bins (Rigetti brief Section
5.1 has the quantitative before/after: p=0.18 smeared vs p=1.00 exact,
n_prec=5, phi=1/8).

### 5.2 `XYXYX` is not a decoupling sequence
Carried over unchanged from `error_mitig.py`: it composes to `iX`, a net
bit flip. `error_mitig_ionq.py` keeps the same `NON_IDENTITY_SEQUENCES`
guard and `verify_sequences()` check -- the algebra is independent of which
device executes the pulses.

---

## 6. Experiment design

**Task economics.** Same structure as the Rigetti campaign's Section 6:
distinct tasks are determined by (DDD level, lambda), plus REM calibration
(2 tasks, reused across every DDD/lambda cell since REM is post-processing).
With fractional lambda in `{1, 1.5, 2, 2.5, 3}` and DDD in `{off, XY4}` as a
minimal first pass:

| | tasks |
|---|---|
| DDD-off x lambda in {1, 1.5, 2, 2.5, 3} | 5 |
| DDD-on (XY4) x lambda in {1, 1.5, 2, 2.5, 3} | 5 |
| REM calibration (cal0, cal1) | 2 |
| **all cells** | **12** |

**On-demand, not a reservation.** Unlike Rigetti, there is no booked window
and no `DirectReservation` context, so:
- Tasks may queue behind other users' work; per-task wall time is not under
  this project's control the way it is inside a reservation.
- There is no drift-confounding-with-block-order plan analogous to the
  Rigetti brief's "3 randomized blocks" -- on-demand tasks are naturally
  spread over whatever queue time the shared device gives them, which
  argues *for* still including a repeated anchor condition (DDD-off,
  lambda=1) across the batch to measure drift, but the blocking discipline
  itself does not transfer directly and should be redesigned once real
  queue-time behaviour is observed.

**Pairing.** Same as Rigetti: REM-on vs REM-off is a deterministic map on
identical counts (paired bootstrap, never quadrature); DDD-on vs DDD-off is
unpaired (different circuits, different times, and on-demand, different and
uncontrolled queue delay).

**The `II` null control is MANDATORY here, not optional** -- stronger than
the Rigetti brief's framing, because on IonQ the compiler-cancellation
question (Section 9.1) is entirely open. Without `II`, "DDD helped" and "the
compiler didn't cancel it and adding gates alone helped" and "DDD is
genuinely inert here" are three different explanations for the same
positive number, and only the null control's own comparison to `XY4`
distinguishes them (Section 9.2's three-outcome table).

---

## 7. Pre-flight validation in noisy simulation

**Not yet run as part of creating this campaign directory.** Unlike the
Rigetti brief's Section 7, which reports actual `qiskit-aer` validation
numbers, nothing below is invented to fill this section -- per CLAUDE.md's
instruction not to report a physics or numerical claim without a specific
check behind it. `mitig_analysis.py --validate` is ready to run (defaults:
`n_prec=4`, `phi=1/8`, `F1Q=0.9998`, `F2Q=0.985` -- IonQ's stated typical
values, not Rigetti's placeholders) and uses the same two-part noise model
as the Rigetti validation (depolarizing on `rx`/`rxx` + readout error, plus
a coherent static detuning on idle qubits for the DDD arm to have anything
to refocus -- see that function's docstring for why a purely Markovian
model would produce a guaranteed false negative for DDD).

Running it and recording the results here (or in a follow-up dated brief
section) is the natural next step before spending any real IonQ shots, and
should be done before Section 9.1/9.2's open questions are used to justify
*not* running the simulator check -- the simulator check is unaffected by
either open question, since it never touches the real device's compiler.

---

## 8. Known caveats and unknowns (summary; see Section 9 for the full detail)

- **Device caps not fetched.** No `device_caps.json` exists in this
  directory. `query_device_caps.py` is ready to run from the DCV box but has
  not been; `load_device_caps()` falls back to a 36-qubit all-to-all
  placeholder sourced from CLAUDE.md's project description, not a live
  query.
- **Verbatim support unconfirmed** -- see Section 9.1. `to_braket_qasm`
  defaults to `verbatim=False`.
- **No hardware runs yet under this campaign.** Everything above is
  circuit construction and (once run) simulation.
- **Timing constants and fidelities are estimates, not calibration.** See
  Section 9.2 and `ionq_mitigation_common.py`'s module docstring point 5.
- **The DDD result, once taken, is conditional on the coherent-detuning
  noise-model proxy**, same caveat as the Rigetti brief's Section 8: a null
  result would mean "not under this noise model", not "never".

---

## 9. Open questions (status: OPEN -- read this before writing any mitigation code)

Two things stand between this campaign and a trustworthy ZNE or DDD number on IonQ Forte-Enterprise-1. Both are *unresolved*, and neither is resolved by analogy to the Rigetti campaign. The Rigetti brief could state its verbatim requirement as a device fact because `braket_direct/rigetti_cepheus_20260807/device_caps.json` was actually fetched and records `"supportsPartialVerbatimBox": true` (line 659), alongside a full `device_properties_raw.json`. **No IonQ equivalent exists anywhere in this repository.** Everything we currently "know" about IonQ's compilation behaviour is an assumption written in a comment, and every IonQ hardware task in this repo (`noise_mitigation/job_results/`, `noise_mitigation/job_results_2stage/`) carries `error_mitigation` ∈ {`none`, `debias`} -- there has never been a folded circuit or a decoupling pulse on this device.

The two questions are not independent: Q2 is partly gated by Q1. Treat Q1 as blocking.

### 9.1 OPEN -- Does IonQ recompile our circuits, and does that recompilation cancel ZNE folds and DD pulses?

**Status: unconfirmed in both directions.** We have not established that IonQ preserves our gate sequence, and we have not established that it destroys it. Do not write either into a comment as settled.

What the repo actually contains:

- The only statement on the subject is an assumption inside a *cost estimator*, in `zne_resource_estimate()` at `/Users/yonnoylee/python/QUEN_QPUF/noise_mitigation/ionq_noise_mitig.py:242-244`, quoted exactly:

  > `Conservative model: no gate cancellation at the fold boundaries (an upper`
  > `bound on cost). Measurements are terminal and are NOT folded/duplicated, so`
  > `readout is charged once per shot per circuit, independent of lambda.`

  Read it for what it is: a deliberately pessimistic assumption chosen to make a *dollar/runtime* estimate an upper bound. It is not a claim about the device, it was never tested, and it is exactly backwards as a physics assumption -- for cost, "no cancellation" is conservative; for validity, "no cancellation" is the optimistic case.

- `/Users/yonnoylee/python/QUEN_QPUF/braket_direct/ionq_20260601/query_device_caps.py` does interrogate `supportsPartialVerbatimBox` (line 21), but **its output has never been captured**. There is no `device_caps.json`, no `device_properties_raw.json`, and no saved stdout under `braket_direct/ionq_20260601/` or `noise_mitigation/`. The flag's value for Forte-Enterprise-1 is unknown to this project.

- Meanwhile the repo's own comments assert that a compilation pass *does* exist: `braket_direct/ionq_20260601/submit_QPUF_ntarg.py:312-314` states that `rz/rx/rxx` are mapped server-side to native `gpi/gpi2/zz` at submission (same claim at `submit_test.py:104`). So the question is not "is there a compiler in the path" -- by our own account there is -- but "is that pass peephole-optimising enough to annihilate `U†U` and `XX → I`". That is precisely the property Quilc has and that forced verbatim on Rigetti.

- There is no post-hoc audit available from returned data. I checked every saved IonQ result: `additional_metadata.ionqMetadata` contains **only** `sharpenedProbabilities`, and `additional_metadata.action.source` is the *submitted* QASM, not the device-compiled program. So the compiled-vs-submitted gate-count comparison cannot be reconstructed from task metadata as it is currently retrieved and stored. It has to be probed deliberately.

**Why this is fatal rather than cosmetic.** ZNE's premise is that the *executed* error rate scales with λ, via `C → C(C†C)^k`. If the folds are cancelled, every λ executes as λ = 1. The failure is silent and looks like a result: the observable is flat in λ, the extrapolator dutifully returns the λ = 1 value, and the campaign reports "ZNE gave no improvement" when what actually happened is that ZNE never ran. Identically for DDD: if `XX → I` is applied, the DDD-on and DDD-off arms are the same circuit, and the `II` null control degenerates too -- you get a clean, well-bootstrapped null that means nothing.

**Expected forms, so the flatness is readable.** Under a working fold ladder the primary observable must *decay monotonically* in λ toward the incoherent floor: `1/d = 2^−n_prec` for `P_correct` on a single-stage on-grid-phase instance, and also `2^−n_prec` for `P(agree)` on the two-stage MCM QPUF (two independent uniform n_prec-bit registers agree with probability `2^−n_prec`). Under cancellation, the observable is constant in λ to within shot noise. Those two are cleanly distinguishable at the shot counts this campaign will use, and the check costs nothing extra because it uses the ZNE points themselves. Related: whatever extrapolant is fitted must carry an offset at the floor (e.g. `A·e^{−bλ} + 2^−n_prec`), not an asymptote at zero -- a fully decohered QPUF register returns the uniform distribution, not zero, and a zero-asymptote fit biases the λ→0 estimate upward. That mis-specification is one route to the out-of-`[0,1]` extrapolations already documented for Rigetti in §7 of `MITIGATION_BRIEF.md`.

**Sub-questions that are also open and are not answered by reading the capability flag:**
- Does IonQ verbatim, if supported, require physical-qubit `$n` addressing and native-gates-only bodies as Rigetti's does? If so it collides directly with the flat single-`QuantumRegister` OpenQASM 3 export Braket requires (CLAUDE.md known-bug list, item 3).
- Does a verbatim box compose with `EnableExperimentalCapability()`, which this project needs for MCM + `reset` (see `submit_QPUF_ntarg.py:380-385`)? Verbatim and mid-circuit measurement have never been used together here.
- Is IonQ Debias compatible with verbatim? Debias is a shot-level symmetrisation over gate variants; it is not obvious it can operate on a circuit the device is forbidden to rewrite.
- **Blind spot that survives even a "yes" answer:** unitary folding duplicates gate layers only, and by our own estimator's statement (`ionq_noise_mitig.py:243-244`) measurements are not folded. Debias likewise acts on unitary layers. So on the two-stage MCM QPUF, λ scales everything *except* the mid-circuit measurement channel -- which is the channel this project's central hypothesis says dominates. ZNE and Debias share that blind spot. Whatever the campaign reports, it must not be phrased as extrapolating the QPUF's total noise to zero.

**What would settle it (and must be recorded, not just observed):**
1. Run `query_device_caps.py` against the live device from the DCV box and commit both the raw stdout and a `device_caps.json` / `device_properties_raw.json`, mirroring the Rigetti directory layout.
2. The flag alone is insufficient -- it tells you whether verbatim is *available*, not what the compiler does when it is off. Add a cheap single-qubit probe: prepare |0⟩, apply `rx(π)` 2k times for k = 0,1,2,4,8, measure. Executed as written, P(0) decays smoothly with k; cancelled, P(0) is flat at the readout floor for all k. Run the ladder twice, verbatim on and off. **Falsification statement:** "IonQ does not cancel" is falsified iff the non-verbatim arm is flat in k while the verbatim arm decays. If both decay identically, verbatim is unnecessary for this campaign and we say so with data.
3. Build that probe (and all folded/DD circuits) with `optimization_level=0` and assert the submitted-QASM gate count before submission. Our own pipeline transpiles at `optimization_level=1` (`ionq_noise_mitig.py:158, 259, 476`; `submit_QPUF_ntarg.py:319`), which will cancel folds client-side before IonQ ever sees them. Folding and DD insertion must happen **after** the final transpile and must never be followed by a pass above level 0 -- the same constraint the Rigetti code documents at `error_mitig.py:46` and `mitig_analysis.py:538`.

Until (1) and (2) are done and recorded, **no ZNE or DDD number taken on real IonQ hardware under this campaign should be reported, cited, or plotted.** REM and Debias are unaffected -- REM adds no gates, so nothing can be cancelled out of it.

### 9.2 OPEN -- DDD has never been run on IonQ here, and the timing model it depends on is unvalidated

**Status: entirely untried.** There is no `apply_ddd`, no decoupling sequence, and no pulse-insertion code anywhere in `noise_mitigation/*.py` or `braket_direct/ionq_20260601/*.py`. The only DDD in the IonQ half of this repo is `mitiq`'s `ddd.rules.xyxy` invoked against a **cirq simulator** in `noise_mitigation/noise_mitigation.ipynb` (cell 13), which never touched hardware -- and that notebook's own text says DDD is expected to be near-flat there because its noise model is purely depolarizing.

**What does transfer.** The pulse algebra in `braket_direct/rigetti_cepheus_20260807/error_mitig.py:111-112` -- `X = rx(π)`, `Y = rz(−π/2)·rx(π)·rz(π/2)` -- is basis-agnostic and expressible in `IONQ_BASIS = ['rz', 'rx', 'rxx', 'measure']` (`ionq_noise_mitig.py:69`). Reuse the math.

**What does not automatically transfer, and why.**

- *That basis is not IonQ's hardware-native basis.* `rz/rx/rxx` is the Braket-level supported operation set; by the repo's own account (`submit_QPUF_ntarg.py:312-314`) IonQ re-synthesises it into `gpi/gpi2/zz`. Our `Y` decomposition is therefore *re-derived* by a compiler we have not characterised. Whether it comes out as a genuine π rotation about ŷ, and whether the surrounding `rz` frame updates get merged into neighbouring pulses in a way that destroys the sequence's refocusing structure, is unknown and folds back into §9.1.

- *The timing model has no provenance and no validation.* The constants are, exactly, in `/Users/yonnoylee/python/QUEN_QPUF/noise_mitigation/ionq_noise_mitig.py:39-42`:
  ```
  ONE_QUBIT_TIME_S = 130e-6
  TWO_QUBIT_TIME_S = 970e-6
  READOUT_TIME_S   = 150e-6 + 50e-6
  STARTUP_TIME_S   = 0.5
  ```
  duplicated at `braket_direct/ionq_20260601/submit_QPUF_ntarg.py:48-51`, where the comment reads "IonQ Forte-1 gate-time model (**used only for the wall-clock estimate**)". Correcting a phrasing that has been circulating: these are *not* calibration-derived in this repo. No calibration fetch produced them, no source is cited, and their only declared purpose is cost estimation. They have also never been checked against reality -- `qpu_time_seconds` is `None` in **every** saved IonQ result, so predicted-vs-measured execution duration has never been compared even once. (`task_time_seconds` includes queue time and ranges 197 s-4842 s against 137 s-290 s predicted; it is not a usable check.) The same caveat applies to `F2Q = 0.985`, `F1Q = 0.9998` at `ionq_noise_mitig.py:45-46`, whose own comment says "IonQ Forte typical; TUNE to your calibration."

  This matters because DDD is the one technique here whose *construction*, not just its interpretation, depends on the timing model: `apply_ddd` finds idle windows from real durations and only inserts where a window exceeds `min_idle_s`, defaulting to one 2q-gate time.

- *The two models in this repo disagree about `rz`, and the disagreement changes DDD qualitatively.* The Rigetti scheduler hard-codes `rz = 0` (`error_mitig.py:190-205`), justified explicitly as a *superconducting* frame update. The IonQ cost model charges every `rz` a full 130 µs (`n_1q = ops['rz'] + ops['rx']`, `ionq_noise_mitig.py:165, 261`). Both cannot be right. Consequence: an `XY4` sequence is 8 native ops (4 `rx` + 4 `rz`). With `rz` free it occupies 4 × 130 µs = 520 µs and fits comfortably inside a 970 µs default minimum window; with `rz` charged it occupies 8 × 130 µs = 1040 µs and does not fit at all. The insertion decision flips on an unresolved convention. Virtual-Z is standard on trapped ions too, so `rz = 0` is the likely answer -- but "likely" is not a basis for a scheduler, and the estimator currently encodes the opposite.

- *Even with correct timings and confirmed verbatim, the Rigetti DDD result does not predict the IonQ one.* On Rigetti the DD pulses (40 ns) are negligible against the idle windows they fill. On IonQ a single π pulse is ~130 µs against ~970 µs windows -- the pulses are *not* short compared with the window, so finite-pulse-width corrections to the filter function are not negligible, and inserting them measurably lengthens the circuit and hence total exposure. The trapped-ion noise being decoupled is also different in character (low-frequency magnetic-field and laser-phase drift, i.e. slow and correlated -- genuinely good DD territory, and arguably better than Rigetti's) but its spectral density is unmeasured by us. The honest prior is: plausible mechanism, unknown magnitude, unknown sign.

**Expected forms -- three outcomes, and the arm that distinguishes them.** Let Δ be measured on `P_correct` (single-stage, on-grid phase) or `P(agree)` (two-stage MCM), floor `2^−n_prec` in both cases:

| outcome | signature | reading |
|---|---|---|
| DDD compiled away or genuinely inert | XY4 − off ≈ 0 **and** II − off ≈ 0 | uninterpretable until §9.1 is closed |
| DDD suppresses correlated dephasing | XY4 − off > 0 **and** XY4 − II > 0, both CIs clear of zero | the effect is the pulses |
| pulse cost dominates | XY4 ≈ II, both < off | not a decoupling failure -- added time/infidelity |

The `II` null control is therefore **mandatory, not optional**, on IonQ; without it the second and third rows are indistinguishable and the result is uninterpretable. Note the control has its own dependency: a compiler is entitled to delete identity gates, so `II` is only meaningful inside a working verbatim box (`error_mitig.py:726-731`). Whether Braket/IonQ will accept and actually execute an `i` or `delay` instruction as real idle time is itself unverified here.

Two regression items carried over: (a) `XYXYX` is **not** a decoupling sequence -- it composes to `iX`, a net bit flip (`MITIGATION_BRIEF.md` §5.2) -- and must stay behind the explicit override on IonQ too; (b) the inverse-QFT endianness bug is documented as **unpatched** in `ionq_noise_mitig.py` and `submit_QPUF_ntarg.py` (`MITIGATION_BRIEF.md` §5.1, and CLAUDE.md known-bug item 4). Any IonQ observable is invalid until that is fixed and confirmed; a DDD comparison built on a smeared distribution measures nothing about decoupling.

**Bottom line for the coder:** implement the DDD path, but treat "does DDD help on IonQ" as an open experimental question with a real chance of a null or negative answer. Per §2 of `CLAUDE.md`, a correct null here is a deliverable. Do not port the Rigetti `+0.0237` XY4 number into any IonQ text, caption, or default parameter as if it were a prediction.

---

**Editorial addendum (not part of the physicist's text above -- added when this brief was written, 2026-08-28):** item (b) in Section 9.2's "Two regression items carried over" states the inverse-QFT endianness bug is "unpatched" in `ionq_noise_mitig.py` and `submit_QPUF_ntarg.py`. That was true when the physicist wrote the passage above earlier today, but **it has since been fixed and verified**, in a separate task the same day -- both files now correctly append the inverse QFT to `prec[::-1]`. See `docs/LOG.md`'s 2026-08-28 entry ("Worker2 (coder/verifier/physicist chain)") for the fix, the verification method (known-answer test on a noiseless `Statevector`, exact eigenstate), and its downstream physics impact on prior hardware data. This campaign's own `ionq_mitigation_common.build_qpe_stage` (Section 5.1 above) is written with the fix already in place from the start. Section 9.1 and the rest of Section 9.2 remain fully OPEN and unaffected by this fix -- the endianness bug and the compiler-cancellation question are unrelated failure modes.
