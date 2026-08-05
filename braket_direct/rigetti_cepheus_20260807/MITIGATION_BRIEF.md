# Error-Mitigation Implementation Brief

Self-contained context for discussing this implementation with someone (or
something) that has not seen the codebase.

---

## 1. What we are building

Benchmarking three error-mitigation techniques — **REM**, **DDD**, **ZNE** — on
**Quantum Phase Estimation** running on **Rigetti Cepheus-1-108Q** via **AWS
Braket**, under a booked Braket Direct reservation. It is a characterization
project: the deliverable is how each technique's benefit scales with precision
register size and depth, plus an honest account of where each fails.

**Environment:** Python 3.13.13, qiskit 2.4.1, amazon-braket-sdk 1.117.2,
qiskit-aer 0.17.2, numpy 2.4.4, scipy 1.17.1, cirq 1.7.0. Mitiq **not**
installed.

**Device facts that shape everything:**
- Native gate set `rz` / `rx` / `cz`. `rz` is a virtual frame change — **zero
  duration**.
- Twelve 9-qubit chiplets in a 3×4 array; intermodule couplers are the weakest
  links.
- **No mid-circuit measurement** exposed for Rigetti through Braket (unlike IonQ
  Forte, which has it behind an experimental-capability flag).
- Rigetti's Quilc compiler recompiles submissions unless they are inside a
  `#pragma braket verbatim box`. **It will cancel `U U† U → U` (ZNE folds) and
  `XX → I` (DD pulses).** Verbatim is therefore mandatory, not optional — and
  verbatim requires physical-qubit addressing (`$n`) and native gates only,
  which means we must route the circuit ourselves rather than let Quilc do it.

**Circuit:** single-stage QPE. Precision qubit `k` controls `U^(2^(n_prec-1-k))`
(MSB-first), then inverse QFT, then one terminal measurement of the precision
register. Qubits used = `n_targ + n_prec`. Primary benchmark instance is an
on-grid phase: `U` diagonal with an exact `|1…1⟩` eigenstate carrying phase
`φ = 1/8`, so noiseless QPE puts **all** weight in bin `φ·2^n_prec` and any
spread is measured error.

---

## 2. Code layout

| file | role |
|---|---|
| `rigetti_qpuf_common.py` | circuit builders, lattice routing, OpenQASM3/verbatim export, device-caps cache, runtime model |
| `error_mitig.py` | `ErrorMitigation` class — placement, DDD, ZNE folding, REM calibration circuits, verification |
| `mitig_analysis.py` | estimator half — P_correct, REM inversion, ZNE extrapolation, bootstrap CIs, noisy-sim validation |
| `submit_qpe.py`, `submit_test.py`, `checkRetrieve.py`, `query_device_caps.py` | submission / retrieval pipeline |

---

## 3. The contract split (the central design point)

The original intent was "every mitigation is a function: circuit in, circuit
out." **Only half the techniques actually fit that shape**, and forcing the
other half into it would have been a lie:

| technique | circuit → circuit? | extra hardware tasks |
|---|---|---|
| chiplet-local placement | **yes** | 0 (always-on substrate, not a variable) |
| DDD | **yes** | 0 |
| ZNE | one circuit *per λ* | N−1, plus an extrapolation fit |
| REM | **no — adds zero gates** | 2 calibration circuits + classical inversion |

REM leaves the submitted circuit byte-for-byte unchanged; the mitigation is an
inversion applied to the *counts*. ZNE's `fold()` is genuinely circuit→circuit,
but a single folded circuit is not a mitigated result.

Consequence for structure: `error_mitig.py` builds circuits, `mitig_analysis.py`
consumes counts. REM and ZNE each have a half in both files.

---

## 4. Implementation decisions and rationale

**No mitiq — two independent reasons.**
1. *It will not install.* mitiq 1.0.0 requires `>=3.11,<3.13`; the environment is
   Python 3.13.13. pip silently walks back to mitiq 0.25.0 (≈2 years old), and
   that resolution fails trying to build an ancient numpy from source
   (`pkgutil.ImpImporter`, removed in 3.12).
2. *Its converters would break this project even if it did.* mitiq is
   cirq-native; its qiskit frontend round-trips **qiskit → QASM2 text → cirq →
   QASM2 → qiskit**. QASM2 carries no `TranspileLayout`, and idle qubits are
   dropped with survivors renumbered contiguously. Our circuits are 108-qubit
   registers with ~6–18 qubits touched, submitted with `$n` physical addressing
   inside a verbatim box. Physical qubit 47 would return as `q[3]` — the job
   would run on the wrong qubits, silently.

   Separately, mitiq's DDD slack detection is **moment-based, not
   duration-based**, which is precisely the part we needed to do differently.

**Timing model.** `rz` = 0 s, `rx` = 40 ns, `cz` = 180 ns, readout ≈ 2 µs
(placeholders except the rz=0, which is structural). Layer duration = longest op
in the layer, since same-layer ops act on disjoint qubits. A moment-based
detector treats a run of `rz` as busy and misses real idle windows.

**DD sequences built directly in native gates** — `X = rx(π)`,
`Y = rz(−π/2)·rx(π)·rz(π/2)` — so no second transpile pass is needed after
insertion. A second pass is exactly what cancels the pulses. Pulses are spread
CPMG-style at `(p+0.5)/P` through each idle window.

**ZNE folding is stratified** — 1q and 2q gates folded as separate pools, each
scaled to λ independently (see §5.3).

**Placement scoring is route-aware.** QPE's interaction graph is a *star*, not a
chain: every controlled-`U^(2^k)` touches the target register, so the target is a
hub all `n_prec` precision qubits must reach. A 9-qubit chiplet has max degree
~4, so beyond `n_prec = 4` you need SWAPs even inside one chiplet. Candidates
anchor the hub at high-degree sites, then are scored by **running the actual
router** and evaluating ∏(edge fidelities over routed 2q gates, SWAPs included) ×
∏(readout fidelities). Placements using intermodule couplers are rejected unless
the register cannot fit.

**Composition order (fixed):**
```
placement → DDD insertion → [hardware] → REM → P_correct → ZNE extrapolation
```
Placement first because it changes the routed gate count and hence the idle
structure; DDD before folding because folding after would duplicate the DD
pulses.

---

## 5. Bugs found and fixed (all empirically verified)

### 5.1 Inverse-QFT endianness — the serious one
The controlled-U loop gives `prec[0]` weight `2^(n_prec−1)` (MSB-first), but
qiskit's `QFTGate` is little-endian (qubit *k* gets weight `2^k`). Appending the
inverse QFT to `prec` directly mismatches the conventions.

**This is not a relabelling of output bins — it smears a sharp phase across many
bins.** With `φ=1/8`, `n_prec=5`, exact eigenstate:

| | bin 4 | rest |
|---|---|---|
| before (`prec`) | 0.18 | smeared over ~all bins |
| after (`prec[::-1]`) | **1.00** | 0 |

Fixed in the Rigetti module. **The same line exists in the IonQ scripts**
(`submit_QPUF_ntarg.py`, `ionq_noise_mitig.py`) and has *not* been patched there.
It was invisible in the two-stage QPUF because that only asks whether m1 == m2 —
but it is not benign there either: both stages sample a spread distribution, so
ideal agreement drops well below 1.

### 5.2 `XYXYX` is not a decoupling sequence
It composes to `iX` — a net bit flip, not identity. Verified by its effect on the
noiseless QPE result:

| sequence inserted | result |
|---|---|
| none / XX / YY / XY4 / II | bin 4, p = 1.000000 |
| **XYXYX** | **bin 20, p = 0.376** |

Now raises unless explicitly overridden.

### 5.3 Uniform random folding mis-scales the noise that matters
Folding from one combined pool at λ=1.5 scaled *total* gates by exactly 1.500 but
**2q gates by 1.839** — 23% more noise than requested, while the extrapolator is
told 1.5. On superconducting hardware 2q error dominates, so this biases the fit.

| λ | 2q scale, uniform | 2q scale, stratified |
|---|---|---|
| 1.5 | 1.839 | 1.516 |
| 2.0 | 1.968 | 2.032 |
| 2.5 | 2.355 | 2.484 |

Folding is verified to preserve the ideal answer (bin 4, p=1.0) at every λ.

### 5.4 Two smaller ones
- Routing cache keyed on `id(qc)` — CPython reuses ids after GC, so a transient
  circuit could return another circuit's routing. Now scoped per call.
- Provenance lost: folding rebuilds the circuit, so a placement record written
  before DDD was gone by logging time. Now a single inherited metadata dict
  survives place → DDD → fold, and REM calibration circuits inherit it.

---

## 6. Experiment design

**Task economics.** The naive 2×2×2 factorial (REM × DDD × ZNE) reads as 8
conditions → 32 tasks. But REM is post-processing (same counts serve REM-on and
REM-off) and the λ=1 circuit *is* the ZNE-off circuit. Distinct tasks are
determined only by (DDD level, λ):

| | tasks |
|---|---|
| DDD-off × λ ∈ {1, 1.5, 2, 2.5, 3} | 5 |
| DDD-on × λ ∈ {1, 1.5, 2, 2.5, 3} | 5 |
| REM calibration (cal0, cal1) | 2 |
| **all 8 factorial cells** | **12** |

**Shots are nearly free; tasks are not.** Per-task overhead ≈ 15 s vs 1000 shots
≈ 0.2 s. Use ~10,000 shots/task and economize on task count.

**Drift.** Rigetti calibration drifts over hours, so condition order is
confounded with time. Plan: 3 randomized blocks of the 12-task set via
`run_batch`, cal0/cal1 in *every* block adjacent to the data they correct, and
the DDD-off/λ=1 task as a repeated anchor to measure drift directly.

**Pairing.** REM-on vs REM-off is a *deterministic map on identical counts* —
paired bootstrap, never quadrature. DDD-on vs DDD-off is unpaired (different
circuits, different times).

---

## 7. Noisy-simulation validation results

Run with qiskit-aer before spending hardware shots. **n_prec=4, φ=1/8 (ideal bin
2 at p=1.0), 20,000 shots, F1Q=0.999, F2Q=0.97, readout error 3%.**

### Noise-model design point (important)
A purely Markovian model (depolarizing + thermal relaxation) **cannot show any
DDD benefit at any strength** — memoryless noise gives the same total decay
however you slice an idle window. DD works against slow, correlated noise. So the
model has two parts: (1) depolarizing on rx/cz + readout error, what REM and ZNE
address; (2) a **coherent static detuning** injected as an `rz` proportional to
idle duration, what DDD refocuses. Without (2) a DDD validation returns a false
negative.

### Results

| condition | P_correct | Δ vs baseline | 95% CI | |
|---|---|---|---|---|
| no mitigation | 0.4828 | — | | |
| DDD XX | 0.4839 | +0.0012 | [−0.0096, +0.0114] | n.s. |
| DDD YY | 0.4667 | **−0.0161** | [−0.0251, −0.0059] | sig — *hurts* |
| **DDD XY4** | 0.5064 | **+0.0237** | [+0.0133, +0.0327] | sig |
| DDD II (null control) | 0.4891 | +0.0064 | [−0.0048, +0.0166] | n.s. |
| XY4 **vs null control** | | +0.0177 | [+0.0087, +0.0272] | sig |
| **REM (simplex)** | 0.5397 | **+0.0569** | [+0.0538, +0.0596] | paired |

- **REM is the clear winner** — 2.4× DDD's effect, from 2 tasks every condition
  reuses. Confusion-matrix condition number 1.29, so almost no variance bought.
- **DDD works and the null control validates the mechanism**: XY4 beats `II` with
  the CI clear of zero, so the gain is the pulses, not merely added gates. `YY`
  actively hurts — do not assume all sequences help.

### ZNE fails, quantifiably

| λ | 1 | 1.5 | 2 | 2.5 | 3 |
|---|---|---|---|---|---|
| P_correct | 0.483 | 0.373 | 0.258 | 0.195 | 0.158 |

Same five points → Richardson **0.054**, linear **0.624**, exponential **0.906**.
Model spread **0.85** on a quantity bounded in [0,1].

Combined (placement + DDD + REM + ZNE) extrapolates to **1.49** (Richardson) and
**1.40** (exponential) — impossible for a probability. REM lifts every point on
the curve and the extrapolation multiplies that shift by its lever arm. The
analysis layer now flags out-of-[0,1] results as *invalid*, not merely noisy.

### ZNE viability is gate-fidelity-driven, not depth-driven

| F2Q | model spread, n_prec=3 | n_prec=4 | usable? |
|---|---|---|---|
| 0.97 | 0.50 | 0.85 | no |
| 0.99 | 0.06 | 0.09 | yes |
| 0.995 | 0.09 | 0.13 | yes |

**So whether ZNE is worth any hardware time depends entirely on Cepheus's real CZ
fidelity, which we do not yet have** — 0.97 is a placeholder.

---

## 8. Known caveats and unknowns

- **Device caps not yet fetched.** No AWS credentials locally; `device_caps.json`
  is absent, so routing currently uses a placeholder square lattice and the
  chiplet partition falls back to BFS (14 groups of 1/8/9 rather than twelve 9s).
  Gate-time constants, F1Q/F2Q, and per-edge fidelities are all placeholders.
- **Verbatim support unconfirmed on the live device.** `supportsPartialVerbatimBox`
  has not been read yet. If verbatim is unavailable, ZNE and DDD are both void on
  this hardware.
- **No hardware runs yet.** Everything above is simulation.
- **The DDD result is conditional on the coherent-detuning model.** If Cepheus's
  dominant idle error is Markovian rather than slow-drift, the hardware number
  will be smaller. A null result on hardware means "not under this noise," not
  "never."
- **Placement cannot be validated in a uniform noise model** — it only matters
  with heterogeneous per-qubit calibration, so it is untested in simulation.

---

## 9. Open questions worth discussing

1. **Noise-model realism.** Is a coherent static detuning the right proxy for
   Rigetti idle error? What is the actual spectral character (1/f flux noise,
   TLS)? Should the DD validation use a filter-function treatment instead?
2. **ZNE scaling convention.** We fold all gates stratified so both 1q and 2q
   scale by λ. Alternative: fold *only* CZs, making λ a scale factor on 2q count.
   Which is the more defensible definition when reporting?
3. **Folding vs alternatives.** Global vs local folding vs identity insertion vs
   pulse stretching — is unitary folding the right noise-scaling primitive at
   all on a device where 2q error dominates?
4. **Constrained extrapolation.** Given the unphysical results, should the ZNE
   fit be constrained to [0,1] (e.g. logit-space fit), or does constraining hide
   the diagnostic signal that the extrapolation is invalid?
5. **REM model choice.** Tensored (2 circuits) vs full (2^n). Is readout
   crosstalk on this architecture large enough to justify 2^n?
6. **Observable choice.** P_correct is linear in the state, which is what ZNE
   needs. Is it the right primary observable, or is something like total
   variation distance more informative despite not being ZNE-legitimate?
7. **DD pulse placement.** Uniform CPMG spacing vs Uhrig (UDD) — worth testing
   UDD given the geometric idle-time asymmetry across the precision register?
8. **Composition order.** We do REM then ZNE. Is there an argument for ZNE then
   REM, and does REM's variance amplification interact badly with the
   extrapolation lever arm?
9. **Interactions.** Better placement shrinks the idle windows DDD exploits, so
   their gains are known to be sub-additive. How should the study report
   interaction terms rather than assume additivity?
