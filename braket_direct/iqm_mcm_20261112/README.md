# iqm_mcm_20261112 -- sequential two-stage PE-QPUF on IQM Garnet / Emerald

## Purpose

Run a genuinely sequential two-stage PE-QPUF (stage-1 QPE, mid-circuit
measurement, stage-2 QPE on the same target T) on IQM Garnet or Emerald, using
Braket experimental dynamic circuits (`measure_ff` + `cc_prx`) submitted as
verbatim programs (eu-north-1). Stage outcomes are m1 and m2; precision
registers are P1 and P2.

## Arms

- **A (deferred twin):** same sites and native gates, no `measure_ff` / `cc_prx`.
- **B (sequential):** stage-1 QPE, `measure_ff` on every P1 qubit, stage-2 QPE.
  IQM does not return `measure_ff` outcomes, so m1 is kept by copying it into a
  fresh ancilla with `cc_prx(pi, 0, key)` (X flip iff the P1 result was 1).
  P1 is re-read at the end; ancilla == re-read P1 witnesses that the
  measurement happened mid-circuit, and the disagreement rate is the decay of
  P1 during stage 2.
- **Breuse (on request, `--arms A B Breuse`):** after `measure_ff`, P1 is reset
  by `cc_prx` on itself and reused as P2.
- **Known-answer arm (each arm):** phase 1/2^d, T in |1>. Expected
  m1 = m2 = 1 with probability 1 (noiseless).

Default arms are now A and B.

## Files

- `iqm_qpuf_common.py` -- circuit builders, routing, dynamic-circuit rule check, decode, cost.
- `query_device_caps.py` -- read-only; writes `device_caps_<dev>.json`.
- `device_caps_garnet.json` -- edges, groups, partial readout fidelities, from the
  DCV data. Its groups were read off AWS figures, not from a published property.
- `validate_local.py` -- free local checks (Aer). Self-contained: the noiseless
  reference is `reference_pairs` in `iqm_qpuf_common.py` (one statevector of the
  whole two-stage circuit); nothing is imported from `docs/`. Also checks the
  smoke layout (non-coupler pairs, known answer m1=m2=1, ancilla == re-read P1,
  rules OK). All checks pass.
- `submit_qpuf_mcm.py` -- builds tasks, `--dry-run`, real submission.
- `checkRetrieve.py` -- retrieves results, decodes (m1, m2), writes `job_results/<uuid>.json`.
- `job_results/` -- manifest `job_log.txt` (one record per task, written at submit) and programs.

## Submit options

- `--smoke`: a single circuit, known-answer arm B, d=2, 200 shots, Garnet,
  `--layout nonadj`, no calibration. Layout T=[10], P1=[11,12], ancillas=[20,4];
  copy pairs $11->$20 and $12->$4 are both non-couplers within group 2; 22 CZ,
  est. survival 0.69; cost $0.59. It tests whether the device accepts
  non-adjacent `cc_prx`. The dry run prints each copy pair as coupler /
  non-coupler. Run this first.
- `--layout primary|adjacent|nonadj`: primary puts the ancilla anywhere in P1's
  qubit group (`cc_prx` is classically controlled). Adjacent makes every
  P1_i-ancilla_i pair also a coupler. Nonadj makes no P1_i-ancilla_i pair a
  coupler and uses only qubits with readout fidelity >= 0.98 for P1 and
  ancillas (Garnet: 3, 4, 10, 11, 12, 20).
- No fallback above d=2: `--layout adjacent` builds d=2 only; the search found no
  d=3 layout (a search limit, not a proof). If the smoke test shows non-adjacent
  `cc_prx` is rejected, d=3 is not possible with this code. The smoke test
  decides whether d=3 is possible.
- Two readout-calibration circuits (|0..0>, |1..1>) on the measured qubits are
  added to each batch (`--no-cal` omits them). `checkRetrieve.py` prints the
  witness agreement next to the readout-only prediction.
- The layout stays in the largest group (Garnet group 2) and avoids the
  poor-readout qubits 5, 16, 17 for P1 and ancillas.

## Workflow

1. On the DCV: `python query_device_caps.py garnet|emerald`. If the device
   publishes no qubit groups, hand-fill `groups` in `device_caps_<dev>.json`.
2. `python validate_local.py` -- free. Uses Aer because the Braket
   LocalSimulator rejects `measure_ff`.
3. `python submit_qpuf_mcm.py --dry-run` (add `--smoke` to preview the first run) --
   builds everything, prints gate counts and cost, submits nothing.
4. Real submission only by Yon, after an explicit go-ahead for that exact
   device, circuits, shots, and cost (CLAUDE.md hardware gate). There is no
   `--yes` flag; you must type `SUBMIT`.
5. `python checkRetrieve.py` (`--watch` to poll).

## Cost (AWS Braket pricing page)

Garnet $0.30/task + $0.00145/shot; Emerald $0.30/task + $0.00160/shot.

## Open items

- Until `device_caps_<dev>.json` exists the lattice is a placeholder; real
  submission is refused in that state. Emerald has no caps file yet.
- Garnet groups were read off AWS figures, so the `cc_prx` group rule is only
  as good as that reading; Emerald groups are unchecked.
- One `measure_ff` key driving two `cc_prx` (Breuse) is unconfirmed on the device.
- Whether the device accepts non-adjacent feed-forward (primary / nonadj
  layouts) is unconfirmed. The smoke test decides it. `--layout adjacent` works
  for d=2 only, so if non-adjacent `cc_prx` is rejected there is no d=3.
- Bit ordering of returned strings (ascending physical qubit) is to be
  confirmed by the known-answer arm on the first hardware run.
- Prices are from the AWS pricing page, not from a billed task.
