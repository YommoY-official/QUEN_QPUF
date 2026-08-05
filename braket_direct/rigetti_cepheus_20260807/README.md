# QPUF + noise mitigation on Rigetti Cepheus-1-108Q

Braket Direct reservation run. Same two-stage PE-QPUF algorithm as the
IonQ Forte-Enterprise-1 run in `../ionq_20260601/`, plus client-side noise
mitigation.

## Order of operations on the DCV

```bash
export AWS_DEFAULT_REGION=us-west-1      # Cepheus lives in us-west-1

python query_device_caps.py              # 1. cache real topology + calibration
python submit_test.py                    # 2. tiny job, prove the pipeline
python checkRetrieve.py job_results_test # 3. retrieve it, calibrate the model
python submit_qpuf_mitigation.py         # 4. the real run
python checkRetrieve.py                  # 5. retrieve everything
```

Set `RES_ARN` in `rigetti_qpuf_common.py` before step 4, otherwise tasks go
out on-demand and are billed per task instead of against the reservation.

## Files

| file | role |
|---|---|
| `rigetti_qpuf_common.py` | circuit builders, routing, ZNE folding, REM circuits, runtime model — shared by everything else |
| `query_device_caps.py` | dumps live device caps, writes `device_caps.json` |
| `submit_test.py` | tiny end-to-end pipeline test (local sim / SV1 / QPU / reservation) |
| `submit_qpuf_mitigation.py` | the real run; prompts for N_PREC / N_TARG / shots / mitigation, prints full cost, confirms before submitting |
| `checkRetrieve.py` | polls task status, saves counts + Rigetti native-Quil metadata to `<dir>/<uuid>.json` |

## What changed versus the IonQ scripts

| | IonQ Forte-Enterprise-1 | Rigetti Cepheus-1-108Q |
|---|---|---|
| qubits | 35 | 108 |
| connectivity | all-to-all | fixed lattice — we route ourselves |
| mid-circuit measurement | yes (experimental capability) | **no** — QPUF uses the deferred-measurement two-register form |
| native 2q gate | `xx` (Mølmer-Sørensen) | `cz` |
| error mitigation | `Debias` (vendor-side, 1 task) | **none vendor-side** — ZNE + readout calibration, client-side, extra tasks |
| runtime driver | gate count (serial ion chain) | circuit depth + per-shot reset (parallel lattice) |

## Mitigation

- **ZNE** — global folding `C -> C (C^dag C)^k`, one task per odd lambda.
  Folded gate counts are exactly `lambda x` the base (verified).
  **Requires verbatim mode**: a normal submission is recompiled by Rigetti's
  Quilc, which cancels the `C^dag C` pairs and silently executes every
  lambda as lambda=1. `submit_test.py` has a verbatim option — confirm the
  device accepts it before relying on ZNE.
- **REM** — two extra tasks (`cal0` = all-|0>, `cal1` = all-|1>) on exactly
  the physical qubits the QPUF reads, giving per-qubit 2x2 confusion
  matrices to invert offline. Cheap and high-value: readout is usually the
  dominant error on superconducting hardware.

## Constants that are still placeholders

`rigetti_qpuf_common.py` carries a gate-time model
(`ONE_QUBIT_TIME_S`, `TWO_QUBIT_TIME_S`, `READOUT_TIME_S`, `SHOT_RESET_S`,
`STARTUP_TIME_S`) and a fidelity model (`F1Q`, `F2Q`). Rigetti does not
publish per-gate durations through Braket, so these are order-of-magnitude
guesses and the runtime estimate inherits their error.

`query_device_caps.py` prints the calibration medians to paste into `F1Q` /
`F2Q`. `checkRetrieve.py` prints the measured task time and Rigetti's own
`qpuRuntimeEstimation` next to our prediction — re-fit the timing constants
from the step-3 test job before sizing the reservation window.
