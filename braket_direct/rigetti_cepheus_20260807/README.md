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
python submit_qpe.py                     # 4a. QPE run
python checkRetrieve.py job_results_qpe

python submit_qpuf_mitigation.py         # 4b. two-stage QPUF run
python checkRetrieve.py
```

Set `RES_ARN` in `rigetti_qpuf_common.py` before step 4, otherwise tasks go
out on-demand and are billed per task instead of against the reservation.

## Files

| file | role |
|---|---|
| `rigetti_qpuf_common.py` | circuit builders, routing, runtime model — shared by everything else |
| `error_mitig.py` | `ErrorMitigation` class: chiplet-local placement, DDD, ZNE folding, REM calibration circuits, verification |
| `mitig_analysis.py` | estimator half: P_correct, REM inversion, ZNE extrapolation, bootstrap CIs. `python mitig_analysis.py --validate` runs every technique in noisy simulation |
| `query_device_caps.py` | dumps live device caps, writes `device_caps.json` |
| `submit_test.py` | tiny end-to-end pipeline test (local sim / SV1 / QPU / reservation) |
| `submit_qpe.py` | **plain single-stage QPE** with/without mitigation; prompts for N_PREC / N_TARG / shots / mitigation / input state |
| `submit_qpuf_mitigation.py` | the two-stage QPUF run; prompts for N_PREC / N_TARG / shots / mitigation |
| `checkRetrieve.py` | polls task status, saves counts + Rigetti native-Quil metadata to `<dir>/<uuid>.json` |
| `cancel_tasks.py` | cancel tasks by job log, by reservation tag, or every queued task on the device. Dry run unless `--yes` |

## Which results directory

Each submit script writes its own, and `checkRetrieve.py` with no argument
reads `job_results/`. Passing the wrong one is the usual cause of
"contains no records" — the error now lists the directories that do have
records.

| script | writes | retrieve with |
|---|---|---|
| `submit_test.py` | `job_results_test/` | `python checkRetrieve.py job_results_test` |
| `submit_qpe.py` | `job_results_qpe/` | `python checkRetrieve.py job_results_qpe` |
| `submit_qpuf_mitigation.py` | `job_results/` | `python checkRetrieve.py` |

## Cancelling

Tasks carry a `ReservationRun` tag (see `RUN_TAG` in
`rigetti_qpuf_common.py`) because Braket has no "list everything for
reservation X" API — the tag is the only handle that finds them all.

```bash
python cancel_tasks.py --log job_results_qpe   # from recorded task ARNs
python cancel_tasks.py --tag                   # everything tagged for this run
python cancel_tasks.py --queued                # every cancellable task on Cepheus
python cancel_tasks.py --tag --yes             # actually cancel
```

Every mode is a dry run until `--yes`. If a task's status cannot be read the
script says so loudly and exits non-zero rather than reporting "nothing to
cancel" — an unchecked task is not a cancelled one.

Per the Braket Direct guidance: if the workload looks wrong, diagnose before
cancelling. Idle tasks cost nothing extra, and a cancelled task returns no
results.

## QPE vs QPUF

| | `submit_qpe.py` | `submit_qpuf_mitigation.py` |
|---|---|---|
| stages | 1 | 2 |
| qubits | `N_TARG + N_PREC` | `N_TARG + 2*N_PREC` |
| measurement | one terminal readout of the precision register | two terminal readouts (deferred-measurement form) |
| reference | exact — `known` mode puts all weight in bin `PHI*2^N_PREC` | m1 vs m2 agreement statistic |
| results dir | `job_results_qpe/` | `job_results/` |

QPE is the cleaner mitigation benchmark: with `STATE_MODE=known` you know the
answer exactly, so any spread is measured error and the effect of ZNE/REM is
directly readable. Verified end to end on the Braket simulator — `PHI=1/8`,
`N_PREC=6` returns 500/500 shots in bin 8.

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
