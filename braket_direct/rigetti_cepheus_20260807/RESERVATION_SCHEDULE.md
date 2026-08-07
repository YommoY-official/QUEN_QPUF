# Braket Direct reservation schedule — Cepheus-1-108Q QPE mitigation run

Documents the 2026-08-07 reservation session plan. 60-minute window. Reservation ARN `arn:aws:braket:us-west-1:767397707562:reservation/339f1ef2-2aec-459e-8e09-8a945b52755a`.

## What runs

Three `(n_prec, n_targ)` configurations, each run in both `known` and `haar` state modes, 3 batches each.

| Config | n_prec | n_targ | Logical qubits | Note |
|---|---|---|---|---|
| A | 5 | 1 | 6 | |
| B | 3 | 2 | 5 | |
| C | 5 | 2 | 7 | only if time remains |

Fixed for every run:

| Parameter | Value |
|---|---|
| N_SHOTS | 10000 |
| MITIGATION | both |
| ZNE_SCALES | 1 3 5 |
| VERBATIM | y |
| PHI | 1/8 (known mode) |

## The task economy (why 5 tasks give 4 conditions)

One `submit_qpe.py` invocation with `MITIGATION=both` and scales `1 3 5` emits 5 tasks which post-process into four mitigation conditions. Do not run raw/REM/ZNE/ALL as separate submissions — REM is post-processing on the same counts, and lambda=1 IS the raw circuit.

| Task | What it feeds |
|---|---|
| QPE lambda=1 | "raw" baseline, and the lambda=1 point of ZNE |
| QPE lambda=3 | ZNE |
| QPE lambda=5 | ZNE |
| cal0 | REM (applied to lambda=1) and ZNE+REM (applied to all lambda) |
| cal1 | same as cal0 |

So: 3 configs x 2 state modes x 3 batches = 18 invocations = 90 tasks.

## Measured circuit cost

Measured by routing the real circuits with `place_and_route()` against the live `device_caps.json` (107 qubits). est F uses the PLACEHOLDER `F1Q=0.999` / `F2Q=0.97` in `rigetti_qpuf_common.py`.

| Config | lambda | 1q gates | 2q gates | Depth | 2q depth | est F | est task time (s) |
|---|---|---|---|---|---|---|---|
| n_prec5_n_targ1 | 1 | 222 | 46 | 125 | 33 | 1.97e-01 | 17.1 |
| n_prec5_n_targ1 | 3 | 666 | 138 | 373 | 99 | 7.68e-03 | 17.3 |
| n_prec5_n_targ1 | 5 | 1110 | 230 | 621 | 165 | 2.99e-04 | 17.5 |
| n_prec3_n_targ2 | 1 | 148 | 28 | 91 | 21 | 3.68e-01 | 17.1 |
| n_prec3_n_targ2 | 3 | 444 | 84 | 271 | 63 | 4.96e-02 | 17.2 |
| n_prec3_n_targ2 | 5 | 740 | 140 | 451 | 105 | 6.71e-03 | 17.3 |
| n_prec5_n_targ2 | 1 | 350 | 88 | 172 | 51 | 4.83e-02 | 17.2 |
| n_prec5_n_targ2 | 3 | 1050 | 264 | 514 | 153 | 1.13e-04 | 17.4 |
| n_prec5_n_targ2 | 5 | 1750 | 440 | 856 | 255 | 2.63e-07 | 17.7 |

Key findings:

1. Task time is FLAT at ~17s regardless of circuit size, because `STARTUP_TIME_S=15s` dominates and gates contribute under 3s. Therefore task COUNT drives the schedule, not circuit size.
2. One invocation (5 tasks) = ~86s of QPU time. 18 invocations = ~26 min estimated, ~45 min if the real per-task overhead is 30s rather than the placeholder 15s.
3. Fidelity ranking at lambda=1: Config B (0.368) > Config A (0.197) > Config C (0.048). Config B has the most headroom to show a mitigation gain; Config C sits at the noise floor even at lambda=1.
4. For Config C, lambda=3 (1.1e-4) and lambda=5 (2.6e-7) will be indistinguishable from a uniform distribution — its ZNE result is a null by construction.

## Pre-flight (before the window opens)

```bash
export AWS_DEFAULT_REGION=us-west-1
cd ~/python/QUEN_QPUF/braket_direct/rigetti_cepheus_20260807
tmux new -s res
aws sts get-caller-identity
aws s3 ls | grep braket
python query_device_caps.py
python submit_test.py     # target: local  -> free, proves the QASM path
```

- Region must be `us-west-1` (Cepheus is us-west-1 only).
- The default S3 bucket `amazon-braket-<account>-us-west-1` must exist or task creation fails.
- Run everything in tmux so a dropped DCV session does not kill a submit loop.

## T+0 verbatim gate (do not skip)

```bash
python submit_test.py
# target -> res ; verbatim -> y ; submit -> y
```

Costs 1 task, about 1 minute. Proves both verbatim acceptance and that the reservation ARN routes.

If verbatim is REJECTED: answer `n` to verbatim on everything afterward and drop ZNE scales to `1`. lambda=3/5 would otherwise be silently executed as lambda=1 by Rigetti's Quilc (it cancels the C-dagger-C folds), wasting 2 tasks per invocation. REM still works fine without verbatim.

## Run order

Batch-major, interleaved by state mode. Batch-major matters: if you run out of time you end up with complete batches of A and B rather than 3 batches of A and none of B. Interleaving state modes keeps chip drift from confounding known vs haar.

| Round | Runs | Invocations | Tasks |
|---|---|---|---|
| 1 | A-known, A-haar, B-known, B-haar | 4 | 20 |
| 2 | A-known, A-haar, B-known, B-haar | 4 | 20 |
| 3 | A-known, A-haar, B-known, B-haar | 4 | 20 |
| 4 (only if time) | C-known, C-haar x 3 batches | 6 | 30 |

Insurance option: if you want at least one Config C replicate guaranteed, insert C-known and C-haar once after Round 2.

Approximate clock, T+0 = window open:

| Time | Activity |
|---|---|
| T+0 to T+3 | verbatim gate (1 task) |
| T+3 to T+13 | Round 1 |
| T+13 to T+23 | Round 2 |
| T+23 to T+33 | Round 3 |
| T+33 to T+50 | Round 4 (Config C) |
| T+50 to T+60 | buffer |

Retrieval costs NO QPU time and happens after the window. Front-load submissions: a task must be submitted inside the window to execute inside it.

## Commands

All runs use the same non-interactive heredoc form. Both line counts below were tested locally by aborting at the confirmation prompt. IMPORTANT: `known` mode takes 9 answer lines and `haar` takes 8, because haar skips the "Phase PHI" prompt. A miscount shifts every subsequent answer.

Config A known (9 lines):

```bash
python submit_qpe.py <<'EOF'
5
1
10000
both
1 3 5
known

y
y
EOF
```

Config A haar (8 lines):

```bash
python submit_qpe.py <<'EOF'
5
1
10000
both
1 3 5
haar
y
y
EOF
```

| Run | Lines | Difference from Config A |
|---|---|---|
| Config B known | 9 | line 1 is `3`, line 2 is `2` |
| Config B haar | 8 | line 1 is `3`, line 2 is `2` |
| Config C known | 9 | line 1 is `5`, line 2 is `2` |
| Config C haar | 8 | line 1 is `5`, line 2 is `2` |

Line meanings in order: N_PREC, N_TARG, N_SHOTS, MITIGATION, ZNE scales, state mode, [PHI — known mode only, blank = default 1/8], verbatim, final submit confirmation.

The scripts also accept `--yes` to skip the final confirmation, but keeping the final `y` in the heredoc is deliberately safer — if the line count is ever wrong, the last answer will probably not be `y` and the run aborts harmlessly instead of submitting a mis-configured circuit.

Do NOT change N_PREC, N_TARG or N_SHOTS within a config between batches — it breaks the comparison.

## Same-unitary guarantee

`SEED=10` and `TARGET_INIT_SEED=99` are module-level constants in `submit_qpe.py` and are NOT prompted. Every `haar` run therefore draws the identical U and identical target state across all batches, all configs of the same n_targ, and all mitigation conditions. The unitary is also written into each result's `unitary` field so the analysis can verify rather than assume.

Note U depends on n_targ (it is 2^n_targ square), so Config A's U differs from Config B's and C's — comparisons must stay within a config.

## After the window

```bash
python checkRetrieve.py job_results_qpe --watch
```

Then open `qpe_mitigation_analysis.ipynb` and set `USE_SYNTHETIC = False`.

Known post-processing caveat: the notebook's run-family key is `(n_prec, n_targ, state_mode, phi, zne_scales)`. The three batches of a given config share that key, so they collapse into one family and only the newest survives the slotting step. All JSON files are safely on disk — this only affects grouping and needs a batch dimension added to the notebook. The three configs and the two state modes DO separate correctly on their own.

## Not available this session

DDD (dynamical decoupling) is NOT reachable from `submit_qpe.py`. Its mitigation prompt accepts only `none`/`rem`/`zne`/`both`. `ErrorMitigation.apply_ddd()` exists in `error_mitig.py` but no submit script calls it. Wiring it up was not attempted mid-reservation.
