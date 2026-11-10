# QUEN_QPUF: phase-estimation quantum PUF on cloud quantum hardware

Implementation and hardware tests of a phase-estimation quantum physical unclonable function (PE-QPUF), following Ghosh et al., *Phys. Rev. Research* **6**, 043306 (2024). The scheme runs quantum phase estimation (QPE) twice on a challenge state. A response is accepted if the two phase readouts m1, m2 agree within a tolerance Δ on the cyclic distance d_c(m1, m2).

Experiments ran on Amazon Braket devices: IonQ Forte-1 and Forte-Enterprise-1, IQM Garnet and Emerald, Rigetti Ankaa-3 and Cepheus-1-108Q, and AQT IBEX-Q1. The repo also holds GHZ, QFT and QPE benchmarks and error-mitigation tests (IonQ debias; REM and ZNE via Mitiq on Rigetti).

Authors: YongHoon Lee, Wonmin Son (Quantum Enhancement Inc.)

> **Cost warning.** The `submit_*.py` scripts send jobs to real quantum hardware on AWS. This costs real money and may use shared reservations. Do not run them unless you intend to pay. Analyzing the stored results (`job_results/*.json`, notebooks, plots) needs no AWS account.

## Known caveats (read before using any data)

Negative and null results are part of the record. Please read these before trusting any number here.

1. **Inverse-QFT bit-ordering bug in the June 2026 IonQ runs.** All IonQ Forte-Enterprise-1 runs from June 2026 used a QPE circuit with an inverse-QFT bit-ordering bug (a missing `prec[::-1]`). They are `braket_direct_reserve/`, the reservation runs in `braket_direct/ionq_20260601/`, and `noise_mitigation/`. The bug was fixed in commit `35f6f28` (2026-08-28). Compare those outputs with a simulation of the circuit that was actually submitted, not with ideal QPE. The April 2026 Forte-1 runs and the Rigetti runs are unaffected.
2. **No genuine mid-circuit measurement was executed.** Qiskit transpilation moves the stage-1 measurements to the end of the circuit. By the deferred-measurement principle the ideal statistics are identical (proof in the project notes). So these data test two-stage QPE consistency. They do not test mid-circuit-measurement disturbance.
3. **Known decode bug.** `braket_direct_reserve/braket_direct_analysis.ipynb` and `braket_direct/ionq_20260601/braket_direct_analysis.ipynb`, cell 3 (`parse_outcome`, flat-register fallback), decode incorrectly. Do not reuse them. Bit-order conventions also depend on how results were retrieved: `qiskit-braket-provider` returns Qiskit order, `result.measurement_counts` returns raw Braket order. See `docs/plot_ledger/`.
4. **Some logged tasks were never retrieved.** They are listed in the plot ledger.

## Repository layout

| Path | Contents |
|---|---|
| `qpu_benchmark/` | GHZ, QFT and T-gate QPE benchmarks across devices (Apr 2026). `submit_GHZ/QFT/QPE.py`, `checkRetrieve.py`, `qpu_analysis.ipynb`, `plots/`. |
| `garnet-approxQPEQPUF-1target/` | Despite the name, this is the IonQ Forte-1 one-target two-stage QPUF sweep, d = 2..10 (Apr 2026). `submit_job.py`, `checkRetrieve_job.py`, `result_analysis.ipynb`. |
| `braket_direct_reserve/` | Plots, results and analysis notebook from the Forte-Enterprise-1 Braket Direct reservation of 2026-06-01 (multi-target). Caveats 1 and 3 apply. |
| `braket_direct/ionq_20260601/` | Submission and retrieval scripts for that reservation (`submit_QPUF_ntarg.py`, `checkRetrieve.py`), the on-grid pipeline (`on_grid_qpuf_*.py`, `two_stage_on_grid_baseline.py`, `on_grid_results/`), and the gate-count-limit probe (`probe_gate_limit.py`, `decode_mcm_qpuf_result.py`). |
| `noise_mitigation/` | IonQ debias vs raw comparison (June 2026): `ionq_noise_mitig.py`, `ionq_qpuf_two_stage.py`, notebooks, results in `job_results/` and `job_results_2stage/`. Caveat 1 applies. |
| `braket_direct/rigetti_cepheus_20260807/` | Cepheus-1-108Q reservation of 2026-08-07: QPE and QPUF with REM/ZNE (Mitiq). Has its own `README.md` and `RESERVATION_SCHEDULE.md`. |
| `braket_direct/20260828_ionq_error_mitigation/` | IonQ mitigation scaffolding (`MITIGATION_BRIEF_IONQ.md`). **Not run on hardware.** |
| `examples/` | QPE and QFT tutorial notebooks and helper modules (`utils_qpe.py`, `utils_qft.py`). |
| `quantumReadoutPUF/` | Notebook (`qrPUF.ipynb`) on a quantum readout PUF. |
| `docs/` | `LOG.md` (lab log); `plot_ledger/` (interactive ledger of all runs, open `index.html`); `papers/`. |
| `write/` | Local Overleaf clone of the manuscript. Not part of the public repo (gitignored). |

## Workflow

1. `submit_*.py` builds the circuit, submits it to a device, and appends one JSON line to `job_results/job_log.txt` (task ARN, seed, parameters, unitary). **This costs money.**
2. `checkRetrieve*.py` reads the log, fetches finished tasks from AWS, and saves each to `job_results/<uuid>.json`.
3. Notebooks and analysis scripts load the saved JSON and plot. This needs no AWS access.

The `garnet-approxQPEQPUF-1target/` log is named `jobs_log.txt` (with an `s`).

**Seeds.** The Haar-random unitary seed is logged for each run. The target initial-state seed is 99.

## Setup

```
pip install -r requirements.txt
```

Submitting or retrieving needs AWS credentials with Braket access and the relevant device or reservation. Gate counts depend on the Qiskit version. The results were produced with qiskit 2.3.0, and qiskit 2.0.0 transpiles to about 2.4x more gates. Pin the version if you need to reproduce gate counts.

## Citation

If you use this code or data, please cite: [paper in preparation]

Base protocol: S. Ghosh et al., *Phys. Rev. Research* **6**, 043306 (2024).

## License

To be decided. Contact the authors.
