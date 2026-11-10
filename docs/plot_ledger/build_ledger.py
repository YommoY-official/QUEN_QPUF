#!/usr/bin/env python3
"""
build_ledger.py -- read-only inventory of every result dataset in the repo ->
docs/plot_ledger/index.html + figs/.  No hardware access, no data is modified.

DECODE RULES (each established by a known-answer test, see kat_rows below):
  IonQ  (OpenQASM via Braket, flat c register): the raw key is in Qiskit order,
        key = c[2d-1]..c[0].  Stage m1 = c[0..d-1] with c[0] the MSB (build_qpe_stage
        is MSB-first), so m1 = int(key[d:][::-1],2), m2 = int(key[:d][::-1],2).
        Single stage: m = int(key[::-1],2).  The per-stage bit reversal is exactly the
        step missing from braket_direct_analysis.ipynb cell 3 (TODO.md s0 item 3).
  Rigetti (verbatim OpenQASM): key char i = c[i], so m1 = int(key[:d],2), m2 = int(key[d:],2).
  Garnet-approx dir (Qiskit provider): m2 = int(key[:d],2), m1 = int(key[d:],2)  (Director's KAT).
Ideal curves use the exact eigenbasis formula  P(m1,m2) = sum_j w_j p_j(m1) p_j(m2)
(valid because the stage-1 collapse is diagonal in U's eigenbasis); it is checked
against decode_mcm_qpuf_result.ideal_joint and Rigetti's logged ideal_distribution.
Gate counts are only echoed from logs, never recomputed.
"""
import glob, json, os, re, shutil, sys, html
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/plot_ledger"
FIGS = OUT / "figs"
REUSED = FIGS / "reused"
for p in (FIGS, REUSED):
    p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "braket_direct/ionq_20260601"))

IONQ, RIG = "ionq", "rigetti"
# (results dir, label)
RESULT_DIRS = [
    "qpu_benchmark/job_results",
    "garnet-approxQPEQPUF-1target/job_results",
    "braket_direct_reserve/job_results",
    "braket_direct/ionq_20260601/gate_limit_probe_results",
    "noise_mitigation/job_results",
    "noise_mitigation/job_results_2stage",
    "braket_direct/rigetti_cepheus_20260807/job_results",
    "braket_direct/rigetti_cepheus_20260807/job_results_qpe",
    "braket_direct/rigetti_cepheus_20260807/job_results_test",
]
LOG_FILES = [p for p in glob.glob(str(ROOT / "**/job*_log.txt"), recursive=True)
             if "/.claude/" not in p] + glob.glob(str(ROOT / "**/jobs_log.txt"), recursive=True)
LOG_FILES = sorted({p for p in LOG_FILES if "/.claude/" not in p})

uu = lambda arn: arn.split("/")[-1]
rel = lambda p: str(Path(p).relative_to(ROOT))


# ------------------------------------------------------------------ ideal physics
def unitary_of(res):
    u = res["unitary"]
    return np.array(u["real"]) + 1j * np.array(u["imag"])


def eig_weights(U, nt, tis):
    """phases phi_j in [0,1) and weights w_j=|<v_j|psi>|^2 for the builders' product-state init."""
    r = np.random.default_rng(tis)
    psi = np.array([1.0 + 0j])
    for _ in range(nt):                       # qubit i drawn in order; qubit 0 is the LSB of U's index
        th, ph = r.uniform(0, np.pi), r.uniform(0, 2 * np.pi)
        q = np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])
        psi = np.kron(q, psi)
    T, Z = sla.schur(U, output="complex")
    phi = (np.angle(np.diag(T)) / (2 * np.pi)) % 1.0
    w = np.abs(Z.conj().T @ psi) ** 2
    return phi, w


def kernel(d, phi):
    D = 2 ** d
    x = np.exp(2j * np.pi * np.arange(D) * phi)
    return np.abs(np.fft.fft(x) / D) ** 2     # p(m|phi), MSB-first QPE


def ideal_single(res, d):
    phi, w = eig_weights(unitary_of(res), res["n_targ"], res["target_init_seed"])
    return sum(wj * kernel(d, pj) for pj, wj in zip(phi, w))


def ideal_two_stage(res, d):
    """returns P(m1=m2) and signed-diff pmf over circular distance 0..D/2 for the ideal two-stage QPUF."""
    D = 2 ** d
    phi, w = eig_weights(unitary_of(res), res["n_targ"], res["target_init_seed"])
    diff = np.zeros(D)
    for pj, wj in zip(phi, w):
        p = kernel(d, pj)
        f = np.fft.fft(p)
        diff += wj * np.fft.ifft(np.conj(f) * f).real     # P(m2-m1 = k mod D)
    delta = np.zeros(D // 2 + 1)
    for k in range(D):
        delta[min(k, D - k)] += diff[k]
    return delta


def circ_delta(m1, m2, D):
    a = np.abs(m1 - m2)
    return np.minimum(a, D - a)


# ------------------------------------------------------------------ decoders
def two_stage_pairs(counts, d, rule):
    m1 = []; m2 = []; wt = []
    for k, c in counts.items():
        if rule == "ionq":
            a, b = int(k[d:][::-1], 2), int(k[:d][::-1], 2)
        elif rule == "ionq_direct":
            a, b = int(k[:d], 2), int(k[d:], 2)
        elif rule == "rigetti":
            a, b = int(k[:d], 2), int(k[d:], 2)
        elif rule in ("garnet", "ionq_prefix"):   # provider keys, direct m=int(stage key); key[d:]=stage 1
            a, b = int(k[d:], 2), int(k[:d], 2)
        m1.append(a); m2.append(b); wt.append(c)
    return np.array(m1), np.array(m2), np.array(wt)


def single_hist(counts, d, rule):
    H = defaultdict(int)
    for k, c in counts.items():
        k = k[:d] if len(k) > d else k
        H[int(k[::-1], 2) if rule == "ionq" else int(k, 2)] += c
    return H


# ------------------------------------------------------------------ QASM inspection
def qasm_source(res):
    a = (res.get("additional_metadata") or {}).get("action")
    return a.get("source") if isinstance(a, dict) else None


def trailing_measures(src):
    if src is None:
        return "n/a (no program stored)"
    ls = [l.strip() for l in src.split("\n")]
    idx = [i for i, l in enumerate(ls) if "measure" in l]
    if not idx:
        return "no measures found"
    skip = ("}", "//", "#pragma", "barrier")
    after = [l for l in ls[idx[0]:] if l and "measure" not in l and not l.startswith(skip)]
    return "yes" if not after and "reset" not in src else "NO"


# ------------------------------------------------------------------ load
LOG = {}
for lf in LOG_FILES:
    for line in open(lf):
        if line.strip():
            r = json.loads(line)
            arn = r.get("job_id") or r.get("task_id")
            r["_logfile"] = rel(lf)
            LOG[uu(arn)] = r

RUNS = []
for dn in RESULT_DIRS:
    for f in sorted(glob.glob(str(ROOT / dn / "*.json"))):
        res = json.load(open(f))
        if "measurement_counts" in res and "counts" not in res:
            res["counts"] = res["measurement_counts"]
        if not (res.get("job_id") or res.get("task_id") or res.get("task_metadata")):
            continue                          # analysis_summary_* etc. are derived files, not task results
        u = Path(f).stem
        RUNS.append(dict(uuid=u, dir=dn, path=rel(f), res=res, log=LOG.get(u)))
seen = {r["uuid"] for r in RUNS}
PENDING = [dict(uuid=u, dir=l["_logfile"], path=None, res=None, log=l) for u, l in LOG.items() if u not in seen]


def classify(r):
    res, log = r["res"] or {}, r["log"] or {}
    m = {**log, **res}
    ct = m.get("circuit_type", "")
    d = r["dir"]
    x = dict(device=m.get("qpu") or m.get("device") or "?", mitig="none", scale="", unitary="")
    x["n_targ"] = m.get("n_targ", ""); x["n_prec"] = m.get("n_prec", m.get("n_counting", ""))
    x["shots"] = m.get("n_shots_actual") or m.get("n_shots") or m.get("n_shots_requested")
    x["seed"] = m.get("seed", ""); x["tis"] = m.get("target_init_seed", "n/a")
    x["date"] = (m.get("datetime") or m.get("submitted_at") or "")[:19].replace("T", " ")
    if "reservation_arn" in m or m.get("submission") == "reservation":
        x["submission"] = "Braket Direct (reservation)"
    elif m.get("submission"):
        x["submission"] = m["submission"]
    else:
        x["submission"] = "not logged"
    if "gate_limit" in d:
        x["submission"] = "Braket Direct (per file note; no log entry)"
    x["gates"] = ""
    if m.get("n_gates") is not None or m.get("n_1q_gates") is not None:
        x["gates"] = f"{m.get('n_gates','?')} total ({m.get('n_1q_gates','?')} 1q / {m.get('n_2q_gates','?')} 2q)"
        if m.get("depth"):
            x["gates"] += f", depth {m['depth']}"
    em = m.get("error_mitigation", "none")
    if em and em != "none":
        x["mitig"] = em
    if m.get("zne_scale") not in (None, ""):
        x["scale"] = m["zne_scale"]
    if d.startswith("qpu_benchmark"):
        x["kind"] = {"GHZ": "ghz", "QFT": "qft", "QPE": "qpe_t"}[ct]
        x["exp"] = {"GHZ": "GHZ", "QFT": "QFT", "QPE": "Single-stage QPE (T gate)"}[ct]
        x["nq"] = m.get("n_qubits")
        x["unitary"] = "known: T gate (phi=1/8), eigenstate |1>" if ct == "QPE" else "none"
        if ct == "QPE":
            x["n_prec"] = m.get("n_counting", x["nq"] - 1); x["n_targ"] = 1
        x["rule"] = "direct (Braket Circuit API, char i = qubit i)"
    elif d.startswith("garnet"):
        x["kind"] = "qpuf2"; x["exp"] = "Two-stage QPE-QPUF"; x["n_targ"] = 1
        x["nq"] = 1 + 2 * m["n_prec"]; x["rule"] = "garnet"
        x["unitary"] = f"Haar-random 2x2, seed {m['seed']} (ZYZ angles logged)"
        x["tis"] = "n/a (not logged)"
    elif "gate_limit" in d or d in ("braket_direct_reserve/job_results", "braket_direct/ionq_20260601/job_results/job_log.txt"):
        x["kind"] = "qpuf2"; x["exp"] = "Two-stage QPE-QPUF"; x["rule"] = "ionq_prefix"
        x["nq"] = x["n_targ"] + 2 * x["n_prec"] if x["n_targ"] != "" else ""
        x["unitary"] = f"Haar-random {2**x['n_targ']}x{2**x['n_targ']}, seed {x['seed']}" if x["n_targ"] != "" else ""
        if "gate_limit" in d:
            x["rule"] = "ionq_direct"   # saved raw from AwsQuantumTask (no qiskit provider), so char i = c[i]
            tm = res["task_metadata"]; cp = res["circuit_params"]
            x.update(device="Forte-Enterprise-1", n_targ=cp["n_targ"], n_prec=cp["n_prec"], seed=cp["seed"],
                     tis=cp["target_init_seed"], shots=tm["shots"], nq=cp["n_targ"] + 2 * cp["n_prec"],
                     date=(tm.get("createdAt") or "")[:19].replace("T", " "),
                     unitary=f"Haar-random 8x8, seed {cp['seed']}", gates="1915 (per TODO.md s0.1; not in file)")
    elif d.startswith("noise_mitigation"):
        two = "2stage" in d
        x["kind"] = "qpuf2" if two else "qpe1"
        x["exp"] = "Two-stage QPE-QPUF" if two else "Single-stage QPE (Haar U)"
        x["nq"] = m.get("n_qubits") or x["n_targ"] + (2 if two else 1) * x["n_prec"]
        x["unitary"] = f"Haar-random {2**x['n_targ']}x{2**x['n_targ']}, seed {x['seed']}"
        x["rule"] = "ionq_prefix"
        if em == "debias":
            x["mitig"] = "debias"
    else:  # rigetti
        lab = m.get("label", "")
        x["nq"] = m.get("n_qubits")
        if "calib" in lab:
            x["kind"] = "cal"; x["exp"] = "Readout calibration (REM)"
        elif ct.startswith("QPUF2stage"):
            x["kind"] = "qpuf2"; x["exp"] = "Two-stage QPE-QPUF" + (" (TEST)" if "TEST" in ct else "")
        else:
            x["kind"] = "qpe1"; x["exp"] = "Single-stage QPE"
        sm = m.get("state_mode")
        if sm == "known":
            x["unitary"] = f"known: phi={m.get('phi')} (ideal bin {m.get('ideal_bin')})"
        else:
            x["unitary"] = f"Haar-random {2**x['n_targ']}x{2**x['n_targ']}, seed {x['seed']}" if "unitary" in m else ""
        x["rule"] = "rigetti"
        if x["mitig"] == "none" and x["scale"] != "":
            x["mitig"] = "none"
        x["label"] = lab
    x["rule"] = x.get("rule", "")
    x["trail"] = trailing_measures(qasm_source(res)) if res else ""
    if "gate_limit" in d:
        x["trail"] = "n/a (source not stored; TODO.md s0.1 says yes)"
    # circuit version vs the prec[::-1] fix (commit 35f6f28, 2026-08-28): decided by script lineage + submission date
    if "gate_limit" in d:
        x["cver"] = "post-fix (inferred from date, no log entry)"
        x["cver_how"] = f"run date {x['date'][:10]} is after fix commit 35f6f28 (2026-08-28); no log entry and no stored QASM, so the code version is inferred, not recorded"
    elif x["rule"] == "ionq_prefix":
        x["cver"] = "pre-fix"
        x["cver_how"] = (f"submitted {x['date'][:10] or '(log date)'} < 2026-08-28; the builder (submit_QPUF_ntarg.py / ionq_noise_mitig.py) only gained prec[::-1] in 35f6f28; "
                         "KAT: hardware matches the noiseless sim of the pre-fix circuit (see checks table)")
    elif x["rule"] == "garnet":
        x["cver"] = "n/a-unaffected"
        x["cver_how"] = "garnet submit_job.py: LSB-first controlled powers with little-endian QFT-dagger (self-consistent, no reversal needed); April 2026"
    elif x["rule"] == "rigetti":
        x["cver"] = "n/a-unaffected"
        x["cver_how"] = "Rigetti builder already carried the reversal (35f6f28 message: 'already verified on Rigetti'); run 2026-08-07"
    else:
        x["cver"] = "n/a"
        x["cver_how"] = "qpu_benchmark circuits built with the Braket Circuit API (separate builder, not examined for this issue)"
    return x


# ------------------------------------------------------------------ plotting
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.spines.top": False, "axes.spines.right": False})
C_HW, C_ID, C_UNI = "#1f6fb2", "#d1495b", "#6b6b6b"


def save(fig, name):
    p = FIGS / name
    fig.tight_layout()
    fig.savefig(p, dpi=75)
    plt.close(fig)
    return f"figs/{name}"


MAXQ = 14   # statevector sim cap for the pre-fix ideal overlay


def prefix_sim(res, x, two):
    """noiseless sim of the circuit that was actually submitted (pre-fix: no prec[::-1]); keys c[n-1..0]."""
    d, nt = int(x["n_prec"]), int(x["n_targ"])
    if nt + (2 * d if two else d) > MAXQ:
        return None
    from make_notes_figs import sim_counts
    return sim_counts(d, nt, unitary_of(res), int(res["target_init_seed"]), two, False)


def plot_two_stage(r, x, name):
    res = r["res"]; d = x["n_prec"]; D = 2 ** d
    m1, m2, wt = two_stage_pairs(res["counts"], d, x["rule"])
    n = wt.sum(); dl = circ_delta(m1, m2, D)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    if d <= 5:
        M = np.zeros((D, D))
        np.add.at(M, (m1, m2), wt)
        im = ax[0].imshow(M.T, origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax[0], fraction=0.046)
    else:
        ax[0].scatter(m1, m2, s=np.clip(wt * 6, 4, 40), alpha=0.5, color=C_HW, lw=0)
        ax[0].set_xlim(-0.5, D - 0.5); ax[0].set_ylim(-0.5, D - 0.5)
    ax[0].set_xlabel("m1 (stage 1)"); ax[0].set_ylabel("m2 (stage 2)")
    ax[0].set_title(f"coincidence map  P(m1=m2)={wt[m1==m2].sum()/n:.3f}")
    K = min(D // 2, 12); ds = np.arange(K + 1)
    acc = np.array([wt[dl <= k].sum() / n for k in ds])
    ax[1].plot(ds, acc, "o-", color=C_HW, label="hardware")
    ax[1].plot(ds, np.minimum((2 * ds + 1) / D, 1), "--", color=C_UNI, label="uniform guess (2Δ+1)/2^d")
    try:
        if x["rule"] == "ionq_prefix":
            sm = prefix_sim(res, x, True)
            if sm:
                a, b, p = two_stage_pairs(sm, d, "ionq_prefix"); dd = circ_delta(a, b, D)
                ax[1].plot(ds, [p[dd <= k].sum() / p.sum() for k in ds], "s:", color=C_ID, label="noiseless sim of submitted (pre-fix) circuit")
        elif x["rule"] != "garnet":
            dm = ideal_two_stage(res, d)
            ax[1].plot(ds, np.cumsum(dm)[: K + 1], "s:", color=C_ID, label="ideal (noiseless)")
    except Exception:
        pass
    ax[1].set_xlabel("Δ (circular distance)"); ax[1].set_ylabel("acceptance P(dist ≤ Δ)")
    ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=6.5, loc="lower right"); ax[1].set_title(f"acceptance vs Δ  (d={d}, {int(n)} shots)")
    return save(fig, name)


def plot_single(r, x, name, ideal=None, ideal_bin=None, ideal_label="ideal (noiseless)"):
    res = r["res"]; d = x["n_prec"]
    H = single_hist(res["counts"], d, "rig")   # direct m=int(key) for every path now (pre-fix IonQ matches direct)
    n = sum(H.values())
    keys = set(sorted(H, key=H.get, reverse=True)[:20])
    if ideal is not None:
        keys |= set(np.argsort(ideal)[::-1][:3].tolist())
    if ideal_bin is not None:
        keys.add(int(ideal_bin))
    ks = sorted(keys)
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.bar(range(len(ks)), [H.get(k, 0) / n for k in ks], color=C_HW, label="hardware")
    if ideal is not None:
        ax.plot(range(len(ks)), [ideal[k] for k in ks], "v", color=C_ID, label=ideal_label)
    if ideal_bin is not None and int(ideal_bin) in ks:
        ax.axvline(ks.index(int(ideal_bin)), color=C_ID, lw=0.8, ls="--")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=90, fontsize=6)
    ax.set_ylabel("probability"); ax.legend(fontsize=6.5)
    ax.set_title(f"outcome histogram (top bins + ideal peaks), d={d}, {n} shots; uniform = {1/2**d:.2g}/bin")
    return save(fig, name)


def plot_ghz(r, x, name):
    res = r["res"]; c = res["counts"]; n = x["nq"]; N = sum(c.values())
    hw = defaultdict(int)
    for k, v in c.items():
        hw[k.count("1")] += v
    F = (c.get("0" * n, 0) + c.get("1" * n, 0)) / N
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    top = sorted(c.items(), key=lambda kv: -kv[1])[:8]
    lab = ["0^n" if k == "0" * n else "1^n" if k == "1" * n else k[:6] + "…" for k, _ in top]
    ax[0].bar(range(len(top)), [v / N for _, v in top], color=[C_ID if l in ("0^n", "1^n") else C_HW for l in lab])
    ax[0].set_xticks(range(len(top))); ax[0].set_xticklabels(lab, rotation=60, fontsize=6)
    ax[0].set_title(f"top bitstrings;  P(0^n)+P(1^n) = {F:.3f}")
    ax[1].bar(list(hw), [v / N for v in hw.values()], color=C_HW)
    ax[1].set_xlim(-1, n + 1); ax[1].set_xlabel("Hamming weight"); ax[1].set_title(f"weight distribution (ideal: 0 and {n}), n={n}")
    return save(fig, name)


def plot_qft(r, x, name):
    from scipy.stats import binom
    res = r["res"]; c = res["counts"]; n = x["nq"]; N = sum(c.values())
    hw = np.zeros(n + 1)
    for k, v in c.items():
        hw[k.count("1")] += v
    hw /= N; b = binom.pmf(np.arange(n + 1), n, 0.5)
    q1 = np.zeros(n)
    for k, v in c.items():
        q1 += np.frombuffer(k.encode(), dtype=np.uint8) - 48 if len(k) == n else 0
        q1 += 0
    q1 = np.array([sum(v for k, v in c.items() if k[i] == "1") for i in range(n)]) / N
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax[0].bar(np.arange(n + 1), hw, color=C_HW, label="hardware"); ax[0].plot(np.arange(n + 1), b, "v", color=C_ID, label="ideal: Binomial(n,½)")
    ax[0].set_xlabel("Hamming weight"); ax[0].legend(fontsize=6.5); ax[0].set_title(f"QFT|0..0> weight dist, TVD={0.5*abs(hw-b).sum():.3f}")
    ax[1].bar(range(n), q1, color=C_HW); ax[1].axhline(0.5, color=C_ID, ls="--"); ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("qubit (char index)"); ax[1].set_title("per-qubit P(1)  (ideal 0.5)")
    return save(fig, name)


def plot_cal(r, x, name):
    c = r["res"]["counts"]; N = sum(c.values()); n = len(next(iter(c)))
    q1 = np.array([sum(v for k, v in c.items() if k[i] == "1") for i in range(n)]) / N
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.bar(range(n), q1, color=C_HW); ax.set_ylim(0, 1); ax.set_xlabel("clbit"); ax.set_ylabel("P(1)")
    ax.set_title(f"{x.get('label','calibration')}: per-bit P(1)  ({N} shots)")
    return save(fig, name)


def qpe_t_ideal_stats(r, x):
    c = r["res"]["counts"]; nc = x["n_prec"]; N = sum(c.values()); ib = round(2 ** nc / 8)
    fd = sum(v for k, v in c.items() if int(k[:nc], 2) == ib) / N
    fr = sum(v for k, v in c.items() if int(k[:nc][::-1], 2) == ib) / N
    return ib, fd, fr


def make_plot(r, x):
    k, u = x["kind"], r["uuid"][:8]
    nm = f"{k}_{u}.png"
    try:
        if k == "ghz": return plot_ghz(r, x, nm)
        if k == "qft": return plot_qft(r, x, nm)
        if k == "cal": return plot_cal(r, x, nm)
        if k == "qpuf2": return plot_two_stage(r, x, nm)
        if k == "qpe_t":
            ib, fd, fr = qpe_t_ideal_stats(r, x)
            x["extra"] = f"P(ideal bin {ib}) direct={fd:.4f}, bit-reversed counting reg={fr:.4f}"
            return plot_single(r, x, nm, ideal=None, ideal_bin=ib) if x["n_prec"] <= 20 else plot_single(r, x, nm, ideal_bin=ib)
        if k == "qpe1":
            res = r["res"]; d = x["n_prec"]
            ideal = None
            if res.get("ideal_distribution"):
                ideal = np.zeros(2 ** d)
                for kk, v in res["ideal_distribution"].items():
                    ideal[int(kk)] = v
            elif x["rule"] == "ionq_prefix":
                sm = prefix_sim(res, x, False)
                if sm:
                    ideal = np.zeros(2 ** d)
                    for kk, v in sm.items(): ideal[int(kk, 2)] += v
                    return plot_single(r, x, nm, ideal=ideal, ideal_label="noiseless sim of submitted (pre-fix) circuit")
            elif "unitary" in res:
                ideal = ideal_single(res, d)
            ib = res.get("ideal_bin")
            return plot_single(r, x, nm, ideal=ideal, ideal_bin=ib)
    except Exception as e:
        x["extra"] = f"plot failed: {e!r}"
    return None


# ------------------------------------------------------------------ existing plots (reused, copied)
def existing_for(r, x):
    d, res = r["dir"], r["res"] or {}
    out = []
    def add(p, note=""):
        p = Path(p)
        if p.exists():
            dst = REUSED / (rel(p).replace("/", "__"))
            if not dst.exists():
                shutil.copy(p, dst)
            out.append((f"figs/reused/{dst.name}", rel(p) + note))
    if d.startswith("qpu_benchmark"):
        dev = re.sub(r"[^A-Za-z0-9]+", "_", x["device"]).strip("_")
        sub = {"ghz": "ghz", "qft": "qft", "qpe_t": "qpe_target1"}[x["kind"]]
        n = x["nq"]
        add(ROOT / f"qpu_benchmark/plots/{sub}/{dev}_{sub.split('_')[0]}_n{n}.png",
            "  (per device/type/n; may summarise a sibling run)")
    elif d.startswith("garnet"):
        add(ROOT / f"garnet-approxQPEQPUF-1target/plots/{r['uuid']}.png")
    elif d == "braket_direct_reserve/job_results":
        for p in glob.glob(str(ROOT / f"braket_direct_reserve/plots/*{r['uuid'][:8]}*.png")):
            add(p)
    elif "gate_limit" in d:
        add(ROOT / "braket_direct/ionq_20260601/gate_limit_probe_results/plots/mcm_qpuf_joint_observed_vs_ideal.png")
    elif d.startswith("noise_mitigation"):
        pat = f"nprec{x['n_prec']}_ntarg{x['n_targ']}_{x['shots']}shots"
        for p in sorted(glob.glob(str(ROOT / f"noise_mitigation/plots/*{pat}*.png"))):
            if ("2stage" in d) == ("qpuf_2stage" in p):
                add(p)
    return out


# ------------------------------------------------------------------ known-answer tests
def bc(P, Q):
    return float(np.sum(np.sqrt(P * Q)))


def joint_hw(res, d, rule):
    D = 2 ** d; M = np.zeros((D, D))
    m1, m2, w = two_stage_pairs(res["counts"], d, rule)
    np.add.at(M, (m1, m2), w)
    return M / M.sum()


def ideal_joint_matrix(res, d):
    D = 2 ** d
    phi, w = eig_weights(unitary_of(res), res["n_targ"], res["target_init_seed"])
    P = [kernel(d, p) for p in phi]
    return sum(wj * np.outer(p, p) for wj, p in zip(w, P))


PROBE_U = {}


def kat():
    global OKRUNS
    OKRUNS = [q for q in RUNS if q["res"].get("counts")]
    rows = []
    # (a) eigenbasis ideal vs branch-exact qiskit ideal_joint
    try:
        from decode_mcm_qpuf_result import ideal_joint
        for (d, nt, sd) in [(3, 2, 100), (3, 3, 10)]:
            f = [q for q in OKRUNS if q["res"] and q["res"].get("n_prec") == d and q["res"].get("n_targ") == nt and q["res"].get("seed") == sd and "unitary" in q["res"]]
            if f:
                res = f[0]["res"]
                J = ideal_joint(d, nt, sd, 99, U=unitary_of(res)); J2 = ideal_joint_matrix(res, d)
                rows.append(("ideal formula vs ideal_joint", f"d={d} n_targ={nt} seed={sd}", f"max|ΔP|={abs(J-J2).max():.2e}; P(m1=m2)={np.trace(J2):.4f}"))
    except Exception as e:
        rows.append(("ideal formula vs ideal_joint", "skipped", repr(e)))
    # (b) Rigetti logged ideal_distribution vs eigenbasis formula
    errs = []
    for q in OKRUNS:
        res = q["res"]
        if res.get("ideal_distribution") and res.get("state_mode") == "haar" and res["n_prec"] <= 8:
            P = np.zeros(2 ** res["n_prec"])
            for k, v in res["ideal_distribution"].items(): P[int(k)] = v
            errs.append(abs(P - ideal_single(res, res["n_prec"])).max())
    if errs:
        rows.append(("ideal formula vs Rigetti logged ideal_distribution", f"{len(errs)} Haar runs", f"max|ΔP|={max(errs):.2e}"))
    # (c) IonQ pre-fix KAT: TVD of hardware counts vs noiseless sim of the PRE-FIX and the FIXED circuit, each under direct / per-stage-reversed decode
    rows.append(("SUPERSEDED earlier ledger KAT", "IonQ provider-path runs, per-stage-reversed decode vs fixed-circuit ideal",
                 "an earlier version of this ledger reported overlap 0.80/0.82 (reversed) vs 0.61/0.64 (direct) against the FIXED-circuit ideal and concluded 'reversed'. That was an artefact: those runs executed the PRE-FIX circuit (see next rows). Direct decode + pre-fix sim is the correct pair."))
    from make_notes_figs import sim_counts
    def tv(sim, cn, f):
        b = defaultdict(float); t = sum(cn.values())
        for k, c in cn.items(): b[f(k)] += c / t
        return 0.5 * sum(abs(sim.get(k, 0) - b.get(k, 0)) for k in set(sim) | set(b))
    for q in OKRUNS:
        res, x = q["res"], q["x"]
        if x["rule"] != "ionq_prefix" or "unitary" not in res:
            continue
        d, nt = int(x["n_prec"]), int(x["n_targ"]); two = x["kind"] == "qpuf2"
        tag = f"{q['uuid'][:8]} {'two' if two else 'single'}-stage d={d} n_targ={nt} {x['mitig']} shots={x['shots']} {x['date'][:10]}"
        if nt + (2 * d if two else d) > MAXQ:
            rows.append(("IonQ pre-fix KAT (TVD)", tag, f"not computed: {nt + (2*d if two else d)} qubits > {MAXQ} (statevector cap); circuit version still pre-fix by date/script")); continue
        U = unitary_of(res); ti = int(res["target_init_seed"]); cn = res["counts"]
        sp = sim_counts(d, nt, U, ti, two, False); sf = sim_counts(d, nt, U, ti, two, True)
        rv = (lambda k, d=d: k[:d][::-1] + k[d:][::-1]) if two else (lambda k: k[::-1])
        t = dict(pd=tv(sp, cn, lambda k: k), pr=tv(sp, cn, rv), fd=tv(sf, cn, lambda k: k), fr=tv(sf, cn, rv))
        best = min(t, key=t.get)
        rows.append(("IonQ pre-fix KAT (TVD)", tag, f"vs PRE-FIX sim: direct={t['pd']:.3f} reversed={t['pr']:.3f} | vs FIXED sim: direct={t['fd']:.3f} reversed={t['fr']:.3f} | best = {dict(pd='pre-fix+direct', pr='pre-fix+reversed', fd='fixed+direct', fr='fixed+reversed')[best]}"))
    # (c2) gate-limit probe file has no unitary: regenerate it from seed 10 with the repo's own Haar sampler
    try:
        from submit_QPUF_ntarg import haar_random_unitary
        pr = json.load(open(ROOT / "braket_direct/ionq_20260601/gate_limit_probe_results/0ede858e-d8ba-44a5-bdbc-14154b5aeeaa.json"))
        U = haar_random_unitary(8, rng=np.random.default_rng(10))
        fake = dict(unitary=dict(real=U.real.tolist(), imag=U.imag.tolist()), n_targ=3, target_init_seed=99, counts=pr["measurement_counts"])
        J = ideal_joint_matrix(fake, 3)
        rows.append(("IonQ two-stage decode", "0ede858e (gate-limit probe) d=3 n_targ=3 seed=10 shots=100; U regenerated from seed",
                     f"BC: per-stage-reversed={bc(joint_hw(fake,3,'ionq'),J):.3f}  vs  direct={bc(joint_hw(fake,3,'ionq_direct'),J):.3f}; ideal P(m1=m2)={np.trace(J):.4f}"))
        PROBE_U["U"] = fake
    except Exception as e:
        rows.append(("IonQ two-stage decode", "0ede858e", "skipped: " + repr(e)))
    # (d) simulator on-grid noiseless counts: every decoded bin must be in ideal_bins
    try:
        og = ROOT / "braket_direct/ionq_20260601/on_grid_results/2026-09-08"
        nl = json.load(open(og / "noiseless_results.json")); nz = json.load(open(og / "noisy_results.json"))
        bins = set(nl["m_j"]) | set(nz["ideal_bins"]); d = nl["n_prec"]
        cnt = nl["sampled"]["10000"]["counts"]
        for rule in ("ionq", "ionq_direct"):
            m1, m2, w = two_stage_pairs(cnt, d, rule)
            inb = w[[a in set(nz["ideal_bins"]) and b in set(nz["ideal_bins"]) for a, b in zip(m1, m2)]].sum() / w.sum()
            rows.append(("on-grid noiseless SIMULATOR decode", f"d={d}, 10000 shots, ideal_bins={sorted(nz['ideal_bins'])}", f"{rule}: fraction of shots with m1,m2 both in ideal bins = {inb:.3f}"))
    except Exception as e:
        rows.append(("on-grid sim decode", "skipped", repr(e)))
    # (e) Garnet: Director's numbers
    for q in OKRUNS:
        x = q["x"]
        if q["dir"].startswith("garnet") and x["n_prec"] in (4, 10) and x["seed"] == 100:
            d = x["n_prec"]; m1, m2, w = two_stage_pairs(q["res"]["counts"], d, "garnet"); dl = circ_delta(m1, m2, 2 ** d); n = w.sum()
            rows.append(("Garnet-dir decode (Director KAT)", f"{q['uuid'][:8]} d={d} seed=100 shots={int(n)}",
                         f"P(Δ=0)={w[dl==0].sum()/n:.3f}, P(Δ≤2)={w[dl<=2].sum()/n:.3f}   (Director: d=4 0.476/0.810; d=10 P(Δ≤2)=0.570)"))
    # (f) Rigetti single-stage: TVD vs logged ideal, direct vs reversed, d<=4
    acc = defaultdict(list)
    for q in OKRUNS:
        res, x = q["res"], q["x"]
        if x["kind"] == "qpe1" and x["rule"] == "rigetti" and res.get("ideal_distribution") and x["n_prec"] <= 4 and res.get("zne_scale") == 1:
            d = x["n_prec"]; P = np.zeros(2 ** d)
            for k, v in res["ideal_distribution"].items(): P[int(k)] = v
            tv = {}
            for rl in ("rig", "ionq"):
                H = single_hist(res["counts"], d, rl); h = np.zeros(2 ** d)
                for k, v in H.items(): h[k] = v
                tv[rl] = 0.5 * abs(h / h.sum() - P).sum()
            acc[d].append((tv["rig"], tv["ionq"]))
    for d, v in sorted(acc.items()):
        a = np.array(v).mean(0)
        rows.append(("Rigetti single-stage decode", f"d={d}, {len(v)} runs (λ=1)", f"mean TVD vs ideal: direct={a[0]:.3f}  vs  reversed={a[1]:.3f}"))
    # (g) qpu_benchmark QPE
    acc = []
    for q in OKRUNS:
        if q["x"]["kind"] == "qpe_t":
            ib, fd, fr = qpe_t_ideal_stats(q, q["x"]) if False else qpe_t_ideal_stats(q, q["x"])
            acc.append((q["x"]["device"], q["x"]["nq"], fd, fr, 1 / 2 ** q["x"]["n_prec"]))
    rows.append(("qpu_benchmark QPE ideal-bin mass", f"{len(acc)} runs",
                 f"mean P(ideal) direct={np.mean([a[2] for a in acc]):.4f}, reversed={np.mean([a[3] for a in acc]):.4f}; best single run direct={max(a[2] for a in acc):.3f} on {max(acc,key=lambda a:a[2])[0]} n={max(acc,key=lambda a:a[2])[1]}"))
    return rows


# ------------------------------------------------------------------ main
def flat(m):
    skip = {"counts", "unitary", "additional_metadata", "ideal_distribution", "measurement_counts", "placement", "_logfile"}
    out = {}
    for k, v in m.items():
        if k in skip:
            if k == "unitary" and v:
                out["unitary"] = f"shape {v.get('shape')} (matrix in result JSON)"
            continue
        s = json.dumps(v) if not isinstance(v, str) else v
        out[k] = s if len(s) < 150 else s[:147] + "…"
    return out


def main():
    for q in RUNS + PENDING:
        q["x"] = classify(q)
    kat_rows = kat()
    for q in RUNS:
        if "gate_limit" in q["dir"] and "U" in PROBE_U:
            q["res"]["unitary"] = PROBE_U["U"]["unitary"]; q["res"]["n_targ"] = 3; q["res"]["target_init_seed"] = 99
    for q in RUNS:
        x = q["x"]
        x["plot"] = make_plot(q, x) if q["res"].get("counts") else None
        if not q["res"].get("counts"): x["extra"] = "task status in result JSON: " + str(q["res"].get("status"))
        if q["res"].get("measurement_counts") and "counts" not in q["res"]:
            q["res"]["counts"] = q["res"]["measurement_counts"]
            x["plot"] = make_plot(q, x)
        x["existing"] = existing_for(q, x)
    # simulator / prepared datasets
    og = "braket_direct/ionq_20260601/on_grid_results/2026-09-08"
    sims = []
    nz = json.load(open(ROOT / og / "noisy_results.json")); nl = json.load(open(ROOT / og / "noiseless_results.json"))
    mi = json.load(open(ROOT / og / "mitigate/mitigate_results.json")); mf = json.load(open(ROOT / og / "submit/on_grid_seed9_manifest.json"))
    base = dict(device="simulator", n_targ=3, n_prec=4, nq=11, seed=9, tis=99, submission="n/a (simulator)", date="2026-09-08",
                unitary="on-grid diagonal (seed 9, m_j=[6,13,15,4,1,9,10,12])", trail="n/a (simulated)", rule="ionq (sim keys = Qiskit order; KAT above)", cver="post-fix (simulator, current code)", cver_how="simulated with the current (fixed) builder",
                exp="Two-stage QPE-QPUF (on-grid, simulator)", kind="sim")
    def img(*names):
        out = []
        for n in names:
            p = ROOT / og / n
            if p.exists():
                dst = REUSED / (rel(p).replace("/", "__")); shutil.copy(p, dst); out.append((f"figs/reused/{dst.name}", rel(p)))
        return out
    sims.append(dict(base, label="noiseless sampled", mitig="none", shots="1000 and 10000", gates=f"{nl['gate_count']['total']} total ({nl['gate_count']['n_1q']} 1q/{nl['gate_count']['n_2q']} 2q) logged", existing=img("on_grid_computational_eigenbasis_circle_plot.png"), uuid="sim-noiseless", path=f"{og}/noiseless_results.json", extra=f"P(m1=m2) sampled = {nl['sampled']['1000']['p_m1_eq_m2']}"))
    for arm, v in nz["arms"].items():
        sims.append(dict(base, label=f"noisy sim arm {arm}", mitig="none (noise sweep)", shots=v["n_shots"], gates=f"{nz['gate_count_noisy_transpile']['total']} total as logged (local transpile; braket-arm reference {nz['gate_count_braket_arm_reference_opt1']})",
                         existing=img("on_grid_noisy_nominal_circle_plot.png", "on_grid_noisy_sensitivity_circle_plot.png"), uuid=f"sim-noisy-{arm}", path=f"{og}/noisy_results.json",
                         extra=f"P(m1=m2)={v['p_m1_eq_m2']['p']}, P(m1 in ideal)={v['p_m1_in_ideal']['p']}, seed {nz['arm_seeds'].get(arm if arm!='f2q_0.9882' else 'nominal_10000')}"))
    sims.append(dict(base, label="mitigation sim (raw/REM/ZNE/DDD)", mitig="REM, ZNE scales 1/1.5/2, DDD", shots=mi["shots"], gates="raw 1908; ZNE1.5 2862; ZNE2 3816 (as logged)", existing=img("mitigate/on_grid_mitigation_circle_plots.png"), uuid="sim-mitigate", path=f"{og}/mitigate/mitigate_results.json",
                     extra=f"raw P(m1=m2)={mi['results_point']['raw']['p_m1_eq_m2']:.3f}; REM cond. number {mi['rem_condition_number']:.3f}"))
    sims.append(dict(base, label="hardware submission PREPARED, not submitted", mitig="none", shots=mf["shots"], device="Forte-Enterprise-1 (target)", submission="not submitted (manifest only)", gates=f"{mf['gate_counts']['total']} total as logged in manifest", existing=[], uuid="prep-manifest", path=f"{og}/submit/on_grid_seed9_manifest.json", extra="No task ARN: no hardware task exists for this manifest."))
    # mitigation-dir with no data
    # ---- aggregate Rigetti plots
    agg = []
    rp = ROOT / "braket_direct/rigetti_cepheus_20260807/plots"
    for p in sorted(rp.glob("*.png")) + sorted((ROOT / "noise_mitigation/plots").glob("noise_mitigation_summary.png")) + sorted((ROOT / "braket_direct_reserve/plots").glob("noise_mitigation_summary.png")):
        dst = REUSED / (rel(p).replace("/", "__")); shutil.copy(p, dst); agg.append((f"figs/reused/{dst.name}", rel(p)))
    # ---- assemble records
    rows = []
    for q in RUNS:
        x, res, log = q["x"], q["res"], q["log"] or {}
        rows.append(dict(uuid=q["uuid"], exp=x["exp"], device=x["device"], n_targ=x["n_targ"], n_prec=x["n_prec"], nq=x["nq"], unitary=x["unitary"], shots=x["shots"],
                         seed=x["seed"], tis=x["tis"], submission=x["submission"], date=x["date"], mitig=x["mitig"], scale=x["scale"], gates=x["gates"], cver=x["cver"], cver_how=x["cver_how"],
                         result="yes" if res.get("counts") else "FAILED task (" + str(res.get("status")) + "), no counts", trail=x["trail"], decode=x["rule"], plot=("generated" if x["plot"] else "none") + (" + existing" if x["existing"] else ""),
                         src=q["path"], log=("joined: " + log["_logfile"]) if log else "NO log entry", kind=x["kind"],
                         params=flat({**log, **{k: v for k, v in res.items()}}), img=x["plot"], existing=x["existing"], extra=x.get("extra", ""), label=x.get("label", "")))
    for s in sims:
        rows.append(dict(uuid=s["uuid"], exp=s["exp"], device=s["device"], n_targ=s["n_targ"], n_prec=s["n_prec"], nq=s["nq"], unitary=s["unitary"], shots=s["shots"], seed=s["seed"],
                         tis=s["tis"], submission=s["submission"], date=s["date"], mitig=s["mitig"], scale="", gates=s["gates"], cver=s["cver"], cver_how=s["cver_how"], result="yes (simulator)" if "PREPARED" not in s["label"] else "no (never submitted)",
                         trail=s["trail"], decode=s["rule"], plot="existing" if s["existing"] else "none", src=s["path"], log="n/a", kind="sim", params={"label": s["label"], "seed": s["seed"], "target_init_seed": 99, "m_j": "[6,13,15,4,1,9,10,12]"},
                         img=None, existing=s["existing"], extra=s["extra"], label=s["label"]))
    for q in PENDING:
        x, log = q["x"], q["log"]
        rows.append(dict(uuid=q["uuid"], exp=x["exp"] or log.get("circuit_type", "?"), device=x["device"], n_targ=x["n_targ"], n_prec=x["n_prec"], nq=x.get("nq", ""), unitary=x["unitary"], shots=x["shots"],
                         seed=x["seed"], tis=x["tis"], submission=x["submission"], date=x["date"], mitig=x["mitig"], scale=x["scale"], gates=x["gates"], cver=x["cver"], cver_how=x["cver_how"], result="NO - pending/not retrieved",
                         trail="n/a", decode="", plot="none", src=None, log="log only: " + log["_logfile"], kind="pending", params=flat(log), img=None, existing=[], extra="", label=""))
    kinds = Counter(r["kind"] for r in rows)
    print("rows", len(rows), dict(kinds), "pending", len(PENDING))
    print("plots generated", sum(1 for r in rows if r["img"]), "existing attached to", sum(1 for r in rows if r["existing"]))
    for k in kat_rows: print("KAT", k)
    json.dump(dict(rows=rows, kat=kat_rows, agg=agg), open(OUT / "ledger_data.json", "w"), default=str)


if __name__ == "__main__":
    main()
