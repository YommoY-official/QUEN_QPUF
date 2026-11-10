"""Figures for the Overleaf notes (images/new_results). Read-only on data; no hardware.

Decode finding (this file's reason to exist): IonQ runs submitted before 2026-08-28
(braket_direct_reserve, noise_mitigation) ran QPE stages WITHOUT the prec[::-1] fix
(commit 35f6f28). Their hardware keys match the noiseless sim of that *pre-fix* circuit
under the DIRECT decode (m = int(stage key)), not the fixed circuit under any decode
(KAT below prints TVDs). So the 'ideal' for those runs is the pre-fix circuit simulated
noiselessly, rebuilt here by omitting the reversal.
Garnet-dir (Forte-1, April) circuits are LSB-first controls + little-endian QFT^dag: correct;
m = int(stage key), key[:d]=stage 2, key[d:]=stage 1 (Director KAT reproduced).
Rigetti (Aug 7) code is post-fix; direct decode, char i = c[i].
"""
import ast, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.quantum_info import Statevector
import scipy.linalg as sla

ROOT = Path("/Users/yonnoylee/python/QUEN_QPUF")
OUT = ROOT / "write/quantumpuf_26/69aa9a0d04b3f1b2cb30cc36/images/new_results"
OUT.mkdir(exist_ok=True)
META = {}
C = dict(sim="#1f77b4", raw="#d6572b", deb="#2a9d4a", ideal="#222222", uni="#888888")
plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def J(p):
    return json.load(open(p))


def find(root, pre):
    return glob.glob(str(ROOT / root / f"{pre}*.json"))[0]


def U_of(r):
    u = r["unitary"]
    return np.array(u["real"]) + 1j * np.array(u["imag"])


# ---------- pre-fix circuit (no prec[::-1]) and noiseless simulation ----------
def stage(d, U, fixed):
    nt = int(round(np.log2(U.shape[0])))
    qc = QuantumCircuit(d + nt)
    prec = list(range(d)); targ = list(range(d, d + nt))
    qc.h(prec)
    for k in range(d):
        P = U  # repeated squaring keeps unitarity at large powers
        for _ in range(d - 1 - k):
            P = P @ P
        qc.append(UnitaryGate(P).control(1), [prec[k]] + targ)
    qc.append(QFTGate(d).inverse(), prec[::-1] if fixed else prec)
    return qc


def sim_counts(d, nt, U, tis, two, fixed):
    n = nt + (2 * d if two else d)
    q = QuantumRegister(n); qc = QuantumCircuit(q)
    r = np.random.default_rng(tis)
    for i in range(nt):
        qc.ry(r.uniform(0, np.pi), q[i]); qc.rz(r.uniform(0, 2 * np.pi), q[i])
    qc.append(stage(d, U, fixed), list(range(nt, nt + d)) + list(range(nt)))
    if two:
        qc.append(stage(d, U, fixed), list(range(nt + d, nt + 2 * d)) + list(range(nt)))
    p = Statevector(qc).probabilities_dict()
    nm = 2 * d if two else d
    pos = [(n - 1) - (nt + c) for c in range(nm - 1, -1, -1)]  # key = c[nm-1]..c[0]
    out = {}
    for k, v in p.items():
        kk = "".join(k[i] for i in pos); out[kk] = out.get(kk, 0) + v
    return out


def tvd(a, b):
    ta, tb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / ta - b.get(k, 0) / tb) for k in set(a) | set(b))


def single_dist(counts, d):
    D = {}; t = sum(counts.values())
    for k, c in counts.items():
        D[int(k, 2)] = D.get(int(k, 2), 0) + c / t
    return D


def pairs(counts, d):
    """direct decode, two-stage; key = c[2d-1..0]: key[d:]=stage1, key[:d]=stage2"""
    M = np.zeros((2 ** d, 2 ** d)); t = sum(counts.values())
    for k, c in counts.items():
        M[int(k[d:], 2), int(k[:d], 2)] += c / t
    return M  # [m1, m2]


def acc(M, maxd=None):
    D = M.shape[0]; i = np.arange(D)
    dl = np.minimum(abs(i[:, None] - i[None, :]), D - abs(i[:, None] - i[None, :]))
    ks = np.arange(D // 2 + 1)
    return ks, np.array([M[dl <= k].sum() for k in ks])


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(name, round(p.stat().st_size / 1e3), "kB")


def kat(label, d, nt, r, two):
    U = U_of(r); tis = int(r["target_init_seed"]); cn = r["counts"]
    sb = sim_counts(d, nt, U, tis, two, False); sf = sim_counts(d, nt, U, tis, two, True)
    rv = (lambda k: k[:d][::-1] + k[d:][::-1]) if two else (lambda k: k[::-1])
    def T(s, f):
        b = {}
        for k, c in cn.items(): b[f(k)] = b.get(f(k), 0) + c
        return round(tvd(s, b), 3)
    META.setdefault("kat", {})[label] = dict(prefix_circuit_direct=T(sb, lambda k: k), prefix_circuit_reversed=T(sb, rv),
                                              fixed_circuit_direct=T(sf, lambda k: k), fixed_circuit_reversed=T(sf, rv))
    return sb


# =============================== IonQ ===============================
def fig_2stage_nt1_d4():
    ra = J(find("noise_mitigation/job_results_2stage", "908c0d58")); rd = J(find("noise_mitigation/job_results_2stage", "9ddd87d6"))
    d = 4; sim = kat("2stage_d4_nt1_raw", d, 1, ra, True); kat("2stage_d4_nt1_debias", d, 1, rd, True)
    Ms = pairs({k: v for k, v in sim.items()}, d); Mr = pairs(ra["counts"], d); Md = pairs(rd["counts"], d)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
    for lab, M, c in (("noiseless sim (submitted circuit)", Ms, C["sim"]), ("hardware, no mitigation", Mr, C["raw"]), ("hardware, debias", Md, C["deb"])):
        k, a = acc(M); ax[0].plot(k, a, "o-", ms=3, color=c, label=lab)
    k = np.arange(9); ax[0].plot(k, np.minimum((2 * k + 1) / 16, 1), "--", color=C["uni"], label="uniform guess (2Δ+1)/2$^d$")
    ax[0].set(xlabel="Δ (circular distance)", ylabel="P(dist ≤ Δ)", title="acceptance, d=4, n_targ=1"); ax[0].legend(fontsize=6.5, loc="lower right")
    for a_, M, t in ((ax[1], Ms, "noiseless sim"), (ax[2], Mr, "hardware raw"), (ax[3], Md, "hardware debias")):
        im = a_.imshow(M.T, origin="lower", cmap="viridis", vmin=0, vmax=Ms.max()); a_.set(title=t, xlabel="m1 (stage 1)", ylabel="m2 (stage 2)")
    META["2stage_d4_nt1"] = {n: [round(float(acc(M)[1][i]), 3) for i in (0, 2)] for n, M in (("sim", Ms), ("raw", Mr), ("debias", Md))}
    fig.suptitle("IonQ Forte-Enterprise-1, two-stage QPUF, 2500 shots, Haar seed 10 (pre-fix circuit; sim = same circuit, noiseless)", fontsize=8.5, y=1.03)
    save(fig, "ionq_fe1_2stage_d4_nt1_debias_vs_raw.png")


def fig_2stage_nt2_d2():
    ra = J(find("noise_mitigation/job_results_2stage", "d35ae14d")); rd = J(find("noise_mitigation/job_results_2stage", "d3dd840f"))
    d = 2; sim = kat("2stage_d2_nt2_raw", d, 2, ra, True); kat("2stage_d2_nt2_debias", d, 2, rd, True)
    Ms = pairs(sim, d); Mr = pairs(ra["counts"], d); Md = pairs(rd["counts"], d)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), gridspec_kw=dict(width_ratios=[1.1, 1, 1, 1]))
    for lab, M, c in (("noiseless sim", Ms, C["sim"]), ("hardware raw", Mr, C["raw"]), ("hardware debias", Md, C["deb"])):
        k, a = acc(M); ax[0].plot(k, a, "o-", color=c, label=lab)
    k = np.arange(3); ax[0].plot(k, np.minimum((2 * k + 1) / 4, 1), "--", color=C["uni"], label="uniform guess")
    ax[0].set(xlabel="Δ", ylabel="P(dist ≤ Δ)", title="acceptance, d=2, n_targ=2"); ax[0].legend(fontsize=7)
    for a_, M, t in ((ax[1], Ms, "noiseless sim"), (ax[2], Mr, "hardware raw"), (ax[3], Md, "hardware debias")):
        a_.imshow(M.T, origin="lower", cmap="viridis", vmin=0, vmax=Ms.max()); a_.set(title=t, xlabel="m1", ylabel="m2"); a_.set_xticks(range(4)); a_.set_yticks(range(4))
    META["2stage_d2_nt2"] = {n: [round(float(acc(M)[1][i]), 3) for i in (0, 1)] for n, M in (("sim", Ms), ("raw", Mr), ("debias", Md))}
    fig.suptitle("IonQ Forte-Enterprise-1, two-stage QPUF, 3000 shots, Haar seed 10 (pre-fix circuit)", fontsize=8.5, y=1.04)
    save(fig, "ionq_fe1_2stage_d2_nt2_debias_vs_raw.png")


def fig_reserve_d3():
    r = J(find("braket_direct_reserve/job_results", "42273344")); d = 3
    sim = kat("reserve_42273344", d, 2, r, True); Ms = pairs(sim, d); Mh = pairs(r["counts"], d)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    for a_, M, t in ((ax[0], Ms, "noiseless sim (submitted circuit)"), (ax[1], Mh, "hardware, 100 shots")):
        a_.imshow(M.T, origin="lower", cmap="viridis", vmin=0, vmax=Ms.max()); a_.set(title=t, xlabel="m1", ylabel="m2")
    for lab, M, c in (("noiseless sim", Ms, C["sim"]), ("hardware (100 shots)", Mh, C["raw"])):
        k, a = acc(M); ax[2].plot(k, a, "o-", color=c, label=lab)
    k = np.arange(5); ax[2].plot(k, np.minimum((2 * k + 1) / 8, 1), "--", color=C["uni"], label="uniform guess")
    ax[2].set(xlabel="Δ", ylabel="P(dist ≤ Δ)", title="acceptance"); ax[2].legend(fontsize=7)
    META["reserve_d3_nt2"] = {n: [round(float(acc(M)[1][i]), 3) for i in (0, 2)] for n, M in (("sim", Ms), ("hw", Mh))}
    fig.suptitle("IonQ Forte-Enterprise-1 via Braket Direct, two-stage QPUF, d=3, n_targ=2, Haar seed 100, task 42273344", fontsize=8.5, y=1.04)
    save(fig, "ionq_fe1_braketdirect_d3_nt2_hw_vs_sim.png")


def top_bars(ax, sim, runs, d, topk=12, title=""):
    keys = sorted(sim, key=sim.get, reverse=True)[:topk]
    x = np.arange(len(keys)); n = len(runs) + 1; w = 0.8 / n
    ax.bar(x - 0.4 + w / 2, [sim[k] for k in keys], w, color=C["sim"], label="noiseless sim")
    for i, (lab, cn, col) in enumerate(runs, 1):
        t = sum(cn.values()); ax.bar(x - 0.4 + w * (i + .5), [cn.get(k, 0) / t for k in keys], w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([str(int(k, 2)) for k in keys], fontsize=7)
    ax.set(xlabel="QPE bin m = int(key) (top bins of the sim)", ylabel="probability", title=title); ax.legend(fontsize=6.5)


def fig_single_d6_d10():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a_, d, ids in ((ax[0], 6, ("63c7d73c", "27989b74")), (ax[1], 10, ("71708b38", "d5fbae12"))):
        ra, rd = J(find("noise_mitigation/job_results", ids[0])), J(find("noise_mitigation/job_results", ids[1]))
        sim = kat(f"single_d{d}_raw", d, 1, ra, False); kat(f"single_d{d}_debias", d, 1, rd, False)
        top_bars(a_, sim, [("hardware raw", ra["counts"], C["raw"]), ("hardware debias", rd["counts"], C["deb"])], d,
                 title=f"d={d}, n_targ=1, {ra['n_shots']} shots (raw) / {rd['n_shots']} (debias)")
        b = max(sim, key=sim.get)
        META[f"single_d{d}"] = dict(best_bin=int(b, 2), sim=round(sim[b], 3), raw=round(ra["counts"].get(b, 0) / sum(ra["counts"].values()), 3), debias=round(rd["counts"].get(b, 0) / sum(rd["counts"].values()), 3))
    fig.suptitle("IonQ Forte-Enterprise-1, single-stage QPE, Haar seed 10 (pre-fix circuit; sim = same circuit)", fontsize=8.5, y=1.02)
    save(fig, "ionq_fe1_qpe1_d6_d10_haar_debias_vs_raw.png")


def fig_single_nt2():
    ids = ["c6b89cad", "e18991cc", "f7fc5075"]; R = [J(find("noise_mitigation/job_results", i)) for i in ids]
    sim = kat("single_d4_nt2_raw", 4, 2, R[0], False); kat("single_d4_nt2_debias_a", 4, 2, R[1], False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    top_bars(ax, sim, [("hardware raw", R[0]["counts"], C["raw"]), ("hardware debias (run 1)", R[1]["counts"], C["deb"]), ("hardware debias (run 2)", R[2]["counts"], "#7bc58c")], 4,
             topk=12, title="IonQ FE-1 single-stage QPE, d=4, n_targ=2, 3000 shots each, seed 10 (pre-fix circuit)")
    b = max(sim, key=sim.get)
    META["single_d4_nt2"] = dict(best_bin=int(b, 2), sim=round(sim[b], 3), raw=round(R[0]["counts"].get(b, 0) / 3000, 3), deb1=round(R[1]["counts"].get(b, 0) / 3000, 3), deb2=round(R[2]["counts"].get(b, 0) / 3000, 3))
    save(fig, "ionq_fe1_qpe1_d4_nt2_haar_debias_replicates.png")


# =============================== Forte-1 (garnet dir) ===============================
def garnet_U(r):
    a = r["angles"]
    Rz = lambda t: np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])
    Ry = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
    return Rz(a["phi"]) @ Ry(a["theta"]) @ Rz(a["lam"])


def ideal_delta(U, tis, d):
    """closed form (see build_ledger.ideal_two_stage); correct-QPE ideal, MSB-first kernel."""
    D = 2 ** d; r = np.random.default_rng(tis); th, ph = r.uniform(0, np.pi), r.uniform(0, 2 * np.pi)
    psi = np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])
    T, Z = sla.schur(U, output="complex"); phi = (np.angle(np.diag(T)) / (2 * np.pi)) % 1; w = np.abs(Z.conj().T @ psi) ** 2
    diff = np.zeros(D)
    for pj, wj in zip(phi, w):
        p = np.abs(np.fft.fft(np.exp(2j * np.pi * np.arange(D) * pj)) / D) ** 2
        f = np.fft.fft(p); diff += wj * np.fft.ifft(np.conj(f) * f).real
    cum = np.zeros(D // 2 + 1); dl = np.zeros(D // 2 + 1)
    for k in range(D): dl[min(k, D - k)] += diff[k]
    return np.cumsum(dl)


def garnet_acc(r, d):
    cn = r["counts"]; t = sum(cn.values()); D = 2 ** d; dl = np.zeros(D // 2 + 1)
    for k, c in cn.items():
        a, b = int(k[d:], 2), int(k[:d], 2); x = abs(a - b); dl[min(x, D - x)] += c / t
    return np.cumsum(dl)


def fig_garnet():
    runs = {}
    for f in glob.glob(str(ROOT / "garnet-approxQPEQPUF-1target/job_results/*.json")):
        r = J(f); runs.setdefault((int(r["n_prec"]), int(r["seed"])), []).append(r)
    for k in runs: runs[k].sort(key=lambda r: -int(r["n_shots"]))
    # sweep over d, seed 100 (largest-shot run for each d)
    ds = sorted(d for d, s in runs if s == 100)
    ref = garnet_U(runs[(4, 100)][0])
    # KAT: reproduce Director numbers
    a4 = garnet_acc(runs[(4, 100)][0], 4); META["garnet_kat"] = dict(d4_p0=round(a4[0], 3), d4_ple2=round(a4[2], 3))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    H, I, N, S = [], [], [], []
    for d in ds:
        r = runs[(d, 100)][0]; U = garnet_U(r); h = garnet_acc(r, d); i = ideal_delta(U, 99, d)
        H.append(h[min(2, len(h) - 1)]); I.append(i[min(2, len(i) - 1)]); S.append(int(r["n_shots"]))
    u = [min(5 / 2 ** d, 1) for d in ds]
    ax[0].plot(ds, I, "s--", color=C["ideal"], label="noiseless ideal"); ax[0].plot(ds, H, "o-", color=C["raw"], label="Forte-1 hardware"); ax[0].plot(ds, u, ":", color=C["uni"], label="uniform guess 5/2$^d$")
    for d, h, s in zip(ds, H, S): ax[0].annotate(str(s), (d, h), fontsize=6, textcoords="offset points", xytext=(0, 5), ha="center")
    ax[0].set(xlabel="n_prec d", ylabel="P(Δ ≤ 2)", title="Haar seed 100, Δ≤2 vs d (labels: shots)"); ax[0].legend(fontsize=7)
    META["garnet_sweep"] = {int(d): dict(hw=round(float(h), 3), ideal=round(float(i), 3), shots=s) for d, h, i, s in zip(ds, H, I, S)}
    cols = {100: C["raw"], 42: C["sim"], 33: C["deb"]}
    for s in (100, 42, 33):
        r = runs[(7, s)][0]; U = garnet_U(r); h = garnet_acc(r, 7); i = ideal_delta(U, 99, 7); k = np.arange(len(h))[:20]
        ax[1].plot(k, h[:20], "o-", ms=3, color=cols[s], label=f"hw seed {s} ({r['n_shots']} shots)"); ax[1].plot(k, i[:20], "--", color=cols[s], alpha=.7)
        META.setdefault("garnet_d7", {})[s] = dict(hw=[round(float(h[j]), 3) for j in (0, 2)], ideal=[round(float(i[j]), 3) for j in (0, 2)])
    ax[1].plot(k, np.minimum((2 * k + 1) / 128, 1), ":", color=C["uni"], label="uniform guess")
    ax[1].set(xlabel="Δ", ylabel="P(dist ≤ Δ)", title="d=7, three Haar seeds (dashed = noiseless ideal)"); ax[1].legend(fontsize=6.5)
    fig.suptitle("IonQ Forte-1 (garnet directory), two-stage QPUF, n_targ=1; target init seed 99 (hard-coded in submit_job.py)", fontsize=8.5, y=1.02)
    save(fig, "ionq_forte1_dsweep_and_d7_seeds_vs_ideal.png")


# =============================== Rigetti ===============================
def rig(sel):
    out = []
    for f in glob.glob(str(ROOT / "braket_direct/rigetti_cepheus_20260807/job_results*/*.json")):
        if "analysis" in f: continue
        r = J(f)
        if r.get("role") == "qpe" and sel(r): out.append(r)
    return out


def pc(r, ideal):
    return r["counts"].get(format(int(ideal), f"0{r['n_prec']}b"), 0) / sum(r["counts"].values())


def fig_rig_zne():
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); meta = {}
    cols = {3: "#1f77b4", 4: "#d6572b", 5: "#2a9d4a"}
    for d in (3, 4, 5):
        rs = rig(lambda r: r["state_mode"] == "known" and r["n_prec"] == d and r["n_targ"] == 1 and r.get("error_mitigation") == "both" and r["n_shots"] == 10000)
        sets = {}
        for r in rs: sets.setdefault(r.get("batch_id") or "none:" + str(r["zne_scale"]) + r["job_id"][-4:], {})[r["zne_scale"]] = r
        # group scale triples: batch_id-less runs are paired by order of completion
        full = [v for v in sets.values() if len(v) == 3]
        if not full:
            loose = sorted([r for r in rs if not r.get("batch_id")], key=lambda r: r["datetime"])
            for sc in (1, 3, 5): pass
            by = {sc: [r for r in loose if r["zne_scale"] == sc] for sc in (1, 3, 5)}
            full = [{sc: by[sc][i] for sc in (1, 3, 5)} for i in range(min(len(v) for v in by.values()))] if loose else []
        for j, v in enumerate(full):
            ys = [pc(v[s], v[s]["ideal_bin"]) for s in (1, 3, 5)]; sl, ic = np.polyfit([1, 3, 5], ys, 1)
            ax.plot([1, 3, 5], ys, "o-", color=cols[d], alpha=.9 if j == 0 else .35, label=f"d={d}" if j == 0 else None)
            ax.plot([0, 1], [ic, ys[0] - 0], ":", color=cols[d], alpha=.5)
            meta.setdefault(d, []).append(dict(P=[round(y, 4) for y in ys], extrap=round(float(ic), 4)))
    ax.axhline(1, color=C["ideal"], lw=.8, ls="--"); ax.text(5.1, 1.0, "ideal = 1", fontsize=7, va="bottom", ha="right")
    ax.set(xlabel="noise scale λ (global folding)", ylabel="P(correct bin), raw", title="Rigetti Cepheus-1-108Q, QPE known φ=1/8, n_targ=1, 10 000 shots\ndotted: linear extrapolation to λ=0", xlim=(-.2, 5.3), ylim=(-.02, 1.05))
    ax.set_xticks([0, 1, 3, 5]); ax.legend(fontsize=7)
    META["rig_zne"] = meta
    save(fig, "rigetti_cepheus_qpe_known_zne_lambda_scan_d3-5.png")


def fig_rig_dist():
    fig, ax = plt.subplots(2, 3, figsize=(11, 5), sharey="row"); meta = {}
    for row, mode in enumerate(("known", "haar")):
        for col, d in enumerate((3, 4, 5)):
            rs = [r for r in rig(lambda r: r["state_mode"] == mode and r["n_prec"] == d and r["n_targ"] == 1 and r["zne_scale"] == 1 and r.get("error_mitigation") == "both" and r["n_shots"] == 10000)]
            r = sorted(rs, key=lambda r: r["datetime"])[0]; D = 2 ** d; t = sum(r["counts"].values())
            hw = np.zeros(D); idl = np.zeros(D)
            for k, c in r["counts"].items(): hw[int(k, 2)] += c / t
            for k, p in r["ideal_distribution"].items(): idl[int(float(k))] += p
            a = ax[row, col]; a.bar(np.arange(D) - .2, idl, .4, color=C["sim"], label="ideal (noiseless)"); a.bar(np.arange(D) + .2, hw, .4, color=C["raw"], label="hardware, raw λ=1")
            a.set(title=f"{'known φ=1/8' if mode=='known' else 'Haar seed 10'}, d={d}", xlabel="QPE bin m"); a.set_xticks(range(0, D, max(1, D // 8)))
            if col == 0: a.set_ylabel("probability")
            meta[f"{mode}_d{d}"] = dict(uuid=r["job_id"].split("/")[-1], ideal_top=int(idl.argmax()), p_ideal_bin=round(float(hw[idl.argmax()]), 3), ideal_p=round(float(idl.max()), 3), tvd=round(float(0.5 * abs(hw - idl).sum()), 3))
    ax[0, 0].legend(fontsize=7)
    META["rig_dist"] = meta
    fig.suptitle("Rigetti Cepheus-1-108Q, single-stage QPE, n_targ=1, 10 000 shots (direct decode: char i = c[i])", fontsize=9, y=1.0)
    fig.tight_layout(); save(fig, "rigetti_cepheus_qpe_known_and_haar_hw_vs_ideal_d3-5.png")


if __name__ == "__main__":
    for f in (fig_2stage_nt1_d4, fig_2stage_nt2_d2, fig_reserve_d3, fig_single_d6_d10, fig_single_nt2, fig_garnet, fig_rig_zne, fig_rig_dist):
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", f.__name__)
    json.dump(META, open(ROOT / "docs/plot_ledger/notes_figs_meta.json", "w"), indent=1, default=str)
