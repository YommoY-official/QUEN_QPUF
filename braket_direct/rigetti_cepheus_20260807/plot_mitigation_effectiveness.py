"""
Plot mitigation effectiveness (REM / ZNE / ZNE+REM) against P_correct = P(measured
bitstring == ideal_bin), the known-theoretical-value observable computed by exact
simulation of each QPE circuit (see job_results_qpe/analysis_summary_*.json ->
"run.ideal_bin" and "table"). Source aggregate: plots/mitigation_benefit_table.csv,
itself built from all 37 real-hardware run families (2026-08-07 reservation).

"valid" here means the point/extrapolation landed inside the physical range [0,1]
(the "table"[i]["valid"] flag from mitig_analysis.py). This is necessary but not
sufficient for trusting a ZNE extrapolation -- MITIGATION_BRIEF.md section 7
("Revision, 2026-08-28") applies a stricter signal + monotonicity + model-agreement
gate that narrows genuinely trustworthy ZNE families to 6 of 26; that stricter
result is not re-derived here and is only referenced in the caption.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("plots/mitigation_benefit_table.csv")
df = df.sort_values(["n_prec", "p_raw"], ascending=[True, False]).reset_index(drop=True)
df["label"] = df.apply(
    lambda r: f"n_prec={r.n_prec} n_targ={r.n_targ} {r.state_mode}"
    + (f" phi={r.phi:g}" if pd.notna(r.phi) else ""),
    axis=1,
)

techniques = [
    ("rem", "REM (simplex)"),
    ("zne", "ZNE (linear)"),
    ("zne_rem", "ZNE+REM (linear)"),
]

n = len(df)
y = np.arange(n)[::-1]  # top = first row

fig, axes = plt.subplots(1, 3, figsize=(16.5, 0.32 * n + 1.8), sharey=True)
fig.subplots_adjust(wspace=0.08, top=0.90)

for ax, (key, title) in zip(axes, techniques):
    avail = df[f"{key}_available"].fillna(False).astype(bool)
    valid = df[f"delta_{key}_valid"].fillna(False).astype(bool)
    p_mit = df[f"p_{key}"]

    # raw markers, always present
    ax.scatter(df["p_raw"], y, marker="x", color="0.4", s=25, zorder=3, label="raw")

    for i in range(n):
        if not avail[i] or pd.isna(p_mit[i]):
            continue
        helped = p_mit[i] > df["p_raw"][i]
        color = "#1a9850" if helped else "#d73027"
        ax.plot([df["p_raw"][i], p_mit[i]], [y[i], y[i]], color=color, lw=1.2, zorder=2, alpha=0.8)
        ax.scatter(
            [p_mit[i]], [y[i]],
            marker="o",
            facecolor=color if valid[i] else "none",
            edgecolor=color,
            s=38,
            zorder=4,
            linewidths=1.4,
        )

    n_avail = int(avail.sum())
    n_valid = int((avail & valid).sum())
    d = (p_mit[avail & valid] - df["p_raw"][avail & valid])
    win_rate = (d > 0).mean() * 100 if len(d) else float("nan")
    mean_d = d.mean() if len(d) else float("nan")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(
        0.5, 1.045,
        f"{n_avail}/{n} attempted, {n_valid} physical\n"
        f"{win_rate:.0f}% improved | mean $\\Delta$={mean_d:+.4f}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3,
    )
    ax.set_xscale("symlog", linthresh=2e-3)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"$P_{correct}$ (symlog)")
    ax.grid(axis="x", color="0.92", lw=0.7)

axes[0].set_yticks(y)
axes[0].set_yticklabels(df["label"], fontsize=7.5)
axes[0].set_ylim(-1, n)

legend_elems = [
    plt.Line2D([0], [0], marker="x", color="0.4", lw=0, label="raw (lambda=1)"),
    plt.Line2D([0], [0], marker="o", color="#1a9850", lw=0, label="mitigated, improved P_correct"),
    plt.Line2D([0], [0], marker="o", color="#d73027", lw=0, label="mitigated, worsened P_correct"),
    plt.Line2D([0], [0], marker="o", color="0.4", markerfacecolor="none", lw=0,
               label="hollow = extrapolation left [0,1] (unphysical)"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Cepheus-1-108Q QPE: mitigation effect on $P_{correct}$ vs known ideal_bin, all 37 hardware families",
    fontsize=12, y=1.03,
)
fig.savefig("plots/mitigation_effectiveness_known_answer.png", dpi=160, bbox_inches="tight")
print("wrote plots/mitigation_effectiveness_known_answer.png")

# ---- console summary ----
print()
for key, title in techniques:
    avail = df[f"{key}_available"].fillna(False).astype(bool)
    valid = df[f"delta_{key}_valid"].fillna(False).astype(bool)
    d_all = df[f"delta_{key}"][avail]
    d_valid = df[f"delta_{key}"][avail & valid]
    print(f"{title}: attempted {avail.sum()}/{n}, physical {int((avail&valid).sum())}, "
          f"mean delta (physical only) = {d_valid.mean():+.5f}, "
          f"win rate (physical only) = {(d_valid>0).mean()*100:.0f}%")
