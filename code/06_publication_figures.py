"""Publication-quality figures for the DPP contracting game.

This script is self-contained: it reads the recorded simulation outcomes in
``code/data/simulation_results.csv`` and re-renders every plotted figure with a
consistent, journal-ready style. It does not recompute the equilibria, so the
figures are guaranteed to match the numbers reported in the paper tables.

Run:  python 06_publication_figures.py
Output: paper/figures/*.pdf  and  dpp_contracting_game_overleaf_v2/figures/*.pdf
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

cache = Path(tempfile.gettempdir()) / "dpp_mpl"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))
os.environ.setdefault("XDG_CACHE_HOME", str(cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Global style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Outcome palette (None / Low / High) -- colour-blind distinguishable.
OUTCOME_COLORS = ["#9aa0a6", "#e0a236", "#2f8f6b"]
OUTCOME_LABELS = ["No disclosure", "Low quality", "High quality"]
OUTCOME_CODE = {"none": 0, "low": 1, "high": 2}

# Contract palette (Basic / Incentive / Audit / Hybrid).
CONTRACT_COLORS = ["#9aa0a6", "#e0a236", "#3e6da6", "#2f8f6b"]
CONTRACT_LABELS = ["Basic compliance", "Incentive-only", "Audit/penalty", "Hybrid"]
CONTRACT_ORDER = ["basic_compliance", "incentive_only", "audit_penalty", "hybrid_support"]
CONTRACT_CODE = {name: i for i, name in enumerate(CONTRACT_ORDER)}

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "code" / "data" / "simulation_results.csv"
OUT_DIRS = [
    ROOT / "paper" / "figures",
    ROOT / "dpp_contracting_game_overleaf_v2" / "figures",
]
# Optional override (set DPP_FIG_OUT) used during iterative rendering to avoid
# intermittent mount write-locks; final copy is done separately.
if os.environ.get("DPP_FIG_OUT"):
    OUT_DIRS = [Path(os.environ["DPP_FIG_OUT"])]


def save(fig, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def grid(df: pd.DataFrame, x: str, y: str, value: str):
    """Return (X edges, Y edges, Z) for pcolormesh with true coordinates."""
    piv = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    piv = piv.sort_index()
    xs = piv.columns.to_numpy(dtype=float)
    ys = piv.index.to_numpy(dtype=float)
    Z = piv.to_numpy(dtype=float)

    def edges(a):
        a = np.asarray(a, dtype=float)
        mid = (a[:-1] + a[1:]) / 2
        return np.concatenate([[a[0] - (mid[0] - a[0])], mid, [a[-1] + (a[-1] - mid[-1])]])

    return edges(xs), edges(ys), Z


def outcome_cmap():
    cmap = ListedColormap(OUTCOME_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    return cmap, norm


def contract_cmap():
    cmap = ListedColormap(CONTRACT_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_equilibrium_regions(d: pd.DataFrame) -> None:
    data = d[d.experiment == "equilibrium_region_grid"].copy()
    data["code"] = data.supplier_choice.map(OUTCOME_CODE)
    cmap, norm = outcome_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True, sharey=True)
    titles = dict(zip(CONTRACT_ORDER, CONTRACT_LABELS))
    for ax, contract in zip(axes.flat, CONTRACT_ORDER):
        sub = data[data.contract == contract]
        X, Y, Z = grid(sub, "q", "r", "code")
        ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading="auto", rasterized=True)
        ax.set_title(titles[contract], fontsize=11)
        ax.set_xlim(X[0], X[-1])
        ax.set_ylim(Y[0], Y[-1])
        ax.tick_params(length=3)
    fig.supxlabel("Supplier capability $q$", fontsize=11.5, y=0.10)
    fig.supylabel("Commercial / IP risk $r$", fontsize=11.5, x=0.02)
    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(OUTCOME_COLORS, OUTCOME_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0.04, 0.13, 1, 1])
    save(fig, "fig_equilibrium_regions")


def fig_outcome_heatmap(d: pd.DataFrame, exp, x, y, xlabel, ylabel, name,
                        threshold=None):
    sub = d[d.experiment == exp].copy()
    sub["code"] = sub.supplier_choice.map(OUTCOME_CODE)
    cmap, norm = outcome_cmap()
    X, Y, Z = grid(sub, x, y, "code")
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(Y[0], Y[-1])
    ax.tick_params(length=3)
    if threshold is not None:
        pf, lo, hi = threshold
        pp = np.linspace(max(X[0], 1e-3), X[-1], 200)
        ff = pf / pp
        m = (ff >= lo) & (ff <= hi)
        ax.plot(pp[m], ff[m], color="white", lw=1.8, ls="--")
        ax.plot(pp[m], ff[m], color="#222222", lw=0.9, ls="--")
        ax.text(0.62, pf / 0.62 + 0.4, r"$pF=%.2f$" % pf, color="#111111",
                fontsize=9.5, rotation=-26, ha="center", va="bottom")
    present = sorted(sub["code"].unique())
    handles = [Patch(facecolor=OUTCOME_COLORS[i], edgecolor="none",
                     label=OUTCOME_LABELS[i]) for i in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=len(present), frameon=False, fontsize=9.5,
              handlelength=1.1, columnspacing=1.2)
    fig.tight_layout()
    save(fig, name)


def fig_contract_choice(d: pd.DataFrame, exp, x, y, xlabel, ylabel, name):
    sub = d[d.experiment == exp].copy()
    sub["code"] = sub.optimal_contract.map(CONTRACT_CODE)
    cmap, norm = contract_cmap()
    X, Y, Z = grid(sub, x, y, "code")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(Y[0], Y[-1])
    ax.tick_params(length=3)
    present = sorted(int(v) for v in np.unique(Z[~np.isnan(Z)]))
    handles = [Patch(facecolor=CONTRACT_COLORS[i], edgecolor="none",
                     label=CONTRACT_LABELS[i]) for i in present]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, title="Buyer-optimal\ncontract", alignment="left")
    fig.tight_layout()
    save(fig, name)


def fig_contract_comparison(d: pd.DataFrame) -> None:
    base = d[d.experiment == "base_grid"].copy()
    shares = (base.groupby(["contract", "supplier_choice"]).size()
              / base.groupby("contract").size()).unstack(fill_value=0)
    for col in ["none", "low", "high"]:
        if col not in shares:
            shares[col] = 0.0
    shares = shares.reindex(CONTRACT_ORDER)[["none", "low", "high"]]
    payoff = base.groupby("contract")["buyer_payoff"].mean().reindex(CONTRACT_ORDER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.7),
                                   gridspec_kw={"width_ratios": [1.15, 0.95]})
    x = np.arange(len(CONTRACT_ORDER))
    bottom = np.zeros(len(x))
    for j, col in enumerate(["none", "low", "high"]):
        ax1.bar(x, shares[col].values, bottom=bottom, width=0.62,
                color=OUTCOME_COLORS[j], label=OUTCOME_LABELS[j],
                edgecolor="white", linewidth=0.6)
        bottom += shares[col].values
    ax1.set_xticks(x)
    ax1.set_xticklabels(CONTRACT_LABELS, rotation=20, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Share of supplier responses")
    ax1.set_title("(a) Supplier responses", fontsize=11)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3,
               frameon=False, fontsize=9.5, handlelength=1.1, columnspacing=1.1)

    bars = ax2.bar(x, payoff.values, width=0.62,
                   color=[CONTRACT_COLORS[CONTRACT_CODE[c]] for c in CONTRACT_ORDER],
                   edgecolor="white", linewidth=0.6)
    ax2.axhline(0, color="#333333", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(CONTRACT_LABELS, rotation=20, ha="right")
    ax2.set_ylabel("Mean buyer payoff")
    ax2.set_title("(b) Buyer payoff", fontsize=11)
    for b, v in zip(bars, payoff.values):
        ax2.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom" if v >= 0 else "top",
                     fontsize=9, xytext=(0, 3 if v >= 0 else -3),
                     textcoords="offset points")
    pad = 0.15 * (payoff.max() - payoff.min())
    ax2.set_ylim(payoff.min() - pad - 1, payoff.max() + pad + 1)
    fig.tight_layout()
    save(fig, "fig_contract_comparison")


def main() -> None:
    d = pd.read_csv(DATA)
    print("Rendering figures...")
    fig_equilibrium_regions(d)
    # Deterrence threshold for the audit/penalty panel (c=4, q=0.8, r=0.6,
    # B-B_L=0.30, g=0.8): pF* = (C_H-C_L)+(r_H-r_L)-(B-B_L)-g.
    c, q, lam, r, phi, rls, g = 4.0, 0.8, 0.45, 0.6, 0.15, 0.35, 0.8
    CH = max(c / q, 0.25)
    pf_star = (CH - lam * CH) + (r * (1 - phi) - rls * r * (1 - phi)) - 0.30 - g
    fig_outcome_heatmap(d, "audit_penalty_heatmap", "p", "F",
                        "Audit probability $p$", "Penalty $F$",
                        "fig_audit_penalty_heatmap",
                        threshold=(pf_star, 0.0, 10.0))
    fig_outcome_heatmap(d, "capability_support_heatmap", "q", "S",
                        "Supplier capability $q$", "Buyer support $S$",
                        "fig_capability_support_heatmap")
    fig_contract_choice(d, "buyer_choice_region", "q", "R",
                        "Supplier capability $q$", "Buyer circularity value $R$",
                        "fig_buyer_choice_region")
    fig_contract_choice(d, "governance_cost_sensitivity", "k", "m",
                        "Support cost $k$", "Confidentiality cost $m$",
                        "fig_governance_cost_sensitivity")
    fig_contract_comparison(d)
    print(f"pF* overlay = {pf_star:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
