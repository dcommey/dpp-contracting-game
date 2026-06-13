"""Generate paper tables and figures for the DPP contracting game."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

cache_root = Path(tempfile.gettempdir())
mpl_config = cache_root / "dpp_mplconfig"
xdg_cache = cache_root / "dpp_xdg_cache"
mpl_config.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "code" / "data"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"

OUTCOME_CODE = {"none": 0, "low": 1, "high": 2}
OUTCOME_LABELS = {0: "None", 1: "Low", 2: "High"}
OUTCOME_COLORS = ["#8f8f8f", "#d8a24a", "#3b8f6b"]
CONTRACT_ORDER = ["basic_compliance", "incentive_only", "audit_penalty", "hybrid_support"]
CONTRACT_CODE = {name: index for index, name in enumerate(CONTRACT_ORDER)}
CONTRACT_COLORS = ["#8f8f8f", "#d8a24a", "#4f7aa8", "#3b8f6b"]

PARAMETER_LABELS = {
    "c": "$c$",
    "cost_form": "Cost specification",
    "cost_floor": "$\\underline{C}$",
    "gamma": "$\\gamma$",
    "q": "$q$",
    "r": "$r$",
    "p": "$p$",
    "F": "$F$",
    "F_bar": "$\\bar{F}$",
    "B": "$B$",
    "S": "$S$",
    "S_bar": "$\\bar{S}$",
    "R": "$R$",
    "L": "$L$",
    "v": "$v$",
    "g": "$g$",
    "k": "$k$",
    "m": "$m$",
    "alpha": "$\\alpha$",
    "lam": "$\\lambda$",
    "beta": "$\\beta$",
    "eta_penalty": "$\\eta$",
    "r_low_share": "$\\rho$",
    "confidentiality": "$\\phi$",
    "B_L": "$B_L$",
    "contract_value": "$V$",
    "outside_option": "$O$",
}

MODEL_LINK_LABELS = {
    "disclosure": "Disclosure duty",
    "C_H;C_L": "$C_H; C_L$",
    "r": "$r$",
    "C_H": "$C_H$",
    "R": "$R$",
    "p;v": "$p; v$",
    "p;v;g": "$p; v; g$",
    "F": "$F$",
    "B": "$B$",
    "C_H;R": "$C_H; R$",
    "S;alpha;q": "$S; \\alpha; q$",
    "R;L": "$R; L$",
    "outside option; B": "$O; B$",
}


def load_module(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, CODE_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def escape_latex(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\", "\u0000BACKSLASH\u0000")
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("\u0000BACKSLASH\u0000", "\\textbackslash{}")


def latex_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    caption: str,
    label: str,
    widths: Sequence[float] | None = None,
    raw_columns: set[int] | None = None,
    wide: bool = False,
) -> str:
    widths = widths or [1 / len(headers)] * len(headers)
    raw_columns = raw_columns or set()
    column_spec = "@{}" + "".join(f"p{{{width:.2f}\\linewidth}}" for width in widths) + "@{}"
    table_env = "table*" if wide else "table"
    lines = [
        f"\\begin{{{table_env}}}[t]",
        "\\centering",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [str(cell) if index in raw_columns else escape_latex(cell) for index, cell in enumerate(row)]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\endgroup", f"\\end{{{table_env}}}", ""])
    return "\n".join(lines)


def ensure_inputs() -> pd.DataFrame:
    model = load_module("02_model_setup.py", "model_setup")
    model.write_default_files()
    simulation = load_module("04_simulation.py", "simulation")
    simulation.main()
    equilibrium = load_module("03_equilibrium_analysis.py", "equilibrium")
    equilibrium.main()
    return pd.read_csv(DATA_DIR / "simulation_results.csv")


def write_model_parameter_table(model) -> None:
    params = model.ModelParameters()
    rows = []
    for key, description in model.PARAMETER_DESCRIPTIONS.items():
        rows.append((PARAMETER_LABELS.get(key, key), getattr(params, key), description))
    table = latex_table(
        ["Parameter", "Default", "Interpretation"],
        rows,
        "Model parameters and default values.",
        "tab:model_parameters",
        widths=[0.16, 0.14, 0.62],
        raw_columns={0},
        wide=True,
    )
    (TABLE_DIR / "table_model_parameters.tex").write_text(table, encoding="utf-8")


def write_contract_parameter_table(model) -> None:
    rows = []
    for contract in CONTRACT_ORDER:
        values = model.CONTRACTS[contract]
        rows.append(
            (
                values["label"],
                f"{values['p']:.2f}",
                f"{values['F']:.2f}",
                f"{values['B']:.2f}",
                f"{values['B_L']:.2f}",
                f"{values['S']:.2f}",
                f"{values['confidentiality']:.2f}",
            )
        )
    table = latex_table(
        ["Contract", "$p$", "$F$", "$B$", "$B_L$", "$S$", "$\\phi$"],
        rows,
        "Discrete contract vectors used in the analytical and numerical model.",
        "tab:contract_vectors",
        widths=[0.28, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        raw_columns={1, 2, 3, 4, 5, 6},
        wide=True,
    )
    (TABLE_DIR / "table_contract_vectors.tex").write_text(table, encoding="utf-8")


def write_taxonomy_table() -> None:
    taxonomy = pd.read_csv(DATA_DIR / "contract_clause_taxonomy.csv")
    rows = []
    for row in taxonomy[["category", "contract_function", "model_parameter_link", "governance_rationale"]].to_dict("records"):
        rows.append(
            [
                row["category"],
                row["contract_function"],
                MODEL_LINK_LABELS.get(row["model_parameter_link"], row["model_parameter_link"]),
                row["governance_rationale"],
            ]
        )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\tiny",
        "\\caption{DPP contract clause taxonomy.}",
        "\\label{tab:contract_clause_taxonomy}",
        "\\begin{tabular}{@{}p{0.20\\textwidth}p{0.22\\textwidth}p{0.12\\textwidth}p{0.34\\textwidth}@{}}",
        "\\toprule",
        "Clause category & Contract function & Model link & Governance rationale \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [escape_latex(row[0]), escape_latex(row[1]), str(row[2]), escape_latex(row[3])]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\endgroup", "\\end{table*}", ""])
    table = "\n".join(lines)
    (TABLE_DIR / "table_contract_clause_taxonomy.tex").write_text(table, encoding="utf-8")


def write_literature_summary_table() -> None:
    rows = [
        (
            "DPP data needs and ecosystem design",
            "\\citet{jensen2023digital}; \\citet{king2023proposed}; \\citet{lopes2024digital}",
            "DPPs require interoperable product, material, environmental, and life-cycle data across supply-chain and reverse-logistics actors.",
            "Motivates the buyer value term $R$ and the distinction between complete and partial disclosure.",
        ),
        (
            "DPP data governance",
            "\\citet{ducuing2023data}; \\citet{eu2024espr}",
            "DPP implementation depends on access rights, trust, data quality, legal compliance, and protection of sensitive information.",
            "Motivates audit probability $p$, confidentiality protection, and supplier risk $r$.",
        ),
        (
            "Circular supply-chain governance",
            "\\citet{deangelis2018circular}; \\citet{govindan2015reverse}",
            "Circularity requires coordination across forward and reverse flows and depends on information for reuse, repair, remanufacturing, and recycling.",
            "Links DPP data quality to circularity losses $L$ and residual value $\\beta R$.",
        ),
        (
            "Contract theory and relational governance",
            "\\citet{cachon2003supply}; \\citet{cachon2005supply}; \\citet{poppo2002do}; \\citet{malhotra2011trust}",
            "Contracts can coordinate supply chains, allocate duties, and complement relational mechanisms when uncertainty and opportunism are present.",
            "Supports comparison of incentive-only, enforcement-only, and hybrid governance.",
        ),
        (
            "Transaction cost economics and capability constraints",
            "\\citet{williamson2008outsourcing}; \\citet{uruena2026digital}",
            "Uncertainty, monitoring costs, asset specificity, and uneven digital capabilities shape governance choices.",
            "Motivates capability $q$, support $S$, and monitoring cost $v$.",
        ),
        (
            "Target-journal supply-chain contracts and circularity",
            "\\citet{kolagar2026material}; \\citet{doijjclepro201803046}; \\citet{doijjclepro2022135135}; \\citet{doijjclepro2021128629}",
            "Journal of Cleaner Production research has used DPP readiness, game theory, coordination contracts, and empirical circular-supply-chain studies to examine sustainable supply-chain governance.",
            "This paper extends that stream from DPP readiness, greening, and circular coordination to verifiable DPP data disclosure.",
        ),
    ]
    table = latex_table(
        ["Literature stream", "Representative sources", "Main insight", "Use in model"],
        rows,
        "Literature summary and link to the analytical model.",
        "tab:literature_summary",
        widths=[0.21, 0.22, 0.30, 0.19],
        raw_columns={1, 3},
        wide=True,
    )
    (TABLE_DIR / "table_literature_summary.tex").write_text(table, encoding="utf-8")


def write_contract_comparison_table(results: pd.DataFrame) -> None:
    base = results[results["experiment"] == "base_grid"].copy()
    grouped = (
        base.groupby(["contract", "contract_label"])
        .agg(
            high_share=("supplier_choice", lambda s: (s == "high").mean()),
            low_share=("supplier_choice", lambda s: (s == "low").mean()),
            none_share=("supplier_choice", lambda s: (s == "none").mean()),
            mean_buyer_payoff=("buyer_payoff", "mean"),
            mean_supplier_payoff=("supplier_payoff", "mean"),
        )
        .reset_index()
    )
    rows = []
    grouped["contract_order"] = grouped["contract"].map(CONTRACT_CODE)
    for _, row in grouped.sort_values("contract_order").iterrows():
        rows.append(
            (
                row["contract_label"],
                f"{100 * row['high_share']:.1f}\\%",
                f"{100 * row['low_share']:.1f}\\%",
                f"{100 * row['none_share']:.1f}\\%",
                f"{row['mean_buyer_payoff']:.2f}",
                f"{row['mean_supplier_payoff']:.2f}",
            )
        )
    table = latex_table(
        ["Contract", "High", "Low", "None", "Buyer payoff", "Supplier payoff"],
        rows,
        "Contract scenario comparison across the baseline parameter grid.",
        "tab:contract_scenario_comparison",
        widths=[0.21, 0.12, 0.12, 0.12, 0.17, 0.18],
        raw_columns={1, 2, 3},
        wide=True,
    )
    (TABLE_DIR / "table_contract_scenario_comparison.tex").write_text(table, encoding="utf-8")


def write_buyer_choice_summary_table(results: pd.DataFrame) -> None:
    mc = results[results["experiment"] == "monte_carlo_buyer_choice"].copy()
    rows = []
    for cost_form, group in mc.groupby("cost_form"):
        total = len(group)
        counts = group["optimal_contract"].value_counts(normalize=True)
        rows.append(
            (
                cost_form.capitalize(),
                f"{100 * counts.get('basic_compliance', 0):.1f}\\%",
                f"{100 * counts.get('incentive_only', 0):.1f}\\%",
                f"{100 * counts.get('audit_penalty', 0):.1f}\\%",
                f"{100 * counts.get('hybrid_support', 0):.1f}\\%",
                f"{group['buyer_payoff'].mean():.2f}",
                total,
            )
        )
    table = latex_table(
        ["Cost form", "Basic", "Incentive", "Audit", "Hybrid", "Mean buyer payoff", "$n$"],
        rows,
        "Monte Carlo buyer-optimal contract robustness by disclosure-cost specification.",
        "tab:monte_carlo_buyer_choice",
        widths=[0.16, 0.12, 0.13, 0.12, 0.12, 0.19, 0.08],
        raw_columns={1, 2, 3, 4, 6},
        wide=True,
    )
    (TABLE_DIR / "table_monte_carlo_buyer_choice.tex").write_text(table, encoding="utf-8")


def write_buyer_comparison_table() -> None:
    rows = [
        (
            "Hybrid $H$ vs. alternative $N$",
            "$R+L \\geq B_H+A_H+K_H+M_H$",
            "Hybrid-induced high-quality disclosure is worthwhile only when DPP value plus avoided loss covers hybrid governance cost.",
        ),
        (
            "Hybrid $H$ vs. incentive $H$",
            "$B_H+A_H+K_H+M_H \\leq B_I+A_I+K_I+M_I$",
            "If both contracts induce high-quality disclosure, the buyer prefers the lower-cost implementation package.",
        ),
        (
            "Hybrid $H$ vs. audit $LQ$",
            "$(1-\\beta)R+L+A_A+B_{L,A}+K_A+M_A \\geq B_H+A_H+K_H+M_H+\\eta p_AF_A$",
            "Hybrid beats audit-induced partial disclosure when the added circularity value and avoided low-quality loss exceed the net extra cost.",
        ),
        (
            "Hybrid $H$ vs. basic $LQ$",
            "$(1-\\beta)R+L+A_B+B_{L,B}+K_B+M_B \\geq B_H+A_H+K_H+M_H+\\eta p_BF_B$",
            "Hybrid beats basic compliance when complete data value offsets the extra incentive, audit, support, and confidentiality costs.",
        ),
        (
            "Audit $LQ$ vs. incentive $H$",
            "$(1-\\beta)R+L+B_{L,A}+A_A+K_A+M_A-\\eta p_AF_A \\leq B_I+A_I+K_I+M_I$",
            "Audit-induced partial disclosure is buyer-preferred only when the cost of inducing high-quality disclosure is larger than the residual-value loss.",
        ),
    ]
    table = latex_table(
        ["Comparison", "Buyer-preference condition", "Interpretation"],
        rows,
        "Buyer-side contract comparison conditions.",
        "tab:buyer_contract_comparisons",
        widths=[0.22, 0.39, 0.31],
        raw_columns={0, 1},
        wide=True,
    )
    (TABLE_DIR / "table_buyer_contract_comparisons.tex").write_text(table, encoding="utf-8")


def write_targeted_robustness_table(results: pd.DataFrame) -> None:
    data = results[results["experiment"] == "targeted_robustness"].copy()
    rows = []
    for _, group in data.groupby("scenario_order", sort=True):
        optimal = group[group["is_optimal"] == True]  # noqa: E712
        best = optimal.iloc[0]
        hybrid = group[group["contract"] == "hybrid_support"].iloc[0]
        best_label = best["contract_label"]
        if len(optimal) > 1:
            best_label = f"{best_label} (tie)"
        rows.append(
            (
                best["scenario_label"],
                best["scenario_note"],
                best_label,
                str(best["supplier_choice"]).capitalize(),
                f"{float(best['buyer_payoff']):.2f}",
                f"{float(hybrid['buyer_payoff']):.2f}",
            )
        )
    table = latex_table(
        ["Robustness case", "Perturbation", "Buyer-optimal contract", "Response", "Best payoff", "Hybrid payoff"],
        rows,
        "Targeted robustness cases where the buyer need not prefer hybrid support.",
        "tab:targeted_robustness",
        widths=[0.17, 0.27, 0.16, 0.09, 0.09, 0.09],
        wide=True,
    )
    (TABLE_DIR / "table_targeted_robustness.tex").write_text(table, encoding="utf-8")


def write_managerial_implications_table() -> None:
    rows = [
        (
            "Low supplier capability",
            "Penalties can induce exit or non-disclosure when suppliers cannot cheaply generate complete data.",
            "Pair DPP obligations with templates, onboarding, shared data infrastructure, and phased capability-building.",
        ),
        (
            "High commercial/IP sensitivity",
            "Suppliers may resist complete disclosure even under strong audit rights.",
            "Use access tiers, confidentiality protections, field-level permissions, and purpose limitation.",
        ),
        (
            "Weak verification",
            "Incentives may buy participation without assuring data quality.",
            "Tie premiums or preferred-supplier status to verifiable completeness and audit trails.",
        ),
        (
            "SME-heavy supplier base",
            "Uniform enforcement shifts compliance burdens to least capable firms.",
            "Segment suppliers by readiness and use hybrid support for strategically important SME suppliers.",
        ),
        (
            "High buyer circularity value",
            "The value of trustworthy DPP data can justify support and verification costs.",
            "Evaluate DPP contracting as circular supply-chain investment rather than administrative compliance.",
        ),
    ]
    table = latex_table(
        ["Governance problem", "Model implication", "Managerial action"],
        rows,
        "Managerial implications from the DPP contracting game.",
        "tab:managerial_implications",
        widths=[0.24, 0.34, 0.34],
        wide=True,
    )
    (TABLE_DIR / "table_managerial_implications.tex").write_text(table, encoding="utf-8")


def plot_game_tree() -> None:
    graph = nx.DiGraph()
    edges = [
        ("Buyer", "Basic compliance"),
        ("Buyer", "Incentive-only"),
        ("Buyer", "Audit/penalty"),
        ("Buyer", "Hybrid"),
    ]
    for contract in ["Basic compliance", "Incentive-only", "Audit/penalty", "Hybrid"]:
        edges.extend(
            [
                (contract, f"{contract}\nHigh"),
                (contract, f"{contract}\nLow"),
                (contract, f"{contract}\nNone"),
            ]
        )
    graph.add_edges_from(edges)
    positions = {
        "Buyer": (0, 0),
        "Basic compliance": (-3, -1),
        "Incentive-only": (-1, -1),
        "Audit/penalty": (1, -1),
        "Hybrid": (3, -1),
    }
    for x, contract in zip([-3, -1, 1, 3], ["Basic compliance", "Incentive-only", "Audit/penalty", "Hybrid"]):
        positions[f"{contract}\nHigh"] = (x - 0.45, -2.2)
        positions[f"{contract}\nLow"] = (x, -2.2)
        positions[f"{contract}\nNone"] = (x + 0.45, -2.2)

    plt.figure(figsize=(10.5, 5.5))
    nx.draw_networkx_edges(graph, positions, arrows=True, arrowstyle="-|>", arrowsize=14, edge_color="#555555")
    node_colors = ["#2f5f8f" if node == "Buyer" else "#d7e6ef" if "\n" not in node else "#f1efe7" for node in graph.nodes]
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=2100, edgecolors="#333333", linewidths=0.8)
    labels = {}
    for node in graph.nodes:
        if "\n" in node:
            labels[node] = node.split("\n", maxsplit=1)[1]
        else:
            labels[node] = (
                node.replace("Basic compliance", "Basic")
                .replace("Incentive-only", "Incentive")
                .replace("Audit/penalty", "Audit")
            )
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_game_tree.pdf")
    plt.close()


def plot_outcome_heatmap(data: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    frame = data.copy()
    frame["outcome_code"] = frame["supplier_choice"].map(OUTCOME_CODE)
    pivot = frame.pivot_table(index=y, columns=x, values="outcome_code", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)
    plt.figure(figsize=(4.3, 3.3))
    ax = sns.heatmap(
        pivot,
        cmap=sns.color_palette(OUTCOME_COLORS, as_cmap=True),
        vmin=0,
        vmax=2,
        cbar_kws={"ticks": [0.33, 1.0, 1.67], "shrink": 0.82},
        xticklabels=4,
        yticklabels=4,
        linewidths=0.0,
    )
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(["None", "Low", "High"])
    colorbar.ax.tick_params(labelsize=7)
    ax.set_title(title, pad=8, fontsize=9.5)
    ax.set_xlabel(x, fontsize=9)
    ax.set_ylabel(y, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout(pad=0.5)
    plt.savefig(path)
    plt.close()


def plot_equilibrium_regions(results: pd.DataFrame) -> None:
    data = results[results["experiment"] == "equilibrium_region_grid"].copy()
    labels = data[["contract", "contract_label"]].drop_duplicates().set_index("contract")["contract_label"].to_dict()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, contract in zip(axes.flat, CONTRACT_ORDER):
        subset = data[data["contract"] == contract].copy()
        subset["outcome_code"] = subset["supplier_choice"].map(OUTCOME_CODE)
        pivot = subset.pivot_table(index="r", columns="q", values="outcome_code", aggfunc="mean").sort_index(ascending=False)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=sns.color_palette(OUTCOME_COLORS, as_cmap=True),
            vmin=0,
            vmax=2,
            cbar=False,
            xticklabels=5,
            yticklabels=5,
        )
        ax.set_title(labels[contract])
        ax.set_xlabel("Capability q")
        ax.set_ylabel("IP/commercial risk r")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in OUTCOME_COLORS]
    fig.legend(handles, ["None", "Low", "High"], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Equilibrium Disclosure Regions by Contract", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(FIGURE_DIR / "fig_equilibrium_regions.pdf")
    plt.close(fig)


def plot_contract_choice_heatmap(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
) -> None:
    frame = data.copy()
    frame["contract_code"] = frame["optimal_contract"].map(CONTRACT_CODE)
    pivot = frame.pivot_table(index=y, columns=x, values="contract_code", aggfunc="first").sort_index(ascending=False)
    plt.figure(figsize=(8, 5.6))
    ax = sns.heatmap(
        pivot,
        cmap=sns.color_palette(CONTRACT_COLORS, as_cmap=True),
        vmin=0,
        vmax=3,
        cbar_kws={"ticks": [0.375, 1.125, 1.875, 2.625]},
        xticklabels=4,
        yticklabels=4,
        linewidths=0.0,
    )
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(["Basic", "Incentive", "Audit", "Hybrid"])
    ax.set_title(title, pad=12)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_contract_comparison(results: pd.DataFrame) -> None:
    base = results[results["experiment"] == "base_grid"].copy()
    summary = base.groupby(["contract", "contract_label", "supplier_choice"]).size().reset_index(name="count")
    summary["share"] = summary["count"] / summary.groupby(["contract", "contract_label"])["count"].transform("sum")
    label_order = [base.loc[base["contract"] == contract, "contract_label"].iloc[0] for contract in CONTRACT_ORDER]
    pivot = summary.pivot_table(index="contract_label", columns="supplier_choice", values="share", fill_value=0)
    for col in ["none", "low", "high"]:
        if col not in pivot:
            pivot[col] = 0
    pivot = pivot.reindex(label_order)[["none", "low", "high"]]
    payoff = base.groupby("contract_label")["buyer_payoff"].mean().reindex(pivot.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), width_ratios=[1.2, 0.9])
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for col, color in zip(["none", "low", "high"], OUTCOME_COLORS):
        ax1.bar(x, pivot[col].values, bottom=bottom, color=color, label=OUTCOME_LABELS[OUTCOME_CODE[col]])
        bottom += pivot[col].values
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot.index, rotation=18, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Share of supplier outcomes")
    ax1.set_title("Supplier outcomes")
    ax1.legend(loc="upper left", frameon=False)

    payoff_colors = [CONTRACT_COLORS[CONTRACT_CODE[contract]] for contract in CONTRACT_ORDER]
    ax2.bar(x, payoff.values, color=payoff_colors)
    ax2.axhline(0, color="#333333", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot.index, rotation=18, ha="right")
    ax2.set_ylabel("Mean buyer payoff")
    ax2.set_title("Buyer payoff")
    fig.suptitle("Contract Performance Across Baseline Grid", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURE_DIR / "fig_contract_comparison.pdf")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", font_scale=0.95)

    model = load_module("02_model_setup.py", "model_setup")
    results = ensure_inputs()

    write_model_parameter_table(model)
    write_contract_parameter_table(model)
    write_taxonomy_table()
    write_literature_summary_table()
    write_contract_comparison_table(results)
    write_buyer_choice_summary_table(results)
    write_buyer_comparison_table()
    write_targeted_robustness_table(results)
    write_managerial_implications_table()

    plot_game_tree()
    plot_equilibrium_regions(results)
    plot_outcome_heatmap(
        results[results["experiment"] == "audit_penalty_heatmap"],
        "p",
        "F",
        "Audit Probability vs Penalty",
        FIGURE_DIR / "fig_audit_penalty_heatmap.pdf",
    )
    plot_outcome_heatmap(
        results[results["experiment"] == "capability_support_heatmap"],
        "q",
        "S",
        "Supplier Capability vs Buyer Support",
        FIGURE_DIR / "fig_capability_support_heatmap.pdf",
    )
    plot_contract_choice_heatmap(
        results[results["experiment"] == "buyer_choice_region"],
        "q",
        "R",
        "Buyer-Optimal Contract by Capability and Circularity Value",
        FIGURE_DIR / "fig_buyer_choice_region.pdf",
    )
    plot_contract_choice_heatmap(
        results[results["experiment"] == "governance_cost_sensitivity"],
        "k",
        "m",
        "Buyer-Optimal Contract by Support and Confidentiality Costs",
        FIGURE_DIR / "fig_governance_cost_sensitivity.pdf",
    )
    plot_contract_comparison(results)
    print(f"Wrote tables to {TABLE_DIR} and figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
