"""Model setup for the DPP contracting game.

The model is a sequential game:
1. The buyer chooses a DPP governance contract.
2. The supplier chooses high-quality disclosure, low-quality disclosure, or non-disclosure.

The script writes default parameters and a reproducible parameter grid to code/data/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "code" / "data"

DISCLOSURE_ORDER = ["high", "low", "none"]


@dataclass(frozen=True)
class ModelParameters:
    """Default parameters for one buyer-supplier dyad."""

    c: float = 5.0
    cost_form: str = "hyperbolic"
    cost_floor: float = 0.25
    gamma: float = 2.0
    q: float = 0.60
    r: float = 1.50
    p: float = 0.05
    F: float = 1.00
    F_bar: float = 10.0
    B: float = 0.25
    S: float = 0.00
    S_bar: float = 3.0
    R: float = 12.0
    L: float = 5.0
    v: float = 1.0
    g: float = 0.80
    k: float = 0.0
    m: float = 0.80
    alpha: float = 1.40
    lam: float = 0.45
    beta: float = 0.35
    eta_penalty: float = 0.20
    r_low_share: float = 0.35
    confidentiality: float = 0.10
    B_L: float = 0.10
    contract_value: float = 7.00
    outside_option: float = 5.40


PARAMETER_DESCRIPTIONS = {
    "c": "Supplier baseline cost of collecting, structuring, and validating high-quality DPP data.",
    "cost_form": "Disclosure cost form used in the simulation: hyperbolic main specification or affine robustness check.",
    "cost_floor": "Lower bound on high-quality disclosure cost.",
    "gamma": "Capability cost-reduction coefficient in the affine robustness specification.",
    "q": "Supplier data capability/readiness level.",
    "r": "Supplier commercial or IP risk from disclosure.",
    "p": "Probability of buyer audit or verification.",
    "F": "Penalty if low-quality disclosure is detected.",
    "F_bar": "Maximum feasible penalty allowed by legal, relational, or enforceability constraints.",
    "B": "Buyer premium or preferred-supplier benefit for high-quality disclosure.",
    "S": "Buyer support or subsidy for supplier capability-building.",
    "S_bar": "Maximum feasible buyer support level.",
    "R": "Buyer value from high-quality DPP data.",
    "L": "Buyer loss from poor data, circularity failure, or non-compliance.",
    "v": "Buyer verification/audit cost coefficient.",
    "g": "Supplier reputational gain from verified high-quality disclosure.",
    "k": "Unit cost to the buyer of providing supplier support.",
    "m": "Buyer cost coefficient for confidentiality and access-control protection.",
    "alpha": "Effectiveness of buyer support in reducing supplier disclosure cost.",
    "lam": "Low-quality disclosure cost share relative to high-quality disclosure cost.",
    "beta": "Residual buyer value from low-quality or partial DPP data.",
    "eta_penalty": "Share of expected penalties that offsets buyer losses as recoverable damages.",
    "r_low_share": "Commercial/IP risk share under low-quality disclosure.",
    "confidentiality": "Strength of confidentiality and access-control protections.",
    "B_L": "Supplier benefit from low-quality disclosure, such as continued transaction access.",
    "contract_value": "Supplier gross value from retaining the buyer relationship.",
    "outside_option": "Supplier payoff from non-disclosure and alternative opportunities.",
}

CONTRACT_INSTRUMENTS = ["p", "F", "B", "B_L", "S", "confidentiality"]


CONTRACTS: Dict[str, Dict[str, float | str]] = {
    "basic_compliance": {
        "label": "Basic compliance",
        "p": 0.05,
        "F": 1.0,
        "B": 0.25,
        "B_L": 0.10,
        "S": 0.0,
        "k": 0.0,
        "confidentiality": 0.10,
    },
    "incentive_only": {
        "label": "Incentive-only",
        "p": 0.08,
        "F": 1.5,
        "B": 2.40,
        "B_L": 0.40,
        "S": 0.0,
        "k": 0.0,
        "confidentiality": 0.20,
    },
    "audit_penalty": {
        "label": "Audit/penalty",
        "p": 0.45,
        "F": 7.0,
        "B": 0.35,
        "B_L": 0.05,
        "S": 0.0,
        "k": 0.0,
        "confidentiality": 0.15,
    },
    "hybrid_support": {
        "label": "Hybrid",
        "p": 0.32,
        "F": 5.0,
        "B": 1.80,
        "B_L": 0.20,
        "S": 1.60,
        "k": 0.80,
        "confidentiality": 0.55,
    },
}


def apply_contract(
    contract_name: str,
    base: ModelParameters | None = None,
    overrides: Mapping[str, float | str] | None = None,
) -> ModelParameters:
    """Return parameters after applying a contract scenario and optional overrides."""

    if contract_name not in CONTRACTS:
        raise KeyError(f"Unknown contract: {contract_name}")
    params = base or ModelParameters()
    scenario_values = {
        key: value
        for key, value in CONTRACTS[contract_name].items()
        if key != "label"
    }
    params = replace(params, **scenario_values)
    if overrides:
        params = replace(params, **dict(overrides))
    return enforce_bounds(params)


def enforce_bounds(params: ModelParameters) -> ModelParameters:
    """Apply feasibility bounds to contract instruments."""

    return replace(
        params,
        p=min(max(float(params.p), 0.0), 1.0),
        F=min(max(float(params.F), 0.0), max(float(params.F_bar), 0.0)),
        S=min(max(float(params.S), 0.0), max(float(params.S_bar), 0.0)),
        confidentiality=min(max(float(params.confidentiality), 0.0), 1.0),
    )


def disclosure_costs(params: ModelParameters) -> Tuple[float, float]:
    """Return high-quality and low-quality disclosure costs."""

    if params.cost_form == "affine":
        high_cost = params.c - params.gamma * params.q - params.alpha * params.S
    else:
        effective_capability = max(params.q + params.alpha * params.S, 1e-9)
        high_cost = params.c / effective_capability
    high_cost = max(high_cost, params.cost_floor)
    low_cost = params.lam * high_cost
    return high_cost, low_cost


def disclosure_risks(params: ModelParameters) -> Tuple[float, float]:
    """Return confidentiality-adjusted high- and low-disclosure commercial risks."""

    protection = min(max(params.confidentiality, 0.0), 0.95)
    high_risk = params.r * (1.0 - protection)
    low_risk = params.r_low_share * high_risk
    return high_risk, low_risk


def supplier_payoffs(params: ModelParameters) -> Dict[str, float]:
    """Supplier payoffs for each disclosure strategy."""

    high_cost, low_cost = disclosure_costs(params)
    high_risk, low_risk = disclosure_risks(params)
    return {
        "high": params.contract_value + params.B + params.g - high_cost - high_risk,
        "low": params.contract_value + params.B_L - low_cost - params.p * params.F - low_risk,
        "none": params.outside_option,
    }


def buyer_payoffs(params: ModelParameters) -> Dict[str, float]:
    """Buyer payoffs conditional on the supplier disclosure strategy."""

    support_cost = params.k * params.S
    audit_cost = params.v * params.p**2
    confidentiality_cost = params.m * params.confidentiality**2
    return {
        "high": params.R - params.B - audit_cost - support_cost - confidentiality_cost,
        "low": params.beta * params.R - params.L - audit_cost - params.B_L - support_cost - confidentiality_cost
        + params.eta_penalty * params.p * params.F,
        "none": -params.L,
    }


def supplier_best_response(params: ModelParameters) -> str:
    """Return the supplier best response, preferring higher-quality disclosure on exact ties."""

    payoffs = supplier_payoffs(params)
    return max(DISCLOSURE_ORDER, key=lambda strategy: (payoffs[strategy], -DISCLOSURE_ORDER.index(strategy)))


def evaluate_contract(
    contract_name: str,
    base: ModelParameters | None = None,
    overrides: Mapping[str, float | str] | None = None,
) -> Dict[str, float | str]:
    """Evaluate one contract scenario under supplier best response."""

    params = apply_contract(contract_name, base=base, overrides=overrides)
    supplier = supplier_payoffs(params)
    buyer = buyer_payoffs(params)
    response = supplier_best_response(params)
    row: Dict[str, float | str] = {
        "contract": contract_name,
        "contract_label": str(CONTRACTS[contract_name]["label"]),
        "supplier_choice": response,
        "supplier_payoff": supplier[response],
        "buyer_payoff": buyer[response],
    }
    row.update({f"supplier_payoff_{key}": value for key, value in supplier.items()})
    row.update({f"buyer_payoff_{key}": value for key, value in buyer.items()})
    row.update(asdict(params))
    return row


def buyer_optimal_contract(
    contracts: Iterable[str] = CONTRACTS.keys(),
    base: ModelParameters | None = None,
    overrides: Mapping[str, float | str] | None = None,
) -> Dict[str, float | str]:
    """Return the contract that maximizes buyer payoff after supplier response."""

    evaluations = [evaluate_contract(name, base=base, overrides=overrides) for name in contracts]
    return max(evaluations, key=lambda row: float(row["buyer_payoff"]))


def make_parameter_grid() -> pd.DataFrame:
    """Create a compact parameter grid for the baseline simulation."""

    rows = []
    for contract in CONTRACTS:
        for q in np.round(np.linspace(0.20, 1.20, 6), 3):
            for c in [3.5, 5.0, 6.5, 8.0]:
                for r in [0.5, 1.5, 2.5, 3.5]:
                    for R in [9.0, 12.0, 15.0]:
                        for outside_option in [4.2, 5.4]:
                            rows.append(
                                {
                                    "experiment": "base_grid",
                                    "contract": contract,
                                    "q": q,
                                    "c": c,
                                    "r": r,
                                    "R": R,
                                    "outside_option": outside_option,
                                }
                            )
    return pd.DataFrame(rows)


def write_default_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    defaults = asdict(ModelParameters())
    payload = {
        "model": "sequential_buyer_supplier_dpp_contracting_game",
        "parameters": defaults,
        "descriptions": PARAMETER_DESCRIPTIONS,
        "contracts": CONTRACTS,
        "cost_functions": {
            "high_quality_disclosure_cost": "C_H = max{c / (q + alpha*S), cost_floor}",
            "affine_robustness_cost": "C_H = max{c - gamma*q - alpha*S, cost_floor}",
            "low_quality_disclosure_cost": "C_L = lambda*C_H",
            "expected_penalty_low_quality": "p*F",
            "audit_cost": "A(p) = v*p^2",
            "commercial_risk_high": "r_H = r*(1-confidentiality)",
            "commercial_risk_low": "r_L = r_low_share*r_H",
            "confidentiality_cost": "M(phi) = m*confidentiality^2",
        },
    }
    with (DATA_DIR / "model_parameters.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    make_parameter_grid().to_csv(DATA_DIR / "parameter_grid.csv", index=False)


def main() -> None:
    write_default_files()
    print(f"Wrote model inputs to {DATA_DIR}")


if __name__ == "__main__":
    main()
