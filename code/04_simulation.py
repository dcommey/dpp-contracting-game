"""Numerical simulation for DPP contracting game equilibrium regions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "code" / "data"


def load_model_module():
    spec = importlib.util.spec_from_file_location("model_setup", CODE_DIR / "02_model_setup.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["model_setup"] = module
    spec.loader.exec_module(module)
    return module


def evaluate_rows(
    model,
    experiment: str,
    contract: str,
    rows: Iterable[Mapping[str, float | str]],
) -> list[Dict[str, float | str]]:
    output = []
    for overrides in rows:
        result = model.evaluate_contract(contract, overrides=overrides)
        result["experiment"] = experiment
        result.update(overrides)
        output.append(result)
    return output


def base_grid_results(model) -> list[Dict[str, float | str]]:
    grid_path = DATA_DIR / "parameter_grid.csv"
    if not grid_path.exists():
        model.write_default_files()
    grid = pd.read_csv(grid_path)
    rows = []
    for record in grid.to_dict("records"):
        contract = str(record.pop("contract"))
        experiment = str(record.pop("experiment"))
        result = model.evaluate_contract(contract, overrides=record)
        result["experiment"] = experiment
        rows.append(result)
    return rows


def heatmap_results(model) -> list[Dict[str, float | str]]:
    rows: list[Dict[str, float | str]] = []

    # Keep participation feasible here so the heatmap isolates the deterrence threshold.
    audit_rows = (
        {
            "p": float(p),
            "F": float(F),
            "q": 0.80,
            "c": 4.0,
            "r": 0.60,
            "contract_value": 8.0,
            "outside_option": 3.0,
        }
        for p in np.round(np.linspace(0.0, 0.90, 19), 3)
        for F in np.round(np.linspace(0.0, 10.0, 21), 3)
    )
    rows.extend(evaluate_rows(model, "audit_penalty_heatmap", "audit_penalty", audit_rows))

    support_rows = (
        {"q": float(q), "S": float(S), "c": 6.5, "r": 2.0}
        for q in np.round(np.linspace(0.15, 1.20, 22), 3)
        for S in np.round(np.linspace(0.0, 3.0, 21), 3)
    )
    rows.extend(evaluate_rows(model, "capability_support_heatmap", "hybrid_support", support_rows))

    region_rows = [
        {
            "q": float(q),
            "r": float(r),
            "c": 4.5,
            "contract_value": 7.5,
            "outside_option": 3.5,
        }
        for q in np.round(np.linspace(0.15, 1.20, 22), 3)
        for r in np.round(np.linspace(0.0, 4.0, 21), 3)
    ]
    for contract in model.CONTRACTS:
        rows.extend(evaluate_rows(model, "equilibrium_region_grid", contract, region_rows))

    cost_risk_rows = (
        {"c": float(c), "r": float(r), "q": 0.65}
        for c in np.round(np.linspace(2.0, 9.0, 22), 3)
        for r in np.round(np.linspace(0.0, 4.0, 21), 3)
    )
    rows.extend(evaluate_rows(model, "cost_risk_heatmap", "hybrid_support", cost_risk_rows))

    buyer_value_rows = (
        {"R": float(R), "v": float(v), "q": 0.65, "c": 5.0, "r": 1.5}
        for R in np.round(np.linspace(6.0, 18.0, 25), 3)
        for v in np.round(np.linspace(0.0, 4.0, 17), 3)
    )
    rows.extend(evaluate_rows(model, "buyer_value_audit_cost_grid", "audit_penalty", buyer_value_rows))

    return rows


def buyer_choice_results(model) -> list[Dict[str, float | str]]:
    """Evaluate the buyer's optimal discrete contract across policy-relevant grids."""

    rows: list[Dict[str, float | str]] = []

    for q in np.round(np.linspace(0.20, 1.20, 21), 3):
        for R in np.round(np.linspace(0.0, 14.0, 29), 3):
            overrides = {
                "q": float(q),
                "R": float(R),
                "c": 5.5,
                "r": 1.8,
                "k": 3.0,
                "m": 3.0,
                "outside_option": 4.4,
            }
            result = model.buyer_optimal_contract(overrides=overrides)
            result["experiment"] = "buyer_choice_region"
            result["optimal_contract"] = result["contract"]
            result.update(overrides)
            rows.append(result)

    for k in np.round(np.linspace(0.1, 8.0, 25), 3):
        for m in np.round(np.linspace(0.1, 8.0, 25), 3):
            overrides = {
                "k": float(k),
                "m": float(m),
                "q": 0.35,
                "c": 7.0,
                "r": 3.2,
                "R": 5.0,
                "L": 3.5,
                "outside_option": 5.4,
            }
            result = model.buyer_optimal_contract(overrides=overrides)
            result["experiment"] = "governance_cost_sensitivity"
            result["optimal_contract"] = result["contract"]
            result.update(overrides)
            rows.append(result)

    rng = np.random.default_rng(20260605)
    for draw_id in range(1200):
        cost_form = "affine" if draw_id >= 600 else "hyperbolic"
        overrides: Dict[str, float | str] = {
            "cost_form": cost_form,
            "q": float(rng.uniform(0.20, 1.20)),
            "c": float(rng.uniform(3.0, 8.5)),
            "r": float(rng.uniform(0.3, 4.0)),
            "R": float(rng.uniform(7.0, 20.0)),
            "L": float(rng.uniform(3.0, 8.0)),
            "v": float(rng.uniform(0.3, 3.0)),
            "k": float(rng.uniform(0.1, 3.0)),
            "m": float(rng.uniform(0.1, 3.0)),
            "contract_value": float(rng.uniform(6.0, 9.5)),
            "outside_option": float(rng.uniform(3.0, 6.5)),
        }
        result = model.buyer_optimal_contract(overrides=overrides)
        result["experiment"] = "monte_carlo_buyer_choice"
        result["optimal_contract"] = result["contract"]
        result["draw_id"] = draw_id
        result.update(overrides)
        rows.append(result)

    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model_module()
    model.write_default_files()
    rows = base_grid_results(model)
    rows.extend(heatmap_results(model))
    rows.extend(buyer_choice_results(model))
    results = pd.DataFrame(rows)
    results.to_csv(DATA_DIR / "simulation_results.csv", index=False)
    print(f"Wrote {len(results):,} simulation rows to {DATA_DIR / 'simulation_results.csv'}")


if __name__ == "__main__":
    main()
