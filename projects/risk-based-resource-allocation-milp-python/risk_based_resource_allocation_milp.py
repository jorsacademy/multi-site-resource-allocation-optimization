from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Shipment:
    id: str
    units: int
    risk: float


@dataclass(frozen=True)
class ProtectionAsset:
    id: str
    effectiveness: float


@dataclass
class AllocationResult:
    status: str
    objective_value: float
    allocation: pd.DataFrame
    assets_used: int


def allocate_protection_resources(
    shipments: Dict[str, Shipment],
    assets: Dict[str, ProtectionAsset],
    max_assets: int,
    *,
    diminishing_return: float = 0.35,
    max_assets_per_shipment: int | None = None,
) -> AllocationResult:
    """
    Generic risk-based resource-allocation MILP.

    Objective:
        maximize modeled protection benefit

    Benefit of assigning asset a to the r-th protection slot of shipment s:

        units[s] * risk[s] * effectiveness[a] * diminishing_return**r

    where r=0 is the first slot. This creates explicit diminishing marginal
    returns without presenting the score as a calibrated survival probability.

    Constraints:
    - each asset can be assigned at most once;
    - each shipment slot can contain at most one asset;
    - slot r may be used only if slot r-1 is used;
    - total assets used <= max_assets.
    """
    if not shipments:
        raise ValueError("at least one shipment is required")
    if not assets:
        raise ValueError("at least one protection asset is required")
    if max_assets < 0:
        raise ValueError("max_assets must be nonnegative")
    if not (0.0 < diminishing_return <= 1.0):
        raise ValueError("diminishing_return must be in (0, 1]")

    for key, shipment in shipments.items():
        if shipment.id != key:
            raise ValueError("shipment dictionary key must match Shipment.id")
        if shipment.units <= 0:
            raise ValueError("shipment units must be positive")
        if shipment.risk <= 0:
            raise ValueError("shipment risk must be positive")

    for key, asset in assets.items():
        if asset.id != key:
            raise ValueError("asset dictionary key must match ProtectionAsset.id")
        if asset.effectiveness < 0:
            raise ValueError("asset effectiveness must be nonnegative")

    shipment_ids = list(shipments)
    asset_ids = list(assets)
    max_assets = min(int(max_assets), len(asset_ids))

    if max_assets_per_shipment is None:
        slots_per_shipment = max_assets
    else:
        if max_assets_per_shipment <= 0:
            raise ValueError("max_assets_per_shipment must be positive")
        slots_per_shipment = min(int(max_assets_per_shipment), max_assets)

    if max_assets == 0:
        rows = [
            {
                "Shipment": s,
                "Units": shipments[s].units,
                "Risk_Index": shipments[s].risk,
                "Assigned_Assets": [],
                "Protection_Score": 0.0,
            }
            for s in shipment_ids
        ]
        return AllocationResult(
            status="OPTIMAL",
            objective_value=0.0,
            allocation=pd.DataFrame(rows),
            assets_used=0,
        )

    index: Dict[Tuple[str, str, int], int] = {}
    reverse: List[Tuple[str, str, int]] = []
    for s in shipment_ids:
        for a in asset_ids:
            for r in range(slots_per_shipment):
                index[s, a, r] = len(reverse)
                reverse.append((s, a, r))

    n = len(reverse)
    c = np.zeros(n, dtype=float)

    for s, a, r in reverse:
        benefit = (
            shipments[s].units
            * shipments[s].risk
            * assets[a].effectiveness
            * (diminishing_return ** r)
        )
        c[index[s, a, r]] = -benefit

    rows: List[np.ndarray] = []
    lbs: List[float] = []
    ubs: List[float] = []

    def add(coeffs: Dict[int, float], lb=-np.inf, ub=np.inf) -> None:
        row = np.zeros(n, dtype=float)
        for i, value in coeffs.items():
            row[i] = value
        rows.append(row)
        lbs.append(lb)
        ubs.append(ub)

    for a in asset_ids:
        coeffs = {}
        for s in shipment_ids:
            for r in range(slots_per_shipment):
                coeffs[index[s, a, r]] = 1.0
        add(coeffs, ub=1.0)

    for s in shipment_ids:
        for r in range(slots_per_shipment):
            add(
                {index[s, a, r]: 1.0 for a in asset_ids},
                ub=1.0,
            )

    for s in shipment_ids:
        for r in range(1, slots_per_shipment):
            coeffs = {}
            for a in asset_ids:
                coeffs[index[s, a, r]] = 1.0
                coeffs[index[s, a, r - 1]] = (
                    coeffs.get(index[s, a, r - 1], 0.0) - 1.0
                )
            add(coeffs, ub=0.0)

    add({i: 1.0 for i in range(n)}, ub=float(max_assets))

    result = milp(
        c=c,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lbs),
            np.asarray(ubs),
        ),
    )

    if result.x is None:
        return AllocationResult(
            status="INFEASIBLE_OR_NO_SOLUTION",
            objective_value=float("-inf"),
            allocation=pd.DataFrame(),
            assets_used=0,
        )

    assigned: Dict[str, List[Tuple[int, str]]] = {s: [] for s in shipment_ids}
    score: Dict[str, float] = {s: 0.0 for s in shipment_ids}

    for s, a, r in reverse:
        if result.x[index[s, a, r]] > 0.5:
            assigned[s].append((r, a))
            score[s] += (
                shipments[s].units
                * shipments[s].risk
                * assets[a].effectiveness
                * (diminishing_return ** r)
            )

    rows_out = []
    total_used = 0
    for s in shipment_ids:
        assigned[s].sort()
        asset_list = [a for _, a in assigned[s]]
        total_used += len(asset_list)
        rows_out.append(
            {
                "Shipment": s,
                "Units": shipments[s].units,
                "Risk_Index": shipments[s].risk,
                "Assigned_Assets": asset_list,
                "Protection_Score": float(score[s]),
            }
        )

    status = "OPTIMAL" if result.status == 0 else "FEASIBLE_LIMIT"
    return AllocationResult(
        status=status,
        objective_value=float(-result.fun),
        allocation=pd.DataFrame(rows_out),
        assets_used=total_used,
    )


if __name__ == "__main__":
    shipments = {
        "S1": Shipment("S1", 45, 8),
        "S2": Shipment("S2", 37, 7),
        "S3": Shipment("S3", 32, 9),
        "S4": Shipment("S4", 40, 6),
        "S5": Shipment("S5", 28, 10),
    }

    assets = {
        "A1": ProtectionAsset("A1", 0.7),
        "A2": ProtectionAsset("A2", 0.6),
        "A3": ProtectionAsset("A3", 0.4),
        "A4": ProtectionAsset("A4", 0.4),
        "A5": ProtectionAsset("A5", 0.5),
        "A6": ProtectionAsset("A6", 0.5),
        "A7": ProtectionAsset("A7", 0.7),
        "A8": ProtectionAsset("A8", 0.6),
        "A9": ProtectionAsset("A9", 0.5),
        "A10": ProtectionAsset("A10", 0.5),
        "A11": ProtectionAsset("A11", 0.6),
        "A12": ProtectionAsset("A12", 0.6),
    }

    result = allocate_protection_resources(
        shipments,
        assets,
        max_assets=8,
        diminishing_return=0.35,
    )
    print(result.status)
    print("Objective:", result.objective_value)
    print("Assets used:", result.assets_used)
    print(result.allocation.to_string(index=False))
