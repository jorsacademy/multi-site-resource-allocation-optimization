from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Site:
    name: str
    site_type: str
    priority: str
    min_staff: int
    max_staff: int
    monthly_cost_per_person: float
    strategic_weight: float


class MultiSiteResourceAllocator:
    """MILP model for allocating staff across multiple operational sites."""

    def __init__(
        self,
        total_personnel: int = 1200,
        total_budget: float = 50_000_000,
        operation_months: int = 3,
        minimum_total_deployment: int = 700,
    ) -> None:
        self.total_personnel = total_personnel
        self.total_budget = total_budget
        self.operation_months = operation_months
        self.minimum_total_deployment = minimum_total_deployment

        self.sites: List[Site] = [
            Site(
                name="Alpha",
                site_type="Urban Service Hub",
                priority="High",
                min_staff=150,
                max_staff=400,
                monthly_cost_per_person=5_300,
                strategic_weight=0.40,
            ),
            Site(
                name="Bravo",
                site_type="Rural Service Center",
                priority="Medium",
                min_staff=100,
                max_staff=350,
                monthly_cost_per_person=5_000,
                strategic_weight=0.35,
            ),
            Site(
                name="Charlie",
                site_type="Mountain Support Post",
                priority="Low",
                min_staff=80,
                max_staff=300,
                monthly_cost_per_person=5_240,
                strategic_weight=0.25,
            ),
        ]

    @property
    def total_cost_coefficients(self) -> np.ndarray:
        return np.array(
            [s.monthly_cost_per_person * self.operation_months for s in self.sites],
            dtype=float,
        )

    def _build_constraints(self) -> LinearConstraint:
        n = len(self.sites)
        rows = []
        lower = []
        upper = []

        # Total deployed staff cannot exceed available personnel.
        rows.append(np.ones(n))
        lower.append(-np.inf)
        upper.append(self.total_personnel)

        # Require enough deployed staff for the planning scenario.
        rows.append(np.ones(n))
        lower.append(self.minimum_total_deployment)
        upper.append(np.inf)

        # Total budget constraint.
        rows.append(self.total_cost_coefficients)
        lower.append(-np.inf)
        upper.append(self.total_budget)

        # Strategic allocation:
        # each site should receive at least 90% of its weighted share
        # of the total deployment.
        tolerance = 0.90
        for i, site in enumerate(self.sites):
            row = np.full(n, tolerance * site.strategic_weight)
            row[i] -= 1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)

        return LinearConstraint(
            np.vstack(rows),
            np.array(lower, dtype=float),
            np.array(upper, dtype=float),
        )

    def solve(self) -> Dict[str, object]:
        c = self.total_cost_coefficients

        lb = np.array([s.min_staff for s in self.sites], dtype=float)
        ub = np.array([s.max_staff for s in self.sites], dtype=float)
        bounds = Bounds(lb=lb, ub=ub)

        result = milp(
            c=c,
            integrality=np.ones(len(self.sites), dtype=int),
            bounds=bounds,
            constraints=self._build_constraints(),
            options={"disp": False},
        )

        if not result.success or result.x is None:
            raise RuntimeError(f"Optimization failed: {result.message}")

        allocation = np.rint(result.x).astype(int)
        total_deployed = int(allocation.sum())
        total_cost = float(c @ allocation)

        records = []
        for site, staff, unit_total_cost in zip(
            self.sites, allocation, c, strict=True
        ):
            site_cost = float(staff * unit_total_cost)
            records.append(
                {
                    "site": site.name,
                    "type": site.site_type,
                    "priority": site.priority,
                    "staff": int(staff),
                    "share_pct": 100.0 * staff / total_deployed,
                    "monthly_cost_per_person": site.monthly_cost_per_person,
                    "three_month_cost": site_cost,
                    "capacity_utilization_pct": 100.0 * staff / site.max_staff,
                }
            )

        return {
            "status": result.message,
            "allocation": allocation,
            "total_deployed": total_deployed,
            "total_cost": total_cost,
            "budget_remaining": self.total_budget - total_cost,
            "personnel_remaining": self.total_personnel - total_deployed,
            "table": pd.DataFrame(records),
        }

    def sensitivity_analysis(
        self,
        deployment_levels: List[int] | None = None,
    ) -> pd.DataFrame:
        if deployment_levels is None:
            deployment_levels = [500, 600, 700, 800, 900, 1000]

        rows = []
        for level in deployment_levels:
            scenario = MultiSiteResourceAllocator(
                total_personnel=self.total_personnel,
                total_budget=self.total_budget,
                operation_months=self.operation_months,
                minimum_total_deployment=level,
            )
            try:
                solved = scenario.solve()
                allocation = solved["allocation"]
                rows.append(
                    {
                        "minimum_total_deployment": level,
                        "alpha": int(allocation[0]),
                        "bravo": int(allocation[1]),
                        "charlie": int(allocation[2]),
                        "total_deployed": int(solved["total_deployed"]),
                        "total_cost": float(solved["total_cost"]),
                        "feasible": True,
                    }
                )
            except RuntimeError:
                rows.append(
                    {
                        "minimum_total_deployment": level,
                        "alpha": None,
                        "bravo": None,
                        "charlie": None,
                        "total_deployed": None,
                        "total_cost": None,
                        "feasible": False,
                    }
                )

        return pd.DataFrame(rows)


def main() -> None:
    allocator = MultiSiteResourceAllocator()
    result = allocator.solve()

    print("=== MULTI-SITE RESOURCE ALLOCATION ===")
    print(f"Status: {result['status']}")
    print(f"Total deployed: {result['total_deployed']}")
    print(f"Total cost: ${result['total_cost']:,.2f}")
    print(f"Budget remaining: ${result['budget_remaining']:,.2f}")
    print(f"Personnel remaining: {result['personnel_remaining']}")
    print()
    print(result["table"].to_string(index=False))

    print("\n=== DEPLOYMENT SENSITIVITY ===")
    print(allocator.sensitivity_analysis().to_string(index=False))


if __name__ == "__main__":
    main()
