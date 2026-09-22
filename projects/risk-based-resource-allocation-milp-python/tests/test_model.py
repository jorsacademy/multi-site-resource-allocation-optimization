import itertools
import math
import unittest

from risk_based_resource_allocation_milp import (
    Shipment,
    ProtectionAsset,
    allocate_protection_resources,
)


class RiskAllocationMILPTests(unittest.TestCase):
    def test_small_instance_matches_bruteforce(self):
        shipments = {
            "S1": Shipment("S1", 10, 3),
            "S2": Shipment("S2", 8, 4),
        }
        assets = {
            "A1": ProtectionAsset("A1", 1.0),
            "A2": ProtectionAsset("A2", 0.5),
        }
        decay = 0.5
        result = allocate_protection_resources(
            shipments, assets, max_assets=2, diminishing_return=decay
        )
        self.assertEqual(result.status, "OPTIMAL")

        best = -1.0
        for a1 in (None, "S1", "S2"):
            for a2 in (None, "S1", "S2"):
                assignment = {"A1": a1, "A2": a2}
                if sum(v is not None for v in assignment.values()) > 2:
                    continue
                score = 0.0
                for s in shipments:
                    effs = sorted(
                        [assets[a].effectiveness for a, v in assignment.items() if v == s],
                        reverse=True,
                    )
                    for r, eff in enumerate(effs):
                        score += shipments[s].units * shipments[s].risk * eff * (decay ** r)
                best = max(best, score)
        self.assertTrue(math.isclose(result.objective_value, best, abs_tol=1e-9))

    def test_each_asset_used_at_most_once(self):
        shipments = {
            "S1": Shipment("S1", 10, 3),
            "S2": Shipment("S2", 12, 2),
            "S3": Shipment("S3", 8, 5),
        }
        assets = {
            "A1": ProtectionAsset("A1", 0.8),
            "A2": ProtectionAsset("A2", 0.7),
            "A3": ProtectionAsset("A3", 0.6),
        }
        result = allocate_protection_resources(shipments, assets, max_assets=3)
        used = []
        for xs in result.allocation["Assigned_Assets"]:
            used.extend(xs)
        self.assertEqual(len(used), len(set(used)))

    def test_max_assets_is_enforced(self):
        shipments = {"S1": Shipment("S1", 10, 3)}
        assets = {
            f"A{i}": ProtectionAsset(f"A{i}", 1.0)
            for i in range(5)
        }
        result = allocate_protection_resources(shipments, assets, max_assets=2)
        self.assertEqual(result.assets_used, 2)

    def test_diminishing_return_changes_distribution(self):
        shipments = {
            "S1": Shipment("S1", 10, 10),
            "S2": Shipment("S2", 10, 9),
        }
        assets = {
            "A1": ProtectionAsset("A1", 1.0),
            "A2": ProtectionAsset("A2", 1.0),
        }
        result = allocate_protection_resources(
            shipments, assets, max_assets=2, diminishing_return=0.1
        )
        counts = sorted(len(x) for x in result.allocation["Assigned_Assets"])
        self.assertEqual(counts, [1, 1])

    def test_zero_budget_returns_empty_allocation(self):
        shipments = {"S1": Shipment("S1", 10, 3)}
        assets = {"A1": ProtectionAsset("A1", 1.0)}
        result = allocate_protection_resources(shipments, assets, max_assets=0)
        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.assets_used, 0)
        self.assertEqual(result.objective_value, 0.0)


if __name__ == "__main__":
    unittest.main()
