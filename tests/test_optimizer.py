import unittest

from src.optimizer import MultiSiteResourceAllocator


class TestMultiSiteResourceAllocator(unittest.TestCase):
    def test_solution_is_feasible_and_integral(self):
        model = MultiSiteResourceAllocator(minimum_total_deployment=700)
        result = model.solve()

        allocation = result["allocation"]

        self.assertTrue(all(int(x) == x for x in allocation))
        self.assertGreaterEqual(result["total_deployed"], 700)
        self.assertLessEqual(result["total_deployed"], model.total_personnel)
        self.assertLessEqual(result["total_cost"], model.total_budget)

        for site, staff in zip(model.sites, allocation, strict=True):
            self.assertGreaterEqual(staff, site.min_staff)
            self.assertLessEqual(staff, site.max_staff)

    def test_strategic_share_floor(self):
        model = MultiSiteResourceAllocator(minimum_total_deployment=700)
        result = model.solve()

        total = result["total_deployed"]
        for site, staff in zip(model.sites, result["allocation"], strict=True):
            required_share = 0.90 * site.strategic_weight
            self.assertGreaterEqual(staff / total + 1e-9, required_share)


if __name__ == "__main__":
    unittest.main()
