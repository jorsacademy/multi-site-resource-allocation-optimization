# Risk-Based Resource Allocation MILP in Python

Educational mixed-integer linear programming model for assigning a limited set of generic protection resources to higher-risk shipments.

The repository is intentionally domain-neutral. It demonstrates binary assignment, scarce-resource constraints, and diminishing marginal benefit without presenting the score as a calibrated loss or survival probability.

## Model

For shipment `s`, asset `a`, and protection slot `r`, define:

```text
z[s,a,r] = 1 if asset a is assigned to shipment s in slot r
```

The modeled benefit is:

```text
units[s] * risk[s] * effectiveness[a] * diminishing_return^r
```

The first slot has full marginal value. Later slots receive progressively smaller benefit.

Constraints ensure:

- each asset is assigned at most once;
- each shipment slot holds at most one asset;
- later slots cannot be used before earlier slots;
- total deployed assets do not exceed the global budget;
- an optional per-shipment asset cap can be imposed.

## Important interpretation

`Protection_Score` is an optimization score only. It is **not** a survival probability, expected-loss estimate, actuarial quantity, or empirically calibrated operational forecast.

## Solver

The implementation uses `scipy.optimize.milp` with the HiGHS MILP backend.

## Validation

The regression suite checks:

- agreement with brute-force enumeration on a small instance;
- no asset is assigned more than once;
- the global asset budget is enforced;
- diminishing returns can diversify allocation across shipments;
- zero-budget behavior.

Run:

```bash
python -m unittest discover -s tests -v
```

## Usage

```python
from risk_based_resource_allocation_milp import (
    Shipment,
    ProtectionAsset,
    allocate_protection_resources,
)

shipments = {
    "S1": Shipment("S1", units=45, risk=8),
    "S2": Shipment("S2", units=37, risk=7),
}

assets = {
    "A1": ProtectionAsset("A1", effectiveness=0.7),
    "A2": ProtectionAsset("A2", effectiveness=0.6),
}

result = allocate_protection_resources(
    shipments,
    assets,
    max_assets=2,
    diminishing_return=0.35,
)

print(result.status)
print(result.objective_value)
print(result.allocation)
```

## Scope

This is a generic operations-research example. For real decision support, risk and effectiveness coefficients require domain-specific empirical calibration, uncertainty treatment, validation, and governance.
