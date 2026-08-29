# Multi-Site Resource Allocation Optimization

A small operations-research project that allocates integer staffing across three service sites while respecting staffing bounds, budget limits, total workforce availability, deployment demand, and strategic allocation targets.

The project is a neutral multi-site planning example intended for learning and demonstration.

## Why MILP instead of plain LP?

Personnel counts are discrete. A continuous linear program can return fractional staffing and then rely on rounding, which may invalidate constraints. This implementation uses `scipy.optimize.milp`, so staffing decisions are integer variables inside the optimization model.

## Model

Decision variables:

- `x_i`: integer staff assigned to site `i`

Objective:

```text
minimize total operating cost
```

Subject to:

- total staffing cannot exceed workforce availability,
- total staffing must meet a minimum deployment/service requirement,
- each site must remain within its minimum and maximum staffing limits,
- total operating cost must remain within budget,
- each site must receive at least 90% of its strategic weighted share.

Default strategic weights:

| Site | Weight |
|---|---:|
| Alpha | 40% |
| Bravo | 35% |
| Charlie | 25% |

The 90% factor creates a small optimization tolerance while still forcing the solution to respect the intended strategic distribution.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── optimizer.py
└── tests/
    └── test_optimizer.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python -m src.optimizer
```

## Test

```bash
python -m unittest discover -s tests
```

## Sensitivity analysis

The model includes a deployment-level sensitivity analysis. Instead of changing a budget that may be non-binding, it changes the required total deployment level. This produces materially different optimization scenarios and is therefore more informative.

## Notes

This version intentionally separates the optimization logic from presentation concerns. It does not embed plotting into the solver class, which keeps the mathematical model easier to test, reuse, and extend.
