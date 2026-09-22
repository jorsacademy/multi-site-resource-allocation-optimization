# Hungarian Assignment Optimization in Python

A compact operations-research example for solving minimum-cost one-to-one assignment problems with SciPy's Hungarian algorithm implementation (`linear_sum_assignment`).

## What it demonstrates

- Minimum-cost worker-to-task assignment
- Rectangular cost matrices
- Input validation
- Structured result objects with Python dataclasses
- Reproducible example data
- Matplotlib visualization of the cost matrix and selected assignments
- Two examples:
  - Textile factory worker-to-station assignment
  - Nurse-to-shift assignment

> The nurse example is an assignment model, not a full staff-rostering model. It does not model multi-day coverage, legal rest rules, skill constraints, or consecutive-shift constraints.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python assignment_optimizer.py
```

## Core model

Given a cost matrix \(C\), the model chooses one-to-one worker/task pairs that minimize total selected cost. SciPy solves the linear sum assignment problem using `scipy.optimize.linear_sum_assignment`.

For rectangular matrices, the number of assignments is:

```text
min(number of workers, number of tasks)
```

## Why this repository is focused

Transportation optimization is intentionally excluded. It is a different linear-programming model and is better kept in a dedicated repository rather than mixing unrelated OR examples into one script.
