from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Assignment:
    """One selected worker-to-task assignment."""

    worker: str
    task: str
    cost: float


@dataclass(frozen=True)
class AssignmentResult:
    """Optimal assignment solution returned by the Hungarian algorithm."""

    assignments: list[Assignment]
    total_cost: float
    row_indices: np.ndarray
    col_indices: np.ndarray


class AssignmentOptimizer:
    """Minimum-cost one-to-one assignment optimizer."""

    @staticmethod
    def _validate_cost_matrix(
        cost_matrix: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        costs = np.asarray(cost_matrix, dtype=float)

        if costs.ndim != 2 or costs.size == 0:
            raise ValueError("cost_matrix must be a non-empty 2D matrix.")
        if not np.isfinite(costs).all():
            raise ValueError("cost_matrix must contain only finite numeric values.")

        return costs

    @staticmethod
    def _validate_names(
        names: Sequence[str] | None,
        expected_count: int,
        prefix: str,
    ) -> list[str]:
        if names is None:
            return [f"{prefix}_{i + 1}" for i in range(expected_count)]

        if len(names) != expected_count:
            raise ValueError(
                f"Expected {expected_count} {prefix.lower()} names, got {len(names)}."
            )

        return list(names)

    def solve(
        self,
        cost_matrix: np.ndarray | Sequence[Sequence[float]],
        worker_names: Sequence[str] | None = None,
        task_names: Sequence[str] | None = None,
    ) -> AssignmentResult:
        """
        Solve a minimum-cost one-to-one assignment problem.

        Rectangular matrices are supported. The number of selected assignments
        equals min(number_of_workers, number_of_tasks).
        """
        costs = self._validate_cost_matrix(cost_matrix)
        n_workers, n_tasks = costs.shape

        workers = self._validate_names(worker_names, n_workers, "Worker")
        tasks = self._validate_names(task_names, n_tasks, "Task")

        row_indices, col_indices = linear_sum_assignment(costs)

        assignments = [
            Assignment(
                worker=workers[row],
                task=tasks[col],
                cost=float(costs[row, col]),
            )
            for row, col in zip(row_indices, col_indices)
        ]

        total_cost = float(costs[row_indices, col_indices].sum())

        return AssignmentResult(
            assignments=assignments,
            total_cost=total_cost,
            row_indices=row_indices,
            col_indices=col_indices,
        )

    @staticmethod
    def visualize(
        cost_matrix: np.ndarray | Sequence[Sequence[float]],
        result: AssignmentResult,
        worker_names: Sequence[str] | None = None,
        task_names: Sequence[str] | None = None,
        title: str = "Minimum-Cost Assignment",
    ) -> None:
        """Plot the cost matrix and highlight selected assignments."""
        costs = AssignmentOptimizer._validate_cost_matrix(cost_matrix)
        n_workers, n_tasks = costs.shape

        workers = AssignmentOptimizer._validate_names(
            worker_names, n_workers, "Worker"
        )
        tasks = AssignmentOptimizer._validate_names(task_names, n_tasks, "Task")

        selected = np.zeros(costs.shape, dtype=int)
        selected[result.row_indices, result.col_indices] = 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title)

        cost_image = axes[0].imshow(costs, aspect="auto")
        axes[0].set_title("Cost Matrix")
        axes[0].set_xlabel("Tasks")
        axes[0].set_ylabel("Workers")
        axes[0].set_xticks(range(n_tasks), labels=tasks, rotation=45, ha="right")
        axes[0].set_yticks(range(n_workers), labels=workers)
        fig.colorbar(cost_image, ax=axes[0], fraction=0.046, pad=0.04)

        for i in range(n_workers):
            for j in range(n_tasks):
                axes[0].text(
                    j,
                    i,
                    f"{costs[i, j]:.1f}",
                    ha="center",
                    va="center",
                )

        assignment_image = axes[1].imshow(selected, vmin=0, vmax=1, aspect="auto")
        axes[1].set_title("Optimal Assignment (1 = selected)")
        axes[1].set_xlabel("Tasks")
        axes[1].set_ylabel("Workers")
        axes[1].set_xticks(range(n_tasks), labels=tasks, rotation=45, ha="right")
        axes[1].set_yticks(range(n_workers), labels=workers)
        fig.colorbar(assignment_image, ax=axes[1], fraction=0.046, pad=0.04)

        for i in range(n_workers):
            for j in range(n_tasks):
                axes[1].text(j, i, str(selected[i, j]), ha="center", va="center")

        fig.tight_layout()
        plt.show()


def textile_factory_example() -> AssignmentResult:
    """Assign eight workers to eight textile production stations."""
    processing_times = np.array(
        [
            [7.2, 8.1, 6.5, 9.2, 7.8, 8.5, 6.9, 7.6],
            [8.0, 7.3, 7.1, 8.8, 9.1, 7.4, 8.2, 7.0],
            [6.8, 9.0, 8.3, 7.5, 6.7, 8.9, 7.7, 8.4],
            [7.9, 6.9, 7.8, 8.1, 8.6, 7.2, 6.8, 9.0],
            [8.5, 8.2, 6.6, 7.4, 7.1, 8.7, 9.2, 7.3],
            [7.1, 7.8, 8.9, 6.8, 8.3, 7.0, 7.5, 8.1],
            [8.8, 7.0, 7.4, 8.6, 7.7, 8.0, 8.4, 6.9],
            [6.7, 8.4, 8.0, 7.2, 8.9, 6.6, 7.8, 8.3],
        ],
        dtype=float,
    )

    workers = [f"Worker_{chr(65 + i)}" for i in range(8)]
    stations = [
        "Cutting",
        "Sewing",
        "Pressing",
        "Quality_Check",
        "Packaging",
        "Embroidery",
        "Finishing",
        "Inspection",
    ]

    optimizer = AssignmentOptimizer()
    result = optimizer.solve(processing_times, workers, stations)

    print("=== TEXTILE FACTORY ASSIGNMENT ===")
    print(f"Minimum total processing time: {result.total_cost:.1f} hours/day")
    for assignment in result.assignments:
        print(
            f"{assignment.worker} -> {assignment.task}: "
            f"{assignment.cost:.1f} hours"
        )

    optimizer.visualize(
        processing_times,
        result,
        workers,
        stations,
        title="Textile Factory Assignment",
    )
    return result


def nurse_shift_assignment_example() -> AssignmentResult:
    """
    Assign nurses to one shift each.

    This is intentionally an assignment example, not a full staff-rostering
    model with multi-day coverage, rest, skill, and labor-rule constraints.
    """
    rng = np.random.default_rng(123)
    costs = rng.integers(50, 150, size=(10, 10)).astype(float)

    # Soft preferences represented as higher assignment costs.
    costs[0:3, 6:10] *= 1.5
    costs[7:10, 0:4] *= 1.3

    nurses = [f"Nurse_{i + 1}" for i in range(10)]
    shifts = [
        "Day_ICU",
        "Day_ER",
        "Day_Surgery",
        "Day_Pediatrics",
        "Day_General",
        "Night_ICU",
        "Night_ER",
        "Night_Surgery",
        "Night_Pediatrics",
        "Night_General",
    ]

    optimizer = AssignmentOptimizer()
    result = optimizer.solve(costs, nurses, shifts)

    print("\n=== NURSE-SHIFT ASSIGNMENT ===")
    print(f"Minimum assignment cost: ${result.total_cost:,.0f}/week")
    for assignment in result.assignments:
        print(
            f"{assignment.worker} -> {assignment.task}: "
            f"${assignment.cost:,.0f}"
        )

    optimizer.visualize(
        costs,
        result,
        nurses,
        shifts,
        title="Nurse-Shift Assignment",
    )
    return result


def main() -> None:
    textile_result = textile_factory_example()
    nurse_result = nurse_shift_assignment_example()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Textile assignment: {textile_result.total_cost:.1f} hours/day")
    print(f"Nurse-shift assignment: ${nurse_result.total_cost:,.0f}/week")


if __name__ == "__main__":
    main()
