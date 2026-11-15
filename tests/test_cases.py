"""Scenario tests: integer cells, zero rows/columns, ties, and cases that
require undoing earlier assignments (where greedy commitment is provably
worse than the global optimum)."""

from __future__ import annotations

import sys

import brute
import rounding


def check_feasible(A, q, result):
    n, m = len(A), len(A[0])
    matrix = result["matrix"]
    assert len(matrix) == n and all(len(row) == m for row in matrix)
    for i in range(n):
        for j in range(m):
            assert isinstance(matrix[i][j], int) and not isinstance(matrix[i][j], bool)
            assert matrix[i][j] in (A[i][j] // q, A[i][j] // q + 1), (i, j)
        assert sum(matrix[i]) == sum(A[i]) // q, f"row {i} total changed"
    for j in range(m):
        assert sum(matrix[i][j] for i in range(n)) == sum(
            A[i][j] for i in range(n)
        ) // q, f"column {j} total changed"
    recomputed = sum(
        abs(q * matrix[i][j] - A[i][j]) for i in range(n) for j in range(m)
    )
    assert recomputed == result["cost"], "cost does not match the matrix"
    assert isinstance(result["cost"], int) and not isinstance(result["cost"], bool)


def main():
    scenario_one = [[6, 4], [4, 6]]
    result = rounding.round_table(scenario_one, 10)
    assert result == {"matrix": [[1, 0], [0, 1]], "cost": 16}, result
    check_feasible(scenario_one, 10, result)
    print("1. specification example         : cost 16, [[1,0],[0,1]]")

    # Integer cells: no rounding needed at all.
    integer = [[10, 20], [30, 40]]
    result = rounding.round_table(integer, 10)
    assert result == {"matrix": [[1, 2], [3, 4]], "cost": 0}, result
    cost, count, _ = brute.brute_force(integer, 10)
    assert (cost, count) == (0, 1)
    print("2. integer cells                 : cost 0, matrix unchanged")

    # Zero row and zero column present, plus a zero cell block.
    sparse = [[0, 0, 0], [0, 10, 10], [0, 10, 10]]
    result = rounding.round_table(sparse, 10)
    check_feasible(sparse, 10, result)
    cost, count, _ = brute.brute_force(sparse, 10)
    assert cost == 0 and count == 1
    assert result["cost"] == 0
    assert result["matrix"][0] == [0, 0, 0]
    assert all(row[0] == 0 for row in result["matrix"])
    print("3. zero row / zero column        : cost 0, zeros preserved")

    # Zero table: every cell is an exact integer multiple.
    zeros = [[0, 0], [0, 0]]
    assert rounding.round_table(zeros, 3) == {"matrix": [[0, 0], [0, 0]], "cost": 0}
    print("4. all-zero table                : cost 0")

    # Tie: both permutation-looking solutions cost 20; either is acceptable.
    tie = [[5, 5], [5, 5]]
    result = rounding.round_table(tie, 10)
    check_feasible(tie, 10, result)
    cost, count, _ = brute.brute_force(tie, 10)
    assert cost == 20 and count == 2, (cost, count)
    assert result["cost"] == 20
    assert result["matrix"] in ([[1, 0], [0, 1]], [[0, 1], [1, 0]])
    print("5. tied optima (either accepted) : cost 20, 2 optimal matrices")

    # Requiring an undo: the unique optimum is strictly better than the
    # greedy row-by-row commitment, whose first choice must be revoked.
    undo = [[10, 11, 19], [15, 15, 10], [15, 14, 11]]
    undo_result = rounding.round_table(undo, 10)
    check_feasible(undo, 10, undo_result)
    cost, count, witness = brute.brute_force(undo, 10)
    assert (cost, count) == (22, 1), (cost, count)
    assert undo_result["cost"] == 22, undo_result
    assert undo_result["matrix"] == witness, (undo_result, witness)
    greedy_matrix, greedy_cost = brute.commit_first_greedy(undo, 10)
    assert greedy_cost == 24, greedy_cost
    print("6. undo needed                   : optimum 22, commit-first greedy 24")

    # Same story, larger gap: every row's locally cheapest ceil cannot all be
    # kept, and the greedy commitment paints itself into a corner.
    greedy_trap = [[5, 17, 8], [23, 14, 13], [2, 19, 29]]
    result = rounding.round_table(greedy_trap, 10)
    check_feasible(greedy_trap, 10, result)
    cost, count, witness = brute.brute_force(greedy_trap, 10)
    assert (cost, count) == (28, 1), (cost, count)
    assert result["matrix"] == witness
    greedy_matrix, greedy_cost = brute.commit_first_greedy(greedy_trap, 10)
    assert greedy_cost == 40, greedy_cost
    print("7. greedy trap                   : optimum 28, commit-first greedy 40")

    # Input must not be modified, and repeated calls must be stable.
    original = [row[:] for row in undo]
    rounding.round_table(undo, 10)
    assert undo == original, "input was modified"
    assert rounding.round_table(undo, 10) == undo_result, "result is not deterministic"

    # Single-cell and single-row/column shapes with valid divisibility.
    assert rounding.round_table([[7]], 7) == {"matrix": [[1]], "cost": 0}
    row = [[3, 3]]
    result = rounding.round_table(row, 3)
    check_feasible(row, 3, result)
    column = [[3], [3]]
    result = rounding.round_table(column, 3)
    check_feasible(column, 3, result)

    # Every result agrees with exhaustive enumeration on these scenarios.
    for A, q in (
        (scenario_one, 10),
        (integer, 10),
        (sparse, 10),
        (tie, 10),
        (undo, 10),
        (greedy_trap, 10),
        (zeros, 3),
    ):
        best, _, _ = brute.brute_force(A, q)
        assert rounding.round_table(A, q)["cost"] == best, (A, q, best)

    print("scenario tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
