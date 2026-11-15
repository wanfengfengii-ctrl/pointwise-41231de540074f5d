"""Independent brute-force reference for small tables.

Used only by the test-suite to cross-check ``rounding.round_table``.  It
enumerates every feasible set of ceiling choices row by row, so its result is
completely independent of the min-cost-flow solver.

It also intentionally offers a "commit now, never revise" greedy so tests can
show that locally sensible commitments can be strictly worse than the optimum.
"""

from __future__ import annotations

import itertools


def _prepare(A, q):
    n = len(A)
    m = len(A[0])
    floors = [[value // q for value in row] for row in A]
    need_rows = [sum(A[i]) // q - sum(floors[i]) for i in range(n)]
    need_cols = [
        sum(A[i][j] for i in range(n)) // q - sum(floors[i][j] for i in range(n))
        for j in range(m)
    ]
    base = sum(value % q for row in A for value in row)
    deltas = [[q - 2 * (value % q) for value in row] for row in A]
    return n, m, floors, need_rows, need_cols, base, deltas


def brute_force(A, q):
    """Return ``(best_cost, number_of_optimal_matrices, one_optimal_matrix)``.

    Only practical for small tables (the test-suite uses at most 4x4).  Returns
    ``(None, 0, None)`` if the instance has no feasible rounding, which cannot
    happen for valid input.
    """
    n, m, floors, need_rows, need_cols, base, deltas = _prepare(A, q)
    if sum(need_rows) != sum(need_cols):
        return None, 0, None

    best = [None]
    count = [0]
    witness = [None]
    chosen = [[0] * m for _ in range(n)]
    col_count = [0] * m

    def recurse(row, delta):
        if row == n:
            if col_count == need_cols:
                cost = base + delta
                if best[0] is None or cost < best[0]:
                    best[0] = cost
                    count[0] = 1
                    witness[0] = [
                        [floors[i][j] + chosen[i][j] for j in range(m)]
                        for i in range(n)
                    ]
                elif cost == best[0]:
                    count[0] += 1
            return
        available = [j for j in range(m) if col_count[j] < need_cols[j]]
        if len(available) < need_rows[row]:
            return
        for combo in itertools.combinations(available, need_rows[row]):
            for j in combo:
                col_count[j] += 1
                chosen[row][j] = 1
            recurse(row + 1, delta + sum(deltas[row][j] for j in combo))
            for j in combo:
                col_count[j] -= 1
                chosen[row][j] = 0

    recurse(0, 0)
    return best[0], count[0], witness[0]


def commit_first_greedy(A, q):
    """Row-by-row greedy that never revises an earlier choice.

    For each row, assign the required ceilings to the cells with the largest
    remainders among columns that still have quota.  Returns the matrix and
    its cost, or ``(None, None)`` if the commitment order paints itself into a
    corner.  This is deliberately *not* optimal; it exists for comparison.
    """
    n, m, floors, need_rows, need_cols, _, _ = _prepare(A, q)
    matrix = [row[:] for row in floors]
    remaining = need_cols[:]
    for i in range(n):
        candidates = sorted(
            (j for j in range(m) if remaining[j] > 0),
            key=lambda j: (-(A[i][j] % q), j),
        )
        if len(candidates) < need_rows[i]:
            return None, None
        for j in candidates[: need_rows[i]]:
            matrix[i][j] += 1
            remaining[j] -= 1
    if any(remaining):
        return None, None
    cost = sum(
        abs(q * matrix[i][j] - A[i][j]) for i in range(n) for j in range(m)
    )
    return matrix, cost
