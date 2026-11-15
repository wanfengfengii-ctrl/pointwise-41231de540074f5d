"""Randomised cross-check of the solver against exhaustive enumeration."""

from __future__ import annotations

import random
import sys

import brute
import rounding


def random_valid_table(rng, n, m, q):
    """Build a table whose row and column sums are all multiples of q.

    Start from random values, then repair each row and finally each column
    with small additions, so remainders stay varied and divisors really matter.
    """
    A = [[rng.randrange(0, 4 * q + 1) for _ in range(m)] for _ in range(n)]

    # Repair rows: add the deficit of each row to one of its cells.
    for i in range(n):
        deficit = (-sum(A[i])) % q
        A[i][rng.randrange(m)] += deficit

    # Repair columns: the total is now divisible by q, so the column deficits
    # sum to a multiple of q and can all be charged to row 0 without breaking
    # that row's divisibility.
    for j in range(m):
        deficit = (-sum(A[i][j] for i in range(n))) % q
        A[0][j] += deficit

    assert all(sum(row) % q == 0 for row in A)
    assert all(sum(A[i][j] for i in range(n)) % q == 0 for j in range(m))
    return A


def main(iterations=400, max_side=4, seed=2026):
    rng = random.Random(seed)
    checked = 0
    for _ in range(iterations):
        q = rng.choice([2, 3, 4, 5, 6, 7, 10, 13])
        n = rng.choice([1, 2, 3, max_side])
        m = rng.choice([1, 2, 3, max_side])
        if max(n, m) > max_side:
            continue
        A = random_valid_table(rng, n, m, q)
        best, count, _ = brute.brute_force(A, q)
        result = rounding.round_table(A, q)
        if result["cost"] != best:
            print("MISMATCH", A, q, "brute", best, "solver", result)
            return 1
        # Recompute the solver cost from its own matrix, independently.
        recomputed = sum(
            abs(q * result["matrix"][i][j] - A[i][j])
            for i in range(n)
            for j in range(m)
        )
        if recomputed != result["cost"]:
            print("SELF-COST MISMATCH", A, q, result)
            return 1
        # Feasibility: every cell floor/ceil, totals preserved.
        for i in range(n):
            for j in range(m):
                if result["matrix"][i][j] not in (
                    A[i][j] // q,
                    A[i][j] // q + 1,
                ):
                    print("CELL OUT OF RANGE", A, q, result)
                    return 1
        if [sum(row) for row in result["matrix"]] != [sum(row) // q for row in A]:
            print("ROW TOTAL MISMATCH", A, q, result)
            return 1
        if [
            sum(result["matrix"][i][j] for i in range(n)) for j in range(m)
        ] != [sum(A[i][j] for i in range(n)) // q for j in range(m)]:
            print("COLUMN TOTAL MISMATCH", A, q, result)
            return 1
        checked += 1
    print(f"cross-checked {checked} small tables against exhaustive enumeration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
