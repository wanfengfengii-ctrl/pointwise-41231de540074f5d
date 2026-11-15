"""Small-matrix comparison: exhaustive enumeration vs the solver.

Prints one row per case: the table, q, the brute-force optimum, the solver's
cost, the number of distinct optimal matrices, and a commit-first greedy cost
where one exists.  Used to document that the flow solver matches an
independent exhaustive search and that greedy commitment does not.
"""

from __future__ import annotations

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import brute  # noqa: E402
import rounding  # noqa: E402


NAMED = [
    ("spec example", [[6, 4], [4, 6]], 10),
    ("integer cells", [[10, 20], [30, 40]], 10),
    ("zero row/col", [[0, 0, 0], [0, 10, 10], [0, 10, 10]], 10),
    ("tie", [[5, 5], [5, 5]], 10),
    ("undo needed", [[10, 11, 19], [15, 15, 10], [15, 14, 11]], 10),
    ("greedy trap", [[5, 17, 8], [23, 14, 13], [2, 19, 29]], 10),
]


def random_case(rng):
    q = rng.choice([3, 4, 5, 6, 10])
    n = rng.choice([2, 3, 4])
    m = rng.choice([2, 3, 4])
    A = [[rng.randrange(0, 3 * q + 1) for _ in range(m)] for _ in range(n)]
    for i in range(n):
        A[i][rng.randrange(m)] += (-sum(A[i])) % q
    for j in range(m):
        A[0][j] += (-sum(A[i][j] for i in range(n))) % q
    return A, q


def fmt(A):
    return "[" + ",".join("[" + ",".join(map(str, row)) + "]" for row in A) + "]"


def main():
    rng = random.Random(4242)
    cases = list(NAMED)
    # Add several random small cases so the comparison is not cherry-picked.
    seen = 0
    while seen < 6:
        A, q = random_case(rng)
        if len(A) * len(A[0]) > 12:
            continue
        cases.append((f"random {seen + 1}", A, q))
        seen += 1

    print(f"{'case':<14} {'A':<38} {'q':>3} {'brute':>6} {'solver':>7} "
          f"{'#opt':>5} {'greedy':>7} {'match':>6}")
    failures = 0
    for name, A, q in cases:
        best, count, _ = brute.brute_force(A, q)
        result = rounding.round_table(A, q)
        _, greedy_cost = brute.commit_first_greedy(A, q)
        match = best == result["cost"]
        if not match:
            failures += 1
        print(
            f"{name:<14} {fmt(A):<38} {q:>3} {best:>6} {result['cost']:>7} "
            f"{count:>5} {str(greedy_cost):>7} {str(match):>6}"
        )
    print()
    print("match = solver cost equals exhaustive-enumeration optimum")
    print("greedy = commit-first row-wise greedy cost (None = infeasible commitment)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
