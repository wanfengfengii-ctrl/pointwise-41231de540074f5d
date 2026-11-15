"""Verification step for the Pointwise CLI runtime.

Runs the full test battery and exits non-zero on any failure:

1. validation of every documented illegal input;
2. scenario cases (integer cells, zero rows/columns, ties, undo);
3. exhaustive brute-force cross-check on small tables;
4. an independent residual-network optimality certificate.

Only the standard library is used.  ``sys.path`` is set up here so the
commands work from any current directory.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_cases  # noqa: E402
import test_validation  # noqa: E402
import crosscheck  # noqa: E402
import optimality  # noqa: E402
import rounding  # noqa: E402


CERT_LOAD = int(os.environ.get("POINTWISE_CERT_LOAD", "120"))


def certify_benchmark_tables():
    """Certify every benchmark-shaped table the solver produces."""
    cases = []
    for q, quotient, remainder in (
        (10, 3, 0),
        (10, 3, 5),
        (10, 3, 9),
        (30, 1, 29),
        (997, 1, 996),
    ):
        diagonal = (-29 * remainder) % q
        cases.append(
            (
                [
                    [
                        q * quotient + (diagonal if i == j else remainder)
                        for j in range(30)
                    ]
                    for i in range(30)
                ],
                q,
            )
        )

    checked = 0
    for A, q in cases:
        result = rounding.round_table(A, q)
        ok, message = optimality.certify(A, q, result)
        if not ok:
            raise AssertionError(f"certificate failed for q={q}: {message}")
        checked += 1
    print(f"certified optimality of {checked} dense 30x30 tables")

    # A larger random battery, certified independently of the solver's own cost.
    import random

    rng = random.Random(20261115)
    for _ in range(CERT_LOAD):
        n = rng.randint(2, 8)
        m = rng.randint(2, 8)
        q = rng.choice([3, 4, 5, 7, 10, 13])
        A = [[rng.randrange(0, 4 * q + 1) for _ in range(m)] for _ in range(n)]
        for i in range(n):
            A[i][rng.randrange(m)] += (-sum(A[i])) % q
        for j in range(m):
            A[0][j] += (-sum(A[i][j] for i in range(n))) % q
        if not all(sum(row) % q == 0 for row in A):
            continue
        if not all(sum(A[i][j] for i in range(n)) % q == 0 for j in range(m)):
            continue
        result = rounding.round_table(A, q)
        ok, message = optimality.certify(A, q, result)
        if not ok:
            raise AssertionError(f"certificate failed: {A} q={q}: {message}")
        checked += 1
    print(f"certified optimality of {checked} tables in total")


def main():
    print("== validation ==")
    assert test_validation.main() == 0
    print("== scenarios ==")
    assert test_cases.main() == 0
    print("== brute-force cross-check ==")
    assert crosscheck.main(iterations=400) == 0
    print("== optimality certificates ==")
    certify_benchmark_tables()
    print("verify: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
