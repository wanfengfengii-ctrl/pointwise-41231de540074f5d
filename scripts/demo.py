"""Real-conversion demonstration for the Pointwise CLI runtime.

Reads the fixtures prepared by ``prepare.py``, rounds a real 5x5 grouped
cross-tabulation with q=100, and prints how the two-way totals are preserved.
Follows the aggregation rule used downstream: every cell may only become its
floor or ceiling, row and column totals stay exact after dividing by q, and
the total absolute error is minimal.

Exits non-zero if anything about the produced table fails verification.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import rounding  # noqa: E402
import optimality  # noqa: E402

FIXTURES = os.path.join(ROOT, ".pointwise-build", "fixtures.json")


def independent_rounding(A, q):
    """Round each cell on its own; used to show why totals break."""
    return [
        [value // q + (1 if 2 * (value % q) >= q else 0) for value in row] for row in A
    ]


def cost_of(matrix, A, q):
    return sum(
        abs(q * matrix[i][j] - A[i][j])
        for i in range(len(A))
        for j in range(len(A[0]))
    )


def print_table(title, matrix, A, q):
    print(title)
    header = "      " + "".join(f"  c{j:<7}" for j in range(len(matrix[0]))) + "  |  sum"
    print(header)
    for i, row in enumerate(matrix):
        cells = "".join(f"  {value:>7}" for value in row)
        print(f"  r{i}  {cells}  |  {sum(row):>3}  (was {sum(A[i]):>4})")
    print("  sum" + "".join(f"  {sum(matrix[i][j] for i in range(len(matrix))):>7}" for j in range(len(matrix[0]))))


def main():
    if not os.path.exists(FIXTURES):
        raise SystemExit("fixtures missing; run the setup step first")

    with open(FIXTURES, "r", encoding="utf-8") as handle:
        fixture = json.load(handle)

    A = fixture["demo"]["A"]
    q = fixture["demo"]["q"]
    original = [row[:] for row in A]

    print(f"input: {len(A)}x{len(A[0])} counts, divisor q={q}, total={sum(map(sum, A))}")
    result = rounding.round_table(A, q)
    matrix = result["matrix"]

    print_table("rounded counts:", matrix, A, q)
    print()

    row_ok = [sum(row) for row in matrix] == [sum(row) // q for row in A]
    col_ok = [sum(matrix[i][j] for i in range(len(A))) for j in range(len(A[0]))] == [
        sum(A[i][j] for i in range(len(A))) // q for j in range(len(A[0]))
    ]
    print(f"row totals preserved: {row_ok}")
    print(f"column totals preserved: {col_ok}")
    print(f"total absolute error: {result['cost']}  (sum of cell remainders: {sum(v % q for row in A for v in row)})")

    comparison = independent_rounding(A, q)
    ind_row_sums = [sum(row) for row in comparison]
    ind_col_sums = [
        sum(comparison[i][j] for i in range(len(A))) for j in range(len(A[0]))
    ]
    print()
    print("independent per-cell rounding, for contrast:")
    print(f"  row totals {ind_row_sums} vs required {[sum(row) // q for row in A]}")
    print(f"  column totals {ind_col_sums} vs required "
          f"{[sum(A[i][j] for i in range(len(A))) // q for j in range(len(A[0]))]}")
    print(f"  error would be {cost_of(comparison, A, q)} (sum of rounded cells), "
          f"but the table violates the two-way totals")

    ok, message = optimality.certify(A, q, result)
    if not ok:
        raise SystemExit(f"demo produced an invalid table: {message}")
    if A != original:
        raise SystemExit("demo mutated its input")

    out_dir = os.path.join(ROOT, ".pointwise-build")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo-output.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"input": original, "q": q, "result": result}, handle, indent=2)
        handle.write("\n")

    print()
    print(f"optimality certificate: {message}")
    print(f"wrote {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
