"""Validation tests: every documented invalid input must raise ValueError."""

from __future__ import annotations

import sys

import rounding

VALID = [[6, 4], [4, 6]]


def expect_invalid(case, A, q):
    try:
        rounding.round_table(A, q)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {case}")


def main():
    # Shape problems.
    expect_invalid("empty table", [], 1)
    expect_invalid("too many rows", [[0]] * 31, 1)
    expect_invalid("too many columns", [[0] * 31], 1)
    expect_invalid("jagged", [[1, 2], [3]], 3)
    expect_invalid("not a list", "12", 1)
    expect_invalid("row not a list", [1, 2], 1)
    expect_invalid("tuple input", ((6, 4), (4, 6)), 10)
    expect_invalid("empty row", [[]], 1)
    expect_invalid("nested cell", [[[1], [1]]], 1)

    # Entry problems.
    expect_invalid("negative entry", [[-1, 1], [1, 1]], 1)
    expect_invalid("float entry", [[1.0, 1.0], [1.0, 1.0]], 2)
    expect_invalid("string entry", [["1", "1"], ["1", "1"]], 2)
    expect_invalid("None entry", [[None, 1], [1, 1]], 1)
    expect_invalid("bool entry True", [[True, 1], [1, 1]], 1)
    expect_invalid("bool entry False", [[False, 0], [0, 0]], 1)

    # Divider problems.
    for q in (0, -10):
        expect_invalid(f"q={q}", VALID, q)
    for q in (True, False):
        expect_invalid(f"q={q} (bool)", VALID, q)
    for q in (10.0, 10.5, "10", None):
        expect_invalid(f"q={q!r}", VALID, q)

    # Divider that cannot divide the totals.
    expect_invalid("row not divisible", [[1, 1], [1, 1]], 3)
    expect_invalid("column not divisible", [[1, 2], [1, 1]], 3)
    expect_invalid("q larger than totals", [[5, 5], [5, 5]], 20)
    expect_invalid("nonzero total, zero cell sums", [[1, 0], [0, 0]], 2)
    expect_invalid("zero table with q zero-ish", [[0, 1], [1, 0]], 4)

    # Inputs must not be modified by a failed call either.
    data = [[1, 1], [1, 1]]
    try:
        rounding.round_table(data, 3)
    except ValueError:
        pass
    assert data == [[1, 1], [1, 1]], "input mutated on invalid call"

    # Boundary-valid sizes still work.
    assert rounding.round_table([[0]], 7) == {"matrix": [[0]], "cost": 0}
    assert rounding.round_table([[7]], 7) == {"matrix": [[1]], "cost": 0}
    tall = [[3], [3]]
    assert rounding.round_table(tall, 3) == {"matrix": [[1], [1]], "cost": 0}
    wide = [[3, 3]]
    assert rounding.round_table(wide, 3) == {"matrix": [[1, 1]], "cost": 0}
    big = [[0] * 30 for _ in range(30)]
    assert rounding.round_table(big, 5) == {"matrix": [[0] * 30 for _ in range(30)], "cost": 0}

    print("validation: all invalid inputs rejected, boundaries accepted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
