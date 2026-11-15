"""Preparation step for the Pointwise CLI runtime.

Creates a small fixture directory used by ``demo.py`` and checks that the
standard-library-only environment can import the solver.  Nothing is installed
and nothing outside the repository is touched.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, ".pointwise-build")

sys.path.insert(0, ROOT)

import rounding  # noqa: E402


DEMO_TABLE = [
    [618, 522, 818, 417, 425],
    [417, 327, 815, 723, 718],
    [624, 718, 422, 216, 220],
    [322, 517, 823, 716, 522],
    [419, 816, 622, 828, 415],
]
DEMO_Q = 100

SMALL_CASES = [
    {"name": "specification example", "A": [[6, 4], [4, 6]], "q": 10},
    {"name": "integer cells", "A": [[10, 20, 30], [40, 50, 60]], "q": 10},
    {"name": "zero row and column", "A": [[0, 0, 0], [0, 10, 10], [0, 10, 10]], "q": 10},
]


def main():
    if sys.version_info < (3, 8):
        raise SystemExit("Python 3.8 or newer is required")

    for A, q in ((DEMO_TABLE, DEMO_Q),):
        assert all(sum(row) % q == 0 for row in A), "demo row totals not divisible"
        assert all(
            sum(A[i][j] for i in range(len(A))) % q == 0 for j in range(len(A[0]))
        ), "demo column totals not divisible"

    # A smoke check that the solver is importable and returns the documented
    # result for the specification example.
    example = rounding.round_table([[6, 4], [4, 6]], 10)
    if example != {"matrix": [[1, 0], [0, 1]], "cost": 16}:
        raise SystemExit(f"solver smoke check failed: {example}")

    os.makedirs(BUILD, exist_ok=True)
    fixture = {"demo": {"A": DEMO_TABLE, "q": DEMO_Q}, "small_cases": SMALL_CASES}
    path = os.path.join(BUILD, "fixtures.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=2)
        handle.write("\n")

    modules = sorted({"rounding.py"} & set(os.listdir(ROOT)))
    print(f"python {sys.version.split()[0]}; stdlib heapq available: {bool(__import__('heapq'))}")
    print(f"solver smoke check: {example}")
    print(f"prepared fixtures: {os.path.relpath(path, ROOT)}")
    print(f"solver module present: {modules}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
