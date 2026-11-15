"""Measure wall-clock time and peak memory for 30x30 dense tables.

Every case runs in a fresh child process, so peak RSS reflects only the solver
and its own data.  A control child builds the same input but does not solve it;
the difference between the two peaks is the incremental memory the solver
needs.  Only the standard library is used.

Each dense case is a 30x30 table whose cells are ``q * quotient + r_main``,
except the diagonal, whose remainder keeps every row and column sum divisible
by ``q``.  Since a row of 30 cells can need at most ``30*(q-1)//q`` ceilings,
870 is the largest possible number of ceiling picks for a 30x30 table, and the
``q=30`` and ``q=997`` cases attain it.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
import tracemalloc

import rounding


CASES = [
    # (label, q, quotient, r_main)
    ("exact multiples", 10, 3, 0),
    ("moderate remainder", 10, 3, 5),
    ("high remainder", 10, 3, 9),
    ("max picks, q=30", 30, 1, 29),
    ("max picks, q=997", 997, 1, 996),
]


def build_case(q, quotient, r_main):
    diagonal = (-29 * r_main) % q
    A = [
        [q * quotient + (diagonal if i == j else r_main) for j in range(30)]
        for i in range(30)
    ]
    assert all(sum(row) % q == 0 for row in A), "row totals not divisible"
    assert all(
        sum(A[i][j] for i in range(30)) % q == 0 for j in range(30)
    ), "column totals not divisible"
    return A


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def run_child(mode, payload):
    label, q, quotient, r_main, repeats = payload
    A = build_case(q, quotient, r_main)
    if mode == "control":
        # Touch the input the same way the solver's validation does, then stop.
        total = sum(sum(row) for row in A)
        print(json.dumps({"label": label, "peak_rss_mb": rss_mb(), "checksum": total}))
        return

    best = None
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = rounding.round_table(A, q)
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed

    # Peak traced allocation during one more solve, separate from the timing
    # loop so tracemalloc overhead does not distort the wall-clock numbers.
    tracemalloc.start()
    rounding.round_table(A, q)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert [sum(row) for row in result["matrix"]] == [sum(row) // q for row in A]
    assert [
        sum(result["matrix"][i][j] for i in range(30)) for j in range(30)
    ] == [sum(A[i][j] for i in range(30)) // q for j in range(30)]
    recomputed = sum(
        abs(q * result["matrix"][i][j] - A[i][j]) for i in range(30) for j in range(30)
    )
    assert recomputed == result["cost"]

    picks = sum(
        1
        for i in range(30)
        for j in range(30)
        if result["matrix"][i][j] != A[i][j] // q
    )
    print(
        json.dumps(
            {
                "label": label,
                "q": q,
                "ceil_picks": picks,
                "seconds": round(best, 4),
                "peak_rss_mb": rss_mb(),
                "traced_peak_kb": round(traced_peak / 1024, 1),
            }
        )
    )


def spawn(mode, payload, env):
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), f"--{mode}", *map(str, payload)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": root}
    rows = []
    for label, q, quotient, r_main in CASES:
        repeats = 20 if q <= 30 else 5
        payload = (label, q, quotient, r_main, repeats)
        control = spawn("control", payload, env)
        solved = spawn("solve", payload, env)
        solved["control_rss_mb"] = control["peak_rss_mb"]
        solved["delta_rss_mb"] = round(
            solved["peak_rss_mb"] - control["peak_rss_mb"], 1
        )
        rows.append(solved)

    print("30x30 dense benchmark: best of N runs, fresh process per case")
    print(
        f"{'case':<20} {'q':>5} {'ceil picks':>10} {'seconds':>9} "
        f"{'peak RSS MB':>12} {'solver delta MB':>16} {'traced peak KB':>15}"
    )
    for row in rows:
        print(
            f"{row['label']:<20} {row['q']:>5} {row['ceil_picks']:>10} "
            f"{row['seconds']:>9.4f} {row['peak_rss_mb']:>12.1f} "
            f"{row['delta_rss_mb']:>16.1f} {row['traced_peak_kb']:>15.1f}"
        )
    print("(peak RSS includes the ~6.5 MB interpreter baseline)")
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--solve", "--control"):
        mode = sys.argv[1][2:]
        label, q, quotient, r_main, repeats = sys.argv[2:]
        run_child(
            mode, (label, int(q), int(quotient), int(r_main), int(repeats))
        )
    else:
        main()
