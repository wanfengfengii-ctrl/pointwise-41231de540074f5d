"""Optimal two-way controlled rounding for grouped cross-tabulation counts.

``round_table(A, q)`` converts a table of non-negative integer counts into a
table of integers in which every cell is either ``A[i][j] // q`` or
``A[i][j] // q + 1``, every row total and every column total equals the
original total divided by ``q``, and the total absolute error

    sum(abs(q * matrix[i][j] - A[i][j]))

is as small as possible over all such tables.

The task is not independent rounding: choosing one cell's value constrains
all others in its row and column.  Writing ``rem = A[i][j] % q``, the floor
choice contributes ``rem`` to the error and the ceiling choice contributes
``q - rem``, so the extra cost of a ceiling is ``delta = q - 2 * rem`` and

    total error = sum(rem over all cells) + sum(delta over ceiling cells).

Exactly ``row_need[i] = sum(rem in row i) / q`` ceilings are needed in row i
(and likewise per column), because the row total is divisible by q.  Picking
which cells become ceilings is therefore a transportation problem:

    source -> row i        capacity row_need[i], cost 0
    row i  -> column j     capacity 1,          cost delta[i][j]
    column j -> sink       capacity col_need[j], cost 0

An integral minimum-cost flow of value ``sum(row_need)`` gives the optimal
rounded table.  The flow is integral because the constraint matrix of a
bipartite transportation network is totally unimodular; the fractional point
``x[i][j] = rem / q`` shows the network is feasible, so valid input always
has a solution.

The flow is solved with successive shortest augmenting paths and Johnson
potentials.  Residual reverse edges let a later augmentation cancel an
earlier cell choice, which is exactly why the result is globally optimal
instead of the outcome of independent rounding or a commit-first greedy.
Cells with ``rem == 0`` (already a multiple of q) are kept at their floor and
are not offered as ceiling candidates.

Only the Python standard library is used.  The input is never modified.
"""

from __future__ import annotations

import heapq

MAX_ROWS = 30
MAX_COLS = 30

__all__ = ["round_table"]


def _is_int(value: object) -> bool:
    """True for integers, excluding bool (which is an int subclass)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(A: object, q: object) -> tuple[list[list[int]], int]:
    """Validate input and return a private, normalised copy of the table."""
    if not _is_int(q) or q <= 0:
        raise ValueError("q must be a positive integer (bool is not accepted)")

    if not isinstance(A, list):
        raise ValueError("A must be a rectangular list of rows")
    n = len(A)
    if n < 1 or n > MAX_ROWS:
        raise ValueError(f"A must have between 1 and {MAX_ROWS} rows")

    rows: list[list[int]] = []
    m: int | None = None
    for row in A:
        if not isinstance(row, list):
            raise ValueError("each row of A must be a list")
        if m is None:
            m = len(row)
            if m < 1 or m > MAX_COLS:
                raise ValueError(f"A must have between 1 and {MAX_COLS} columns")
        elif len(row) != m:
            raise ValueError("A must be rectangular")
        checked: list[int] = []
        for value in row:
            if not _is_int(value) or value < 0:
                raise ValueError(
                    "A entries must be non-negative integers (bool is not accepted)"
                )
            checked.append(value)
        rows.append(checked)

    assert m is not None
    for row in rows:
        if sum(row) % q != 0:
            raise ValueError("every row sum of A must be divisible by q")
    for j in range(m):
        if sum(rows[i][j] for i in range(n)) % q != 0:
            raise ValueError("every column sum of A must be divisible by q")

    return rows, q


def _add_edge(graph, u, v, capacity, cost):
    """Add a directed edge plus its residual reverse; return the forward edge.

    Edge layout: [to, index of reverse edge, residual capacity, cost].
    """
    forward = [v, len(graph[v]), capacity, cost]
    backward = [u, len(graph[u]), 0, -cost]
    graph[u].append(forward)
    graph[v].append(backward)
    return forward


def round_table(A: list[list[int]], q: int) -> dict:
    """Return ``{"matrix": rounded integer matrix, "cost": absolute error}``.

    ``A`` is a 1..30 by 1..30 rectangular list of non-negative integers whose
    row and column sums are all divisible by the positive integer ``q``.  Any
    other input raises ``ValueError``.  The input object is not modified.
    """
    values, q = _validate(A, q)
    n = len(values)
    m = len(values[0])

    floors = [[value // q for value in row] for row in values]
    rems = [[value % q for value in row] for row in values]

    row_need = [sum(rems[i]) // q for i in range(n)]
    col_need = [sum(rems[i][j] for i in range(n)) // q for j in range(m)]
    ceilings = sum(row_need)
    if ceilings != sum(col_need):
        raise RuntimeError("inconsistent row and column requirements")

    # Cells divisible by q contribute no error and cannot become ceilings.
    base_cost = sum(sum(row) for row in rems)

    if ceilings == 0:
        return {"matrix": [row[:] for row in floors], "cost": base_cost}

    # Shifting every cell cost by a constant does not change which flow is
    # optimal: every feasible solution contains exactly `ceilings` cell edges.
    deltas = [[q - 2 * rems[i][j] for j in range(m)] for i in range(n)]
    shift = min(
        deltas[i][j] for i in range(n) for j in range(m) if rems[i][j] > 0
    )

    source = 0
    row_base = 1
    col_base = row_base + n
    sink = col_base + m
    node_count = sink + 1
    graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    for i, need in enumerate(row_need):
        if need:
            _add_edge(graph, source, row_base + i, need, 0)
    for j, need in enumerate(col_need):
        if need:
            _add_edge(graph, col_base + j, sink, need, 0)

    cell_edges: list[list[list[int] | None]] = [[None] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if rems[i][j] == 0:
                continue  # floor == ceil, no decision to make
            cell_edges[i][j] = _add_edge(
                graph,
                row_base + i,
                col_base + j,
                1,
                deltas[i][j] - shift,
            )

    # Successive shortest augmenting paths (Dijkstra with Johnson potentials).
    inf = float("inf")
    potential = [0] * node_count
    flow = 0
    while flow < ceilings:
        dist = [inf] * node_count
        prev_node = [-1] * node_count
        prev_edge = [-1] * node_count
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            current, node = heapq.heappop(heap)
            if current != dist[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge[2] <= 0:
                    continue
                target = edge[0]
                candidate = current + edge[3] + potential[node] - potential[target]
                if candidate < dist[target]:
                    dist[target] = candidate
                    prev_node[target] = node
                    prev_edge[target] = edge_index
                    heapq.heappush(heap, (candidate, target))

        if dist[sink] == inf:
            # Cannot happen for valid input (a feasible flow always exists);
            # guards against an internal bug instead of returning a bad table.
            raise RuntimeError("no feasible controlled rounding found")

        for node in range(node_count):
            if dist[node] != inf:
                potential[node] += dist[node]

        # Unit capacities on cell edges mean one unit per augmentation, but
        # compute the bottleneck to stay correct if capacities ever change.
        amount = ceilings - flow
        node = sink
        while node != source:
            edge = graph[prev_node[node]][prev_edge[node]]
            if edge[2] < amount:
                amount = edge[2]
            node = prev_node[node]
        node = sink
        while node != source:
            parent = prev_node[node]
            edge = graph[parent][prev_edge[node]]
            edge[2] -= amount
            graph[node][edge[1]][2] += amount
            node = parent
        flow += amount

    matrix = [row[:] for row in floors]
    for i in range(n):
        for j in range(m):
            edge = cell_edges[i][j]
            if edge is not None and edge[2] == 0:
                matrix[i][j] += 1

    cost = sum(
        abs(q * matrix[i][j] - values[i][j]) for i in range(n) for j in range(m)
    )
    return {"matrix": matrix, "cost": cost}
