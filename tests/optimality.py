"""Independent optimality certificate for any produced matrix.

Instead of trusting the solver's own cost, this module rebuilds the flow
network from ``A`` and the returned matrix, and checks the classical
min-cost-flow optimality condition: the residual network (after fixing the
flow) must admit node potentials whose reduced costs are non-negative on
every residual arc.  Existence of such potentials is equivalent to the flow
being of minimum cost for its value, i.e. no negative-cost cycle remains.

The check is independent of how the matrix was produced.
"""

from __future__ import annotations


def certify(A, q, result):
    """Return ``(ok, message)`` certifying ``result`` is an optimal rounding."""
    n = len(A)
    m = len(A[0])
    matrix = result.get("matrix")
    if matrix is None or len(matrix) != n or any(len(r) != m for r in matrix):
        return False, "matrix shape mismatch"

    floors = [[v // q for v in row] for row in A]
    row_need = [sum(A[i]) // q - sum(floors[i]) for i in range(n)]
    col_need = [
        sum(A[i][j] for i in range(n)) // q
        - sum(floors[i][j] for i in range(n))
        for j in range(m)
    ]

    # 1. Each cell is floor or ceil, and totals are preserved.
    for i in range(n):
        for j in range(m):
            if matrix[i][j] not in (floors[i][j], floors[i][j] + 1):
                return False, f"cell ({i},{j}) is not floor/ceil"
        if sum(matrix[i]) != sum(A[i]) // q:
            return False, f"row {i} total mismatch"
    for j in range(m):
        if sum(matrix[i][j] for i in range(n)) != sum(A[i][j] for i in range(n)) // q:
            return False, f"column {j} total mismatch"

    # 2. Rebuild the residue network.  Use the same non-negative cost shift as
    # the solver; every feasible solution uses exactly ``total`` cell arcs, so
    # shifting does not change which solution is cheapest.
    deltas = [[q - 2 * (A[i][j] % q) for j in range(m)] for i in range(n)]
    shift = min(min(row) for row in deltas)
    total = sum(row_need)

    source = 0
    row_node = [1 + i for i in range(n)]
    col_node = [1 + n + j for j in range(m)]
    sink = 1 + n + m
    nodes = sink + 1

    # Edges: (u, v, residual_capacity, cost).
    residual = []

    used_ceils = [[matrix[i][j] - floors[i][j] for j in range(m)] for i in range(n)]
    row_used = [sum(row) for row in used_ceils]
    col_used = [sum(used_ceils[i][j] for i in range(n)) for j in range(m)]

    for i in range(n):
        residual.append((source, row_node[i], row_need[i] - row_used[i], 0))
        residual.append((row_node[i], source, row_used[i], 0))
    for j in range(m):
        residual.append((col_node[j], sink, col_need[j] - col_used[j], 0))
        residual.append((sink, col_node[j], col_used[j], 0))
    for i in range(n):
        for j in range(m):
            cost = deltas[i][j] - shift
            if used_ceils[i][j]:
                residual.append((col_node[j], row_node[i], 1, -cost))
            else:
                residual.append((row_node[i], col_node[j], 1, cost))

    # 3. Bellman-Ford potentials (all nodes start at 0, so a negative cycle,
    # if any, is found).  Nodes unreachable from the flow's support still get
    # valid potentials this way.
    dist = [0] * nodes
    for _ in range(nodes):
        changed = False
        for u, v, cap, cost in residual:
            if cap <= 0:
                continue
            if dist[v] > dist[u] + cost:
                dist[v] = dist[u] + cost
                changed = True
        if not changed:
            break
    else:
        return False, "negative-cost cycle in residual graph (not optimal)"

    # 4. Non-negative reduced cost on every residual arc.
    for u, v, cap, cost in residual:
        if cap > 0 and cost + dist[u] - dist[v] < 0:
            return False, "dual infeasibility in residual graph (not optimal)"

    # 5. Flow value must be complete.
    if sum(row_used) != total or sum(col_used) != total:
        return False, "incomplete flow"
    return True, "optimality certificate verified"
