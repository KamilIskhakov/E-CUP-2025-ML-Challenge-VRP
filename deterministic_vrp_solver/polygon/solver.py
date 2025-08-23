import sqlite3
from typing import List, Tuple, Dict
import numpy as np
from functools import lru_cache


class DistanceProvider:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._cache: Dict[Tuple[int, int], int] = {}

    @lru_cache(maxsize=200000)
    def get_distance(self, from_id: int, to_id: int) -> float:
        cur = self.conn.cursor()
        cur.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (from_id, to_id))
        row = cur.fetchone()
        return float(row[0]) if row and row[0] and row[0] > 0 else float('inf')


class PolygonTSPSolver:
    def __init__(self, conn: sqlite3.Connection, max_exact_size: int = 15):
        self.conn = conn
        self.dist = DistanceProvider(conn)
        self.max_exact_size = max_exact_size

    def solve(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], float]:
        n = len(order_ids)
        if n <= 1:
            return order_ids, 0
        if n == 2:
            d = self.dist.get_distance(order_ids[0], order_ids[1])
            return order_ids, d
        if n > self.max_exact_size:
            return self._solve_heuristic(order_ids, start_id)
        return self._solve_exact(order_ids, start_id)

    def _solve_exact(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], float]:
        n = len(order_ids)
        if start_id is None:
            start_id = 0
        dist_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i][j] = self.dist.get_distance(order_ids[i], order_ids[j])

        memo: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}

        def solve_dp(mask: int, pos: int) -> Tuple[int, List[int]]:
            if mask == (1 << n) - 1:
                return dist_matrix[pos][start_id], [pos]
            key = (mask, pos)
            if key in memo:
                return memo[key]
            best_cost: float = float('inf')
            best_path: List[int] = []
            for nxt in range(n):
                if mask & (1 << nxt) == 0:
                    cost, path = solve_dp(mask | (1 << nxt), nxt)
                    total = dist_matrix[pos][nxt] + cost
                    if total < best_cost:
                        best_cost = total
                        best_path = [pos] + path
            memo[key] = (best_cost, best_path)
            return memo[key]

        total_cost, path_idx = solve_dp(1 << start_id, start_id)
        return [order_ids[i] for i in path_idx], total_cost

    def _solve_heuristic(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], float]:
        if start_id is None:
            start_id = order_ids[0]
        unvisited = set(order_ids)
        current = start_id
        path = [current]
        unvisited.remove(current)
        total_cost: float = 0.0
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.dist.get_distance(current, x))
            total_cost += self.dist.get_distance(current, nearest)
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        total_cost += self.dist.get_distance(path[-1], path[0])
        path, total_cost = self._two_opt(path, total_cost)
        return path, total_cost

    def _two_opt(self, path: List[int], current_cost: float) -> Tuple[List[int], float]:
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    a, b = path[i - 1], path[i]
                    c, d = path[j], path[j + 1]
                    old = self.dist.get_distance(a, b) + self.dist.get_distance(c, d)
                    new = self.dist.get_distance(a, c) + self.dist.get_distance(b, d)
                    if new < old:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        current_cost -= (old - new)
                        improved = True
        return path, current_cost


