import sqlite3
from typing import Dict
import polars as pl
import math

from .solver import PolygonTSPSolver


def optimize_all_polygons_hybrid(optimized_polygon_order: pl.DataFrame,
                                 db_path: str,
                                 service_times: Dict[int, Dict[int, int]],
                                 max_workers: int = 1) -> pl.DataFrame:
    """Вычисляет стоимость TSP внутри каждого полигона.

    Возвращает DataFrame с колонками как минимум: MpId, order_ids, total_distance, total_cost.
    """
    conn = sqlite3.connect(db_path)
    try:
        solver = PolygonTSPSolver(conn)
        rows = []
        for row in optimized_polygon_order.iter_rows(named=True):
            mp_id = row['MpId']
            order_ids = row.get('order_ids') or []
            if not order_ids:
                total_cost = 0
            else:
                _, total_cost = solver.solve(order_ids)
            # Безопасное целочисленное представление (все расстояния целые):
            if not math.isfinite(total_cost) or total_cost >= 10**12:
                total_cost_int = 10**12
            else:
                total_cost_int = int(round(total_cost))
            rows.append({
                'MpId': mp_id,
                'order_ids': order_ids,
                'total_distance': total_cost_int,
                'total_cost': total_cost_int,
            })
        return pl.DataFrame(rows)
    finally:
        conn.close()


