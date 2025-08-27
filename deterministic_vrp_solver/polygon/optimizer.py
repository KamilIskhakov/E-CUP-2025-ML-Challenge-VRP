import sqlite3
from typing import Dict, List, Tuple
import multiprocessing as mp
import polars as pl
import math

from .solver import PolygonTSPSolver


def _worker_solve_batch(args: Tuple[str, List[Dict]]):
    db_path, batch_rows = args
    conn = sqlite3.connect(db_path)
    try:
        solver = PolygonTSPSolver(conn)
        out = []
        for row in batch_rows:
            mp_id = row['MpId']
            order_ids = row.get('order_ids') or []
            if not order_ids:
                optimal_route = []
                total_cost = 0
            else:
                optimal_route, total_cost = solver.solve(order_ids)
            if not math.isfinite(total_cost) or total_cost >= 10**12:
                total_cost_int = 10**12
            else:
                total_cost_int = int(round(total_cost))
            out.append({
                'MpId': mp_id,
                'order_ids': order_ids,
                'optimal_route': optimal_route,
                'total_distance': total_cost_int,
                'total_cost': total_cost_int,
            })
        return out
    finally:
        conn.close()


def optimize_all_polygons_hybrid(optimized_polygon_order: pl.DataFrame,
                                 db_path: str,
                                 service_times: Dict[int, Dict[int, int]],
                                 max_workers: int = 1) -> pl.DataFrame:
    """Вычисляет стоимость TSP внутри каждого полигона.

    Возвращает DataFrame с колонками как минимум: MpId, order_ids, total_distance, total_cost.
    """
    rows: List[Dict] = []
    records: List[Dict] = [r for r in optimized_polygon_order.iter_rows(named=True)]

    if max_workers and max_workers > 1 and len(records) > 1:
                                              
        chunks: List[List[Dict]] = []
        bs = max(1, (len(records) + max_workers - 1) // max_workers)
        for i in range(0, len(records), bs):
            chunks.append(records[i:i+bs])
        with mp.get_context('spawn').Pool(processes=max_workers) as pool:
            for out in pool.imap_unordered(_worker_solve_batch, [(db_path, ch) for ch in chunks]):
                rows.extend(out)
    else:
                         
        rows.extend(_worker_solve_batch((db_path, records)))

    return pl.DataFrame(rows)


