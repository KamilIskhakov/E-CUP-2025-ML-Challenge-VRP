from typing import Dict, List
import sqlite3
import polars as pl


def build_polygon_info(optimized_polygons: pl.DataFrame, polygon_ports: Dict[int, List[int]], conn: sqlite3.Connection) -> Dict[int, Dict]:
    polygon_info_for_decomposition: Dict[int, Dict] = {}
    for row in optimized_polygons.iter_rows(named=True):
        polygon_id = row['MpId']
        if polygon_id == 0:
            continue
        ports = polygon_ports.get(polygon_id, [])
        cost = int(row['total_cost']) if row['total_cost'] is not None else 0
        portal_id = int(row.get('portal_id', 0))
        portal_distances: Dict[int, int] = {}
        for port_id in ports:
            cursor = conn.cursor()
            cursor.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (port_id, portal_id))
            result = cursor.fetchone()
            portal_distances[port_id] = int(result[0]) if result and result[0] and result[0] > 0 else 0
        polygon_info_for_decomposition[polygon_id] = {
            'ports': ports,
            'cost': cost,
            'portal_distances': portal_distances,
            'portal_id': portal_id,
        }
    return polygon_info_for_decomposition


