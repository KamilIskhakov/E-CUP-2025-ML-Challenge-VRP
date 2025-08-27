import math
from typing import Dict, List, Tuple

import polars as pl


def _polygon_centroid(row: dict) -> Tuple[float, float]:
    lats = row.get('Lat')
    longs = row.get('Long')
    if isinstance(lats, list) and isinstance(longs, list) and lats and longs and len(lats) == len(longs):
        return float(sum(lats) / len(lats)), float(sum(longs) / len(longs))
                                                                                             
    try:
                                                                                                               
                                                                                                          
        return float(row.get('centroid_lat', 0.0)), float(row.get('centroid_lon', 0.0))
    except Exception:
        return 0.0, 0.0


def split_into_sectors_by_angle(
    optimized_polygons: pl.DataFrame,
    orders_df: pl.DataFrame,
    warehouse_lat: float,
    warehouse_lon: float,
    target_sectors: int = 32,
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, int]]:
    """Разбивает полигоны на сектора по углу относительно склада, балансируя по числу заказов.

    Возвращает:
    - polygon_to_sector: MpId -> sector_id
    - sector_to_polygons: sector_id -> [MpId]
    - sector_orders: sector_id -> суммарное число заказов
    """
                                             
    orders_cols = set(orders_df.columns)
    has_latlon = {'Lat', 'Long', 'MpId'}.issubset(orders_cols)
    mp_to_centroid: Dict[int, Tuple[float, float]] = {}
    if has_latlon:
        grouped = (
            orders_df
            .group_by('MpId')
            .agg([
                pl.mean('Lat').alias('centroid_lat'),
                pl.mean('Long').alias('centroid_lon'),
            ])
        )
        cent_map = {int(r['MpId']): (float(r['centroid_lat']), float(r['centroid_lon'])) for r in grouped.iter_rows(named=True)}
        mp_to_centroid.update(cent_map)

    items: List[Tuple[int, float, int]] = []                             
    for row in optimized_polygons.iter_rows(named=True):
        mp = int(row['MpId'])
        if mp == 0:
            continue
        orders = 0
        try:
            orders_list = row.get('order_ids')
            if isinstance(orders_list, list):
                orders = len(orders_list)
            else:
                orders = int(row.get('order_count', 0) or 0)
        except Exception:
            orders = int(row.get('order_count', 0) or 0)
        lat, lon = mp_to_centroid.get(mp, (0.0, 0.0))
        dy = float(lat) - float(warehouse_lat)
        dx = float(lon) - float(warehouse_lon)
        angle = math.degrees(math.atan2(dy, dx)) % 360.0
        items.append((mp, angle, int(orders)))

                                                                            
    items.sort(key=lambda x: x[1])
    total_orders = sum(o for _, _, o in items) or 1
    target = total_orders / max(1, int(target_sectors))

    polygon_to_sector: Dict[int, int] = {}
    sector_to_polygons: Dict[int, List[int]] = {}
    sector_orders: Dict[int, int] = {}

    current_sector = 0
    acc = 0
    for mp, _, o in items:
        if current_sector >= target_sectors:
            current_sector = target_sectors - 1
        polygon_to_sector[mp] = current_sector
        sector_to_polygons.setdefault(current_sector, []).append(mp)
        sector_orders[current_sector] = sector_orders.get(current_sector, 0) + int(o)
        acc += int(o)
        if acc >= target and current_sector < target_sectors - 1:
            current_sector += 1
            acc = 0

                                                                     
    for s in range(target_sectors):
        if not sector_to_polygons.get(s):
                                                                                       
            donor = max(sector_orders.keys(), key=lambda k: sector_orders.get(k, 0)) if sector_orders else None
            if donor is None or not sector_to_polygons.get(donor):
                continue
            mp = sector_to_polygons[donor].pop()
            polygon_to_sector[mp] = s
            sector_to_polygons[s] = [mp]
            sector_orders[s] = mp and 1 or 0

    return polygon_to_sector, sector_to_polygons, sector_orders


