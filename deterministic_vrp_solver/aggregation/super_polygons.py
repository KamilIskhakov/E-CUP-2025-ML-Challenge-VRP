import sqlite3
from typing import Dict, List, Tuple, Optional

import polars as pl


def _get_portal_neighbors(conn: sqlite3.Connection, portal_id: int, k_neighbors: int, d_portal_max: int) -> List[Tuple[int, int]]:
    """Возвращает список соседних порталов (to_port, distance) из ports_database.sqlite.

    Аргументы:
        conn: соединение с БД портов (ports_database.sqlite)
        portal_id: id портала (from_port)
        k_neighbors: максимум соседей
        d_portal_max: максимальная дистанция портал→портал, сек
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT to_port, distance
        FROM port_distances
        WHERE from_port = ? AND distance > 0 AND distance <= ?
        ORDER BY distance ASC
        LIMIT ?
        """,
        (int(portal_id), int(d_portal_max), int(k_neighbors)),
    )
    return [(int(r[0]), int(r[1])) for r in cur.fetchall()]


def _greedy_route_distance(conn: sqlite3.Connection, portal_ids: List[int]) -> Tuple[List[int], int]:
    """Грубая оценка микро-маршрута по порталам в кластере: жадный NN.

    Возвращает порядок порталов и суммарную дистанцию между соседними порталами.
    """
    if not portal_ids:
        return [], 0
    if len(portal_ids) == 1:
        return [int(portal_ids[0])], 0

    remaining = [int(p) for p in portal_ids]
                                                                     
    cur = conn.cursor()
    best_start = remaining[0]
    best_sum = 1 << 60
    for p in remaining:
        placeholders = ",".join(["?"] * (len(remaining) - 1))
        args = [p] + [x for x in remaining if x != p]
        cur.execute(
            f"SELECT COALESCE(SUM(distance), 0) FROM port_distances WHERE from_port = ? AND to_port IN ({placeholders})",
            args,
        )
        s = int(cur.fetchone()[0] or 0)
        if s < best_sum:
            best_sum = s
            best_start = p

    route = [best_start]
    remaining.remove(best_start)
    total_d = 0
    while remaining:
                                         
        placeholders = ",".join(["?"] * len(remaining))
        args = [route[-1]] + remaining
        cur.execute(
            f"SELECT to_port, distance FROM port_distances WHERE from_port = ? AND to_port IN ({placeholders}) ORDER BY distance ASC",
            args,
        )
        row = cur.fetchone()
        if row is None:
                                             
            route.extend(remaining)
            break
        nxt, d = int(row[0]), int(row[1])
        total_d += d
        route.append(nxt)
        remaining.remove(nxt)
    return route, int(total_d)


def build_super_polygons(
    optimized_polygons: pl.DataFrame,
    ports_db_path: str,
    d_portal_max: int = 900,
    k_neighbors: int = 10,
    max_polygons_per_cluster: int = 5,
    max_orders_per_cluster: int = 120,
    max_internal_distance: int = 1500,
    service_time_by_mp: Optional[Dict[int, int]] = None,
) -> List[Dict]:
    """Строит супер-полигоны (кластеры) на основе соседства порталов.

    Возвращает список кластеров с полями:
      - macro_id: int
      - members: List[int] (MpId)
      - portals: List[int]
      - micro_route_portals: List[int]
      - internal_distance: int
      - total_orders: int
      - sum_total_cost: int
    """
    if not optimized_polygons.is_empty():
        assert 'MpId' in optimized_polygons.columns, "optimized_polygons должен содержать MpId"
        assert 'portal_id' in optimized_polygons.columns, "optimized_polygons должен содержать portal_id"

    conn_ports = sqlite3.connect(ports_db_path)

                                  
    mp_rows = [row for row in optimized_polygons.iter_rows(named=True)]
    mp_to_portal: Dict[int, int] = {}
    mp_to_orders: Dict[int, int] = {}
    mp_to_cost: Dict[int, int] = {}
    mp_to_svc: Dict[int, int] = {}
    for r in mp_rows:
        mp_id = int(r['MpId'])
        portal_id = int(r.get('portal_id', 0) or 0)
        mp_to_portal[mp_id] = portal_id
                
        orders = 0
        try:
            orders_list = r.get('order_ids')
            if isinstance(orders_list, list):
                orders = len(orders_list)
            else:
                orders = int(r.get('order_count', 0) or 0)
        except Exception:
            orders = int(r.get('order_count', 0) or 0)
        mp_to_orders[mp_id] = int(orders)
                                    
        total_cost = int(r.get('total_cost', r.get('total_distance', 0)) or 0)
        mp_to_cost[mp_id] = total_cost
                                                           
        if service_time_by_mp is not None and mp_id in service_time_by_mp:
            mp_to_svc[mp_id] = int(service_time_by_mp[mp_id])

                                                                            
    portal_neighbors_cache: Dict[int, List[Tuple[int, int]]] = {}

    def neighbors(portal_id: int) -> List[Tuple[int, int]]:
        if portal_id not in portal_neighbors_cache:
            portal_neighbors_cache[portal_id] = _get_portal_neighbors(
                conn_ports, portal_id, k_neighbors=k_neighbors, d_portal_max=d_portal_max
            )
        return portal_neighbors_cache[portal_id]

                                                                                      
    unassigned = set(mp_to_portal.keys())
    clusters: List[Dict] = []
    macro_id = 1

                                                                              
                                                                                          
                               
    def work_cost(mp_id: int) -> float:
        base = float(mp_to_cost.get(mp_id, 0) or 0.0)
        svc = float(mp_to_svc.get(mp_id, 0) or 0.0)
        return base + svc

    def score(mp_id: int) -> float:
        wc = work_cost(mp_id)
        if wc <= 0:
            wc = 1.0
        return float(mp_to_orders.get(mp_id, 0)) / wc

    for start_mp in sorted(list(unassigned), key=score, reverse=True):
        if start_mp not in unassigned:
            continue
        cluster_mps = [start_mp]
        cluster_portals = [mp_to_portal[start_mp]] if mp_to_portal[start_mp] else []
        total_orders = mp_to_orders[start_mp]
        sum_work = work_cost(start_mp)

                                                     
        frontier = []
        p0 = mp_to_portal[start_mp]
        if p0:
            frontier.extend([p for p, _ in neighbors(p0)])

                                                      
        portal_to_mps: Dict[int, List[int]] = {}
        for mp_id, prt in mp_to_portal.items():
            portal_to_mps.setdefault(prt, []).append(mp_id)

        while frontier and len(cluster_mps) < max_polygons_per_cluster:
            prt = frontier.pop(0)
                                                                            
            candidates = [mp for mp in portal_to_mps.get(prt, []) if mp in unassigned]
            if not candidates:
                continue
                                             
            next_mp = max(candidates, key=score)

                                      
            if total_orders + mp_to_orders[next_mp] > max_orders_per_cluster:
                continue

            tmp_portals = list(set(cluster_portals + [mp_to_portal[next_mp]])) if mp_to_portal[next_mp] else list(cluster_portals)
                                                   
            route, internal_d = _greedy_route_distance(conn_ports, tmp_portals)
            if internal_d > max_internal_distance:
                continue

                               
            cluster_mps.append(next_mp)
            if mp_to_portal[next_mp] and mp_to_portal[next_mp] not in cluster_portals:
                cluster_portals.append(mp_to_portal[next_mp])
            total_orders += mp_to_orders[next_mp]
            sum_work += work_cost(next_mp)

                                                      
            if mp_to_portal[next_mp]:
                for nb, _ in neighbors(mp_to_portal[next_mp]):
                    frontier.append(nb)

                                        
        micro_route, internal_d = _greedy_route_distance(conn_ports, cluster_portals)

        clusters.append({
            'macro_id': macro_id,
            'members': cluster_mps,
            'portals': cluster_portals,
            'micro_route_portals': micro_route,
            'internal_distance': int(internal_d),
            'total_orders': int(total_orders),
            'sum_total_cost': int(sum_work),
        })
        unassigned.difference_update(cluster_mps)
        macro_id += 1

    conn_ports.close()
    return clusters


