from typing import Dict, List, Tuple
import sqlite3
import polars as pl


def optimize_and_improve_routes(
    optimized_polygons: pl.DataFrame,
    assignment: Dict[int, List[int]],
    courier_service_times: Dict[int, Dict[int, int]],
    provider,
    conn: sqlite3.Connection,
    warehouse_id: int,
    max_time_per_courier: int,
    post_improvers: str,
    fast: bool,
) -> Tuple[Dict[int, Dict], List[Dict]]:
    from deterministic_vrp_solver.route_optimizer import RouteOptimizer
    ro = RouteOptimizer(conn, warehouse_id=warehouse_id, courier_service_times=courier_service_times)
    optimized_routes = ro.optimize_all_courier_routes(assignment, optimized_polygons)
    improved_routes = ro.improve_routes_local_search(optimized_routes, assignment, optimized_polygons)

    from deterministic_vrp_solver.strategies.factory import build_post_improver
    improver = build_post_improver(str(post_improvers))
    improved_routes = improver.improve(
        ro, improved_routes, assignment, optimized_polygons, provider, int(max_time_per_courier)
    )

    from deterministic_vrp_solver.alns import improve_routes_pair_swap
    improved_routes = improve_routes_pair_swap(
        improved_routes,
        assignment,
        optimized_polygons,
        ro,
        time_cap=int(max_time_per_courier),
        max_iters=(1 if fast else 2),
        tail_sample=2,
        candidate_pairs=40,
    )

    time_cap = int(max_time_per_courier)
    for cid, info in list(improved_routes.items()):
        if not info or not info.get('polygon_order'):
            continue
        if int(info.get('total_time', 0)) <= time_cap:
            continue
        polygons = list(info['polygon_order'])
        while polygons and int(info.get('total_time', 0)) > time_cap:
            polygons.pop()
            info = ro.optimize_courier_route(polygons, optimized_polygons, int(cid))
        improved_routes[cid] = info
        assignment[cid] = polygons

    def rotate_to_start(route_list, start_node):
        if not route_list:
            return []
        try:
            idx = route_list.index(int(start_node))
            return route_list[idx:] + route_list[:idx]
        except ValueError:
            return route_list

    polygons_index = {row['MpId']: row for row in optimized_polygons.iter_rows(named=True)}
    final_routes: List[Dict] = []
    for courier_id, info in improved_routes.items():
        if not info or not info.get('polygon_order'):
            continue
        if int(courier_id) == 0:
            continue
        full_route = [0]
        for pid in info['polygon_order']:
            row = polygons_index.get(int(pid))
            if not row:
                continue
            optimal_route = row.get('optimal_route') or []
            portal_id = int(row.get('portal_id', 0))
            rotated = rotate_to_start(optimal_route, portal_id)
            if rotated:
                full_route.extend(int(x) for x in rotated)
        full_route.append(0)
        if len(full_route) <= 2:
            continue
        final_routes.append({
            'courier_id': int(courier_id),
            'route': full_route,
        })

    return improved_routes, final_routes


