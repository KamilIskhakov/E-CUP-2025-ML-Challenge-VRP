import logging
from typing import Dict, List, Tuple, Set

import polars as pl


logger = logging.getLogger(__name__)


def _get_order_count_from_row(row: dict) -> int:
    try:
        orders_list = row.get('order_ids')
        if isinstance(orders_list, list):
            return int(len(orders_list))
        return int(row.get('order_count', 0) or 0)
    except Exception:
        return int(row.get('order_count', 0) or 0)


def _route_incremental_cost(provider, from_node: int, polygon_id: int, per_order_time: int, orders: int, warehouse_id: int) -> float:
    svc = int(per_order_time) * int(orders)
    try:
        return float(provider.get_polygon_access_cost(int(from_node), int(polygon_id), int(svc)))
    except Exception:
        return float('inf')


def _build_pricing_route_for_courier(
    courier_id: int,
    start_node: int,
    max_time: int,
    remaining_polygons: Set[int],
    dual_price: Dict[int, float],
    provider,
    courier_service_times: Dict[int, Dict[int, int]],
    mp_to_order_count: Dict[int, int],
    top_k: int = 16,
    beam_size: int = 3,
) -> Tuple[List[int], float, float]:
    """Голодный pricing: жадно строим маршрут с положительной суммарной выгодой sum(pi) - time.

    Возвращает (маршрут_полигонов, время_маршрута, редуцированная_стоимость) где
    reduced = time - sum(pi[p] for p in route). Добавляем колонку, если reduced < 0.
    """
                                                                                      
                                                                                                   
    candidates0 = sorted(list(remaining_polygons), key=lambda p: float(dual_price.get(int(p), 0.0)) / (1.0 + float(mp_to_order_count.get(int(p), 1))), reverse=True)
    if top_k and len(candidates0) > top_k:
        candidates0 = candidates0[: top_k]

    BeamState = Tuple[float, float, int, List[int], Set[int]]                                       
    initial_remain = set(remaining_polygons)
    beams: List[BeamState] = [(0.0, 0.0, int(start_node), [], initial_remain)]

    best_route: List[int] = []
    best_time = float('inf')
    best_reduced = 0.0

                                                             
    max_depth = len(candidates0)
    for _ in range(max_depth):
        new_beams: List[Tuple[float, BeamState]] = []                         
        progressed = False
        for time_so_far, sum_pi, node, route, remain in beams:
                                                                                             
            candidates = [p for p in candidates0 if p in remain]
            if not candidates:
                              
                reduced = float(time_so_far) - float(sum_pi)
                new_beams.append((reduced, (time_so_far, sum_pi, node, route, remain)))
                continue
                                                  
            scored: List[Tuple[int, float, float]] = []                         
            for pid in candidates:
                st = courier_service_times.get(int(courier_id), {}).get(int(pid))
                if st is None:
                    continue
                orders = int(mp_to_order_count.get(int(pid), 0))
                if orders <= 0:
                    continue
                add_time = _route_incremental_cost(provider, node, int(pid), int(st), int(orders), int(start_node))
                if add_time == float('inf'):
                    continue
                if (time_so_far + add_time) > float(max_time):
                    continue
                gain = float(dual_price.get(int(pid), 0.0)) - float(add_time)
                if gain <= 0.0:
                    continue
                scored.append((int(pid), float(gain), float(add_time)))
            if not scored:
                reduced = float(time_so_far) - float(sum_pi)
                new_beams.append((reduced, (time_so_far, sum_pi, node, route, remain)))
                continue
            progressed = True
            scored.sort(key=lambda x: x[1], reverse=True)
            for pid, gain, add_t in scored[: max(1, int(beam_size))]:
                new_time = float(time_so_far) + float(add_t)
                new_sum_pi = float(sum_pi) + float(dual_price.get(int(pid), 0.0))
                new_route = [*route, int(pid)]
                new_remain = set(remain)
                new_remain.discard(int(pid))
                new_node = node
                try:
                    best_port, _ = provider.find_best_port_to_polygon(int(node), int(pid))
                    if best_port is not None:
                        new_node = int(best_port)
                except Exception:
                    pass
                reduced = float(new_time) - float(new_sum_pi)
                new_beams.append((reduced, (new_time, new_sum_pi, new_node, new_route, new_remain)))
        if not progressed:
                                                    
            if not new_beams:
                break
            reduced, state = min(new_beams, key=lambda x: x[0])
            time_so_far, sum_pi, node, route, remain = state
            best_route = route
            best_time = time_so_far
            best_reduced = reduced
            break
                                                       
        new_beams.sort(key=lambda x: x[0])
        beams = [state for _, state in new_beams[: max(1, int(beam_size))]]

    if best_route == []:
                                                                                    
        candidates_final = [(float(t) - float(s), (t, s, n, r, rem)) for (t, s, n, r, rem) in beams]
        if candidates_final:
            best_reduced, state = min(candidates_final, key=lambda x: x[0])
            best_time, _, _, best_route, _ = state

    return best_route, float(best_time), float(best_reduced)


def run_column_generation_assignment(
    optimized_polygons: pl.DataFrame,
    couriers_df: pl.DataFrame,
    provider,
    courier_service_times: Dict[int, Dict[int, int]],
    mp_to_order_count: Dict[int, int],
    warehouse_id: int,
    max_time_per_courier: int,
    per_order_penalty: int = 3000,
    iterations: int = 3,
    use_espprc: bool = True,
) -> Dict[int, List[int]]:
    """Упрощённая колонно-генерация: dual-прайсинг по полигонам и greedy pricing маршруты per-курьер.

    - dual цены init: per_order_penalty * orders
    - pricing: строим маршрут с максимальной выгодой sum(pi) - time; если reduced<0, добавляем колонку
    - update dual: снижаем цены покрытых полигонов, чтобы избежать перепокрытия
    Возвращает assignment: courier_id -> [polygon_ids]
    """
    polygon_ids = [int(r['MpId']) for r in optimized_polygons.iter_rows(named=True) if int(r['MpId']) != int(warehouse_id)]
    remaining: Set[int] = set(polygon_ids)

                                                 
    dual_price: Dict[int, float] = {int(pid): float(per_order_penalty) * float(mp_to_order_count.get(int(pid), 0)) for pid in polygon_ids}

    courier_ids: List[int] = couriers_df['ID'].to_list() if 'ID' in couriers_df.columns else list(range(len(couriers_df)))
    assignment: Dict[int, List[int]] = {int(cid): [] for cid in courier_ids}
    courier_time: Dict[int, float] = {int(cid): 0.0 for cid in courier_ids}

    for it in range(int(iterations)):
        logger.info(f"ColumnGen: итерация {it+1}/{iterations}, осталось покрыть {len(remaining)}")
        any_column_added = False
                                                          
        for cid in sorted(courier_ids, key=lambda c: courier_time[int(c)]):
            if not remaining:
                break
                                 
            if use_espprc:
                try:
                    from deterministic_vrp_solver.espprc import build_espprc_route_for_courier
                    route, add_time, reduced = build_espprc_route_for_courier(
                        int(cid),
                        int(warehouse_id),
                        int(max_time_per_courier) - int(courier_time[int(cid)]),
                        remaining,
                        dual_price,
                        provider,
                        courier_service_times,
                        mp_to_order_count,
                        top_k=32,
                        max_labels=4000,
                        include_return=True,
                    )
                except Exception as e:
                    logger.warning(f"ESPPRC прайсинг недоступен: {e}; откат к greedy beam")
                    route, add_time, reduced = _build_pricing_route_for_courier(
                        int(cid),
                        int(warehouse_id),
                        int(max_time_per_courier) - int(courier_time[int(cid)]),
                        remaining,
                        dual_price,
                        provider,
                        courier_service_times,
                        mp_to_order_count,
                        top_k=32,
                        beam_size=3,
                    )
            else:
                route, add_time, reduced = _build_pricing_route_for_courier(
                    int(cid),
                    int(warehouse_id),
                    int(max_time_per_courier) - int(courier_time[int(cid)]),
                    remaining,
                    dual_price,
                    provider,
                    courier_service_times,
                    mp_to_order_count,
                    top_k=32,
                    beam_size=3,
                )
            if route and float(reduced) < 0.0 and add_time < float('inf'):
                                 
                assignment[int(cid)].extend(int(p) for p in route)
                courier_time[int(cid)] += float(add_time)
                any_column_added = True
                                                                                
                for p in route:
                    if int(p) in remaining:
                        remaining.discard(int(p))
                                                              
                avg_gain = sum(float(dual_price.get(int(p), 0.0)) for p in route) / max(1, len(route))
                for p in route:
                                                                           
                    cur = float(dual_price.get(int(p), 0.0))
                    step = 0.25 if cur > 1.5 * avg_gain else 0.15
                    dual_price[int(p)] = max(0.0, (1.0 - step) * cur)
        if not any_column_added:
            break

                                                                                   
    if remaining:
        logger.info(f"ColumnGen: финальный добор оставшихся {len(remaining)}")
        for pid in list(sorted(list(remaining), key=lambda p: dual_price.get(int(p), 0.0), reverse=True)):
            best_cid = None
            best_new_time = float('inf')
            for cid in courier_ids:
                st = courier_service_times.get(int(cid), {}).get(int(pid))
                if st is None:
                    continue
                orders = int(mp_to_order_count.get(int(pid), 0))
                add = _route_incremental_cost(provider, int(warehouse_id), int(pid), int(st), int(orders), int(warehouse_id))
                if add == float('inf'):
                    continue
                new_time = float(courier_time[int(cid)]) + float(add)
                if new_time <= float(max_time_per_courier) and new_time < best_new_time:
                    best_new_time = new_time
                    best_cid = int(cid)
            if best_cid is not None:
                assignment[int(best_cid)].append(int(pid))
                courier_time[int(best_cid)] = float(best_new_time)
                remaining.discard(int(pid))

                                                                       
    for cid in list(assignment.keys()):
        seen = set()
        ordered: List[int] = []
        for p in assignment[int(cid)]:
            if int(p) in seen:
                continue
            seen.add(int(p))
            ordered.append(int(p))
        assignment[int(cid)] = ordered

    return assignment


