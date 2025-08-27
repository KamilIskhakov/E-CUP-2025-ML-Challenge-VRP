import heapq
from typing import Dict, List, Set, Tuple


def build_espprc_route_for_courier(
    courier_id: int,
    start_node: int,
    max_time: int,
    remaining_polygons: Set[int],
    dual_price: Dict[int, float],
    provider,
    courier_service_times: Dict[int, Dict[int, int]],
    mp_to_order_count: Dict[int, int],
    top_k: int = 24,
    max_labels: int = 5000,
    include_return: bool = True,
) -> Tuple[List[int], float, float]:
    """Упрощённый ESPPRC (label-setting с отсечками) для прайсинга колонн.

    Возвращает: (маршрут, время_без_возврата, редуцированная_стоимость), где
    reduced = (time + (return_to_wh если include_return)) - sum(dual[p]). Добавляем колонку, если reduced<0.
    """
    if not remaining_polygons or max_time <= 0:
        return [], 0.0, 0.0

                                                                 
    candidates = sorted(
        (int(p) for p in remaining_polygons),
        key=lambda p: float(dual_price.get(int(p), 0.0)) / (1.0 + float(mp_to_order_count.get(int(p), 1))),
        reverse=True,
    )
    if top_k and len(candidates) > int(top_k):
        candidates = candidates[: int(top_k)]

                                                                       
    best_at_node: Dict[int, List[Tuple[float, float]]] = {}

                                                    
                                                                       
    hq: List[Tuple[float, float, float, int, List[int], Set[int]]] = []
    heapq.heappush(hq, (0.0, 0.0, 0.0, int(start_node), [], set()))

    best_route: List[int] = []
    best_time: float = float('inf')
    best_reduced: float = 0.0
    num_labels = 0

    while hq and num_labels < int(max_labels):
        reduced, time_so_far, sum_pi, node, route, forbid = heapq.heappop(hq)
        num_labels += 1

                                        
        finish_penalty = 0.0
        if include_return:
            try:
                finish_penalty = float(provider.get_original_distance(int(node), int(start_node)))
            except Exception:
                finish_penalty = 0.0
        reduced_here = float(time_so_far) + float(finish_penalty) - float(sum_pi)
        if not best_route or reduced_here < best_reduced:
            best_route = list(route)
            best_time = float(time_so_far)
            best_reduced = float(reduced_here)

                                 
        scored: List[Tuple[int, float, float]] = []                         
        for pid in candidates:
            if pid in forbid:
                continue
            st = courier_service_times.get(int(courier_id), {}).get(int(pid))
            if st is None:
                continue
            orders = int(mp_to_order_count.get(int(pid), 0))
            if orders <= 0:
                continue
            add_time = provider.get_polygon_access_cost(int(node), int(pid), int(st) * int(orders))
            if add_time == float('inf'):
                continue
            new_time = float(time_so_far) + float(add_time)
            if new_time > float(max_time):
                continue
                                                                 
            if include_return:
                try:
                    ret = float(provider.get_original_distance(int(pid), int(start_node)))
                except Exception:
                    ret = 0.0
                if new_time + ret > float(max_time):
                    continue
            gain = float(dual_price.get(int(pid), 0.0))
                                                                                
            if (float(add_time) - float(gain)) >= 0.0:
                continue
            scored.append((int(pid), float(add_time), float(gain)))

        if not scored:
            continue

                                                                    
        scored.sort(key=lambda x: (x[1] - x[2]))
        for pid, add_t, gain in scored[: min(8, len(scored))]:
            new_time = float(time_so_far) + float(add_t)
            new_sum = float(sum_pi) + float(gain)
            new_route = [*route, int(pid)]
            new_forbid = set(forbid)
            new_forbid.add(int(pid))                  
            new_node = node
            try:
                best_port, _ = provider.find_best_port_to_polygon(int(node), int(pid))
                if best_port is not None:
                    new_node = int(best_port)
            except Exception:
                new_node = int(node)

                                                                                                                  
            dominated = False
            lst = best_at_node.setdefault(int(new_node), [])
            for t0, s0 in lst:
                if t0 <= new_time and s0 >= new_sum:
                    dominated = True
                    break
            if dominated:
                continue
            lst.append((new_time, new_sum))

                                                   
            lb_return = 0.0
            if include_return:
                try:
                    lb_return = float(provider.get_original_distance(int(new_node), int(start_node)))
                except Exception:
                    lb_return = 0.0
            new_reduced = float(new_time) + float(lb_return) - float(new_sum)
            heapq.heappush(hq, (new_reduced, new_time, new_sum, int(new_node), new_route, new_forbid))

                                                                           
    return best_route, float(best_time), float(best_reduced)


