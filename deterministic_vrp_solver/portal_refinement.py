from typing import Dict, List, Tuple


def _dp_best_port_sequence(
    polygon_ids: List[int],
    provider,
    warehouse_id: int,
    default_portals: List[int],
) -> List[int]:
    """Подбирает по каждому полигону лучший портал с учётом соседей (минимизируя сумму переходов).

    Решается через ДП по слоям: слой = полигон, состояние = выбранный порт, стоимость = переходы между портами.
    Используются провайдерские функции:
      - get_polygon_ports(mp_id)
      - get_distance_from_warehouse_to_port(port)
      - get_distance_between_ports(p1, p2)
    Если список портов пуст — используем дефолтный портал.
    """
    n = len(polygon_ids)
    if n == 0:
        return []

    candidates: List[List[int]] = []
    for i, pid in enumerate(polygon_ids):
        ports = provider.get_polygon_ports(int(pid)) or []
        if not ports:
                                         
            ports = [int(default_portals[i])]
                           
        uniq = []
        seen = set()
        for p in ports:
            p = int(p)
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        candidates.append(uniq)

                                               
    INF = 10**18
    dp: List[Dict[int, Tuple[int, int]]] = [{} for _ in range(n)]
                                     

    for p in candidates[0]:
        cost0 = provider.get_distance_from_warehouse_to_port(int(p))
        dp[0][int(p)] = (int(cost0) if cost0 < float('inf') else INF, -1)

                           
    for i in range(1, n):
        for cur in candidates[i]:
            best_cost = INF
            best_prev = -1
            for prev, (prev_cost, _) in dp[i - 1].items():
                if prev_cost >= INF:
                    continue
                d = provider.get_distance_between_ports(int(prev), int(cur))
                if d >= float('inf'):
                    continue
                total = prev_cost + int(d)
                if total < best_cost:
                    best_cost = total
                    best_prev = int(prev)
            dp[i][int(cur)] = (best_cost, best_prev)

                                              
    best_last_port = None
    best_total = INF
    for last_port, (cost, prev) in dp[-1].items():
        if cost >= INF:
            continue
        back = provider.get_distance_from_warehouse_to_port(int(last_port))
        if back >= float('inf'):
            continue
        total = cost + int(back)
        if total < best_total:
            best_total = total
            best_last_port = int(last_port)

                                                           
    if best_last_port is None:
        return [int(x) for x in default_portals]

                      
    seq: List[int] = [0] * n
    cur = best_last_port
    for i in range(n - 1, -1, -1):
        seq[i] = int(cur)
        cur = dp[i][int(cur)][1]
        if cur == -1:
            break
    return seq


def refine_routes_portals(
    optimized_routes: Dict[int, Dict],
    provider,
    warehouse_id: int = 0,
) -> Dict[int, Dict]:
    """Для каждого маршрута подбирает лучшие порталы и пересчитывает общую длину переходов порт↔порт.

    Полигонные стоимости (внутренний TSP + сервис) не меняем; обновляем только межпортальные переходы.
    Ожидается, что в optimized_routes[cid]['route_details'] есть поля 'portal_id' и 'polygon_cost'.
    """
    if not optimized_routes:
        return optimized_routes

    result = {}
    for cid, route in optimized_routes.items():
        if not route or not route.get('route_details'):
            result[cid] = route
            continue
        details = route['route_details']
        poly_ids = [int(it['polygon_id']) for it in details]
        default_ports = [int(it.get('portal_id', 0) or 0) for it in details]
        best_ports = _dp_best_port_sequence(poly_ids, provider, int(warehouse_id), default_ports)

                                           
        if not best_ports:
            result[cid] = route
            continue
        travel = 0
                         
        travel += int(provider.get_distance_from_warehouse_to_port(int(best_ports[0])) or 0)
                       
        for i in range(len(best_ports) - 1):
            d = provider.get_distance_between_ports(int(best_ports[i]), int(best_ports[i + 1]))
            travel += int(d) if d < float('inf') else 0
                            
        travel += int(provider.get_distance_from_warehouse_to_port(int(best_ports[-1])) or 0)

                                     
        poly_sum = sum(int(it.get('polygon_cost', 0) or 0) for it in details)
        total_time = int(travel) + int(poly_sum)

        new_details: List[Dict] = []
        for i, it in enumerate(details):
            nd = dict(it)
            nd['portal_id'] = int(best_ports[i])
            new_details.append(nd)

        result[cid] = {
            'polygon_order': list(route.get('polygon_order') or []),
            'total_time': int(total_time),
            'route_details': new_details,
        }

    return result


