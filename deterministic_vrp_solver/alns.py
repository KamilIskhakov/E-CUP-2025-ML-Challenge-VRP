import logging
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


def improve_assignment_alns(
    assignment: Dict[int, List[int]],
    provider,
    courier_service_times: Dict[int, Dict[int, int]],
    mp_to_order_count: Dict[int, int],
    warehouse_id: int,
    max_time_per_courier: int,
    per_order_penalty: int = 3000,
    iterations: int = 30,
    overloaded_top: int = 10,
    remove_per_overloaded: int = 2,
) -> Dict[int, List[int]]:
    """Простой детерминированный ALNS поверх назначения представителей.

    Ruin: снимаем хвостовые полигоны у самых нагруженных курьеров.
    Recreate: вставляем каждый снятый полигон к лучшему курьеру с выгодой (penalty - add) и без превышения лимита.
    Принимаем шаг, если суммарное время снизилось.
    """
                               
    def polygon_cost_for(cid: int, pid: int) -> float:
        st = courier_service_times.get(int(cid), {}).get(int(pid))
        if st is None:
            return float('inf')
        svc = int(st) * int(mp_to_order_count.get(int(pid), 0))
        try:
            return float(provider.get_polygon_access_cost(int(warehouse_id), int(pid), int(svc)))
        except Exception:
            return float('inf')

    def total_time(assign: Dict[int, List[int]]) -> Dict[int, float]:
        times: Dict[int, float] = {int(cid): 0.0 for cid in assign.keys()}
        for cid, lst in assign.items():
            acc = 0.0
            for pid in lst:
                c = polygon_cost_for(int(cid), int(pid))
                if c == float('inf'):
                    continue
                acc += float(c)
            times[int(cid)] = acc
        return times

    current = {int(cid): [int(p) for p in lst] for cid, lst in assignment.items()}
    times = total_time(current)
    best_assign = {cid: lst[:] for cid, lst in current.items()}
    best_sum = sum(times.values())

    for it in range(max(1, int(iterations))):
                                      
        sorted_cids = sorted(times.keys(), key=lambda c: times[c], reverse=True)
        victims = sorted_cids[: max(1, int(overloaded_top))]
        removed: List[Tuple[int, int, float]] = []                         
                                            
        for cid in victims:
            take = min(max(1, int(remove_per_overloaded)), len(current.get(cid, [])))
            for _ in range(take):
                if not current[cid]:
                    break
                pid = current[cid].pop()         
                c = polygon_cost_for(int(cid), int(pid))
                removed.append((int(pid), int(cid), float(c)))
                times[cid] -= float(c) if c != float('inf') else 0.0

                                             
        improved = False
        for pid, from_cid, c_old in removed:
            best_target = None
            best_delta = 0.0
            for cid in current.keys():
                st = courier_service_times.get(int(cid), {}).get(int(pid))
                if st is None:
                    continue
                svc = int(st) * int(mp_to_order_count.get(int(pid), 0))
                try:
                    add = float(provider.get_polygon_access_cost(int(warehouse_id), int(pid), int(svc)))
                except Exception:
                    add = float('inf')
                if add == float('inf'):
                    continue
                                            
                penalty_gain = float(per_order_penalty) * float(mp_to_order_count.get(int(pid), 0))
                if not (add < penalty_gain):
                    continue
                if times[int(cid)] + add > float(max_time_per_courier):
                    continue
                delta = add                                                                        
                if best_target is None or delta < best_delta:
                    best_target = int(cid)
                    best_delta = float(delta)

            if best_target is None:
                                                                       
                current[from_cid].append(int(pid))
                if c_old != float('inf'):
                    times[from_cid] += float(c_old)
            else:
                current[best_target].append(int(pid))
                times[best_target] += float(best_delta)
                improved = True

        new_sum = sum(times.values())
        if new_sum + 1e-6 < best_sum:
            best_sum = new_sum
            best_assign = {cid: lst[:] for cid, lst in current.items()}
            logger.info(f"ALNS: улучшение суммы времени до {best_sum:.0f}")
        else:
            if not improved:
                break

    return best_assign


def improve_routes_alns(
    optimized_routes: Dict[int, Dict],
    assignment: Dict[int, List[int]],
    polygons_df,
    route_optimizer,
    max_iters: int = 2,
    remove_k: int = 2,
    time_cap: int = 43200,
) -> Dict[int, Dict]:
    """Пост-ALNS на уровне маршрутов: снимаем хвост у худших и жадно вставляем лучшим."""
    if not optimized_routes:
        return optimized_routes

    for _ in range(max_iters):
        items = sorted(
            [(cid, r.get('total_time', 0)) for cid, r in optimized_routes.items() if r],
            key=lambda x: x[1], reverse=True,
        )
        if not items:
            break
        worst = [cid for cid, _ in items[:10]]
        best = [cid for cid, _ in items[-20:]]

        improved = False
        for cid_w in worst:
            r_w = optimized_routes.get(cid_w) or {}
            polys_w = list(r_w.get('polygon_order') or [])
            if len(polys_w) == 0:
                continue
            removed = polys_w[-remove_k:]
            base_w = polys_w[:-remove_k] if remove_k <= len(polys_w) else []
            new_w = route_optimizer.optimize_courier_route(base_w, polygons_df, cid_w)
            if new_w.get('total_time', 0) == 0:
                continue
            for pid in removed:
                best_gain = 0.0
                best_cid = None
                best_route_t = None
                for cid_t in best:
                    if cid_t == cid_w:
                        continue
                    r_t = optimized_routes.get(cid_t) or {}
                    polys_t = list(r_t.get('polygon_order') or [])
                    new_polys_t = polys_t + [pid]
                    new_t = route_optimizer.optimize_courier_route(new_polys_t, polygons_df, cid_t)
                    if new_t.get('total_time', 0) == 0:
                        continue
                    if new_t['total_time'] > time_cap:
                        continue
                    old_sum = float(r_w.get('total_time', 0)) + float(r_t.get('total_time', 0))
                    new_sum = float(new_w.get('total_time', 0)) + float(new_t.get('total_time', 0))
                    gain = old_sum - new_sum
                    if gain > best_gain:
                        best_gain = gain
                        best_cid = cid_t
                        best_route_t = new_t
                if best_cid is not None and best_gain > 0.0:
                    assignment[cid_w] = list(new_w.get('polygon_order') or [])
                    optimized_routes[cid_w] = new_w
                    r_best_t = optimized_routes.get(best_cid) or {}
                    assignment[best_cid] = list((r_best_t.get('polygon_order') or [])) + [pid]
                    optimized_routes[best_cid] = best_route_t
                    r_w = new_w
                    polys_w = list(r_w.get('polygon_order') or [])
                    improved = True
                else:
                                                                     
                    base_w2 = polys_w + [pid]
                    new_w2 = route_optimizer.optimize_courier_route(base_w2, polygons_df, cid_w)
                    if new_w2.get('total_time', 0) > 0 and new_w2['total_time'] <= time_cap:
                        assignment[cid_w] = list(new_w2.get('polygon_order') or [])
                        optimized_routes[cid_w] = new_w2
                        r_w = new_w2
                        polys_w = list(r_w.get('polygon_order') or [])
            if improved:
                break
        if not improved:
            break

    return optimized_routes


def improve_routes_pair_swap(
    optimized_routes: Dict[int, Dict],
    assignment: Dict[int, List[int]],
    polygons_df,
    route_optimizer,
    time_cap: int = 43200,
    max_iters: int = 2,
    tail_sample: int = 3,
    candidate_pairs: int = 40,
) -> Dict[int, Dict]:
    """Межкурьерский swap: обмениваем хвостовые полигоны между парами курьеров, если суммарное время падает.

    Ограничиваемся хвостовыми tail_sample полигонами у каждого, перебираем не более candidate_pairs лучших пар (перегруженный, разгруженный).
    """
    if not optimized_routes:
        return optimized_routes

    for _ in range(max_iters):
        items = sorted(
            [(cid, r.get('total_time', 0)) for cid, r in optimized_routes.items() if r],
            key=lambda x: x[1], reverse=True,
        )
        if not items:
            break
        worst = [cid for cid, _ in items[:20]]
        best = [cid for cid, _ in items[-40:]]

        improved = False
        tried = 0
        for cid_w in worst:
            if tried >= candidate_pairs:
                break
            r_w = optimized_routes.get(cid_w) or {}
            polys_w = list(r_w.get('polygon_order') or [])
            if len(polys_w) == 0:
                continue
            tail_w = polys_w[-max(1, int(tail_sample)) :]
            base_w = polys_w[: len(polys_w) - len(tail_w)]
            base_w_route = route_optimizer.optimize_courier_route(base_w, polygons_df, cid_w)
            if base_w_route.get('total_time', 0) == 0:
                continue

            for cid_b in best:
                if cid_b == cid_w:
                    continue
                tried += 1
                if tried > candidate_pairs:
                    break
                r_b = optimized_routes.get(cid_b) or {}
                polys_b = list(r_b.get('polygon_order') or [])
                if len(polys_b) == 0:
                    continue
                tail_b = polys_b[-max(1, int(tail_sample)) :]

                                                                  
                best_gain = 0.0
                best_choice = None                                    
                for p in tail_w:
                    for q in tail_b:
                                               
                        new_w_list = base_w + [q] + [x for x in tail_w if x != p]
                        new_b_list = [x for x in polys_b if x != q] + [p]
                        new_w_route = route_optimizer.optimize_courier_route(new_w_list, polygons_df, cid_w)
                        new_b_route = route_optimizer.optimize_courier_route(new_b_list, polygons_df, cid_b)
                        tw = float(new_w_route.get('total_time', 0))
                        tb = float(new_b_route.get('total_time', 0))
                        if tw <= 0 or tb <= 0:
                            continue
                        if tw > float(time_cap) or tb > float(time_cap):
                            continue
                        old_sum = float(r_w.get('total_time', 0)) + float(r_b.get('total_time', 0))
                        new_sum = tw + tb
                        gain = old_sum - new_sum
                        if gain > best_gain:
                            best_gain = gain
                            best_choice = (p, q, new_w_route, new_b_route)

                if best_choice is not None and best_gain > 0.0:
                    p, q, nw, nb = best_choice
                                     
                    assignment[cid_w] = list(nw.get('polygon_order') or [])
                    optimized_routes[cid_w] = nw
                    assignment[cid_b] = list(nb.get('polygon_order') or [])
                    optimized_routes[cid_b] = nb
                    r_w = nw
                    improved = True
                    break

            if improved:
                break
        if not improved:
            break

    return optimized_routes

