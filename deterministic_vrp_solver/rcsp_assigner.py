import logging
from typing import Dict, List, Tuple

import polars as pl


logger = logging.getLogger(__name__)


def _get_order_count(row: dict) -> int:
    try:
        orders_list = row.get('order_ids')
        if isinstance(orders_list, list):
            return int(len(orders_list))
        return int(row.get('order_count', 0) or 0)
    except Exception:
        return int(row.get('order_count', 0) or 0)


def compute_rcsp_assignment(
    polygons_df: pl.DataFrame,
    couriers_df: pl.DataFrame,
    distance_provider,
    courier_service_times: Dict[int, Dict[int, int]],
    max_time_per_courier: int,
    warehouse_id: int = 0,
    per_order_penalty: int = 3000,
    top_k_candidates: int = 12,
    refill_pool_size: int = 2000,
    beam_size: int = 3,
) -> Dict[int, List[int]]:
    """Детерминированный RCSP-подобный ассайнер на графе полигонов (beam=1).

    Идём пошагово: выбираем наименее загруженного курьера и добавляем ему один полигон
    с лучшей маржинальной полезностью (penalty - access_cost), учитывая текущую позицию
    (портал/склад) и ограничение по времени. Продолжаем, пока остаются выгодные назначения.

    Возвращает: assignment {courier_id: [MpId, ...]}.
    """
             
    courier_ids: List[int] = couriers_df['ID'].to_list() if 'ID' in couriers_df.columns else list(range(len(couriers_df)))
                                           
    mp_rows = [row for row in polygons_df.iter_rows(named=True)]
    polygons_index: Dict[int, dict] = {int(r['MpId']): r for r in mp_rows}
    available_polygons: List[int] = [int(r['MpId']) for r in mp_rows if int(r['MpId']) != int(warehouse_id)]

                                                                 
    def static_util(pid: int) -> float:
        try:
            orders = _get_order_count(polygons_index[pid]) or 1
            cost = float(distance_provider.get_polygon_access_cost(int(warehouse_id), int(pid), 0))
            if cost >= float('inf'):
                return 0.0
            return float(orders) * float(per_order_penalty) / (1.0 + cost)
        except Exception:
            return 0.0

    seed_sorted = sorted(available_polygons, key=static_util, reverse=True)
    seed_pool = seed_sorted[: max(1, int(refill_pool_size))]
    available_set = set(available_polygons)

                        
    courier_state: Dict[int, Dict] = {
        int(cid): {
            'position': int(warehouse_id),
            'time': 0.0,
            'assigned': [],
        }
        for cid in courier_ids
    }

                                 
    def candidate_score(cid: int, pid: int, current_pos: int, current_time: float) -> Tuple[float, float]:
        st = courier_service_times.get(int(cid), {}).get(int(pid))
        if st is None:
            return (-float('inf'), float('inf'))
        svc = int(st) * int(_get_order_count(polygons_index[int(pid)]))
        try:
            cost = float(distance_provider.get_polygon_access_cost(int(current_pos), int(pid), int(svc)))
        except Exception:
            return (-float('inf'), float('inf'))
        if cost >= float('inf') or (current_time + cost) > float(max_time_per_courier):
            return (-float('inf'), float('inf'))
        penalty = float(per_order_penalty) * float(_get_order_count(polygons_index[int(pid)]))
        score = float(penalty) - float(cost)
        return (score, cost)

                                           
    State = Tuple[Dict[int, Dict], set]

    def clone_state(state: Dict[int, Dict], avail: set) -> Tuple[Dict[int, Dict], set]:
        return ({cid: {'position': v['position'], 'time': float(v['time']), 'assigned': list(v['assigned'])} for cid, v in state.items()}, set(avail))

    beams: List[State] = [(courier_state, available_set)]

    while beams:
        new_beams: List[Tuple[float, State]] = []                       
        progressed = False
        for state, avail in beams:
                                                   
            active = [c for c in state.keys()]
            if not active or not avail:
                                    
                total = sum(float(v['time']) for v in state.values())
                new_beams.append((total, (state, avail)))
                continue
            cid = min(active, key=lambda c: state[c]['time'])
            current_pos = int(state[cid]['position'])
            current_time = float(state[cid]['time'])

                               
            candidates: List[Tuple[int, float, float]] = []
            for pid in seed_pool:
                if pid not in avail:
                    continue
                score, cost = candidate_score(int(cid), int(pid), current_pos, current_time)
                if score == -float('inf') or score <= 0.0:
                    continue
                candidates.append((int(pid), float(score), float(cost)))
                if len(candidates) >= int(top_k_candidates):
                    break
            if not candidates:
                                             
                for pid in seed_sorted:
                    if pid not in avail:
                        continue
                    score, cost = candidate_score(int(cid), int(pid), current_pos, current_time)
                    if score == -float('inf') or score <= 0.0:
                        continue
                    candidates.append((int(pid), float(score), float(cost)))
                    if len(candidates) >= int(top_k_candidates):
                        break

            if not candidates:
                                                    
                total = sum(float(v['time']) for v in state.values())
                new_beams.append((total, (state, avail)))
                continue

            progressed = True
            candidates.sort(key=lambda x: x[1], reverse=True)
            for pid, score, cost in candidates[: max(1, int(beam_size))]:
                st2, av2 = clone_state(state, avail)
                           
                st2[cid]['time'] = float(st2[cid]['time']) + float(cost)
                try:
                    best_port, _ = distance_provider.find_best_port_to_polygon(int(current_pos), int(pid))
                    if best_port is not None:
                        st2[cid]['position'] = int(best_port)
                except Exception:
                    pass
                st2[cid]['assigned'].append(int(pid))
                if pid in av2:
                    av2.remove(pid)
                total = sum(float(v['time']) for v in st2.values())
                new_beams.append((total, (st2, av2)))

                                                     
        if not progressed:
                                            
            if not new_beams:
                break
            best_state = min(new_beams, key=lambda x: x[0])[1]
            final_state, _ = best_state
            return {int(cid): [int(p) for p in st['assigned']] for cid, st in final_state.items()}

                                                     
        new_beams.sort(key=lambda x: x[0])
        beams = [state for _, state in new_beams[: max(1, int(beam_size))]]

        assigned_so_far = len(available_polygons) - min(len(av) for _, av in beams)
        if assigned_so_far > 0 and assigned_so_far % 200 == 0:
            logger.info(f"RCSP(beam): назначено ≈{assigned_so_far}/{len(available_polygons)} полигонов")

              
    final_state, _ = beams[0]
    return {int(cid): [int(p) for p in st['assigned']] for cid, st in final_state.items()}


