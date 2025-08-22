import logging
from typing import Dict, List

import polars as pl

logger = logging.getLogger(__name__)

def greedy_initialize_assignment(
    polygons_df: pl.DataFrame,
    couriers_df: pl.DataFrame,
    max_time_per_courier: int,
    distance_provider,
    courier_service_times: Dict[int, Dict[int, int]],
    warehouse_id: int = 0,
    top_k: int = 50,
) -> Dict[int, List[int]]:
    """Жадное начальное назначение полигонов курьерам с учетом сервисных времен.

    Возвращает словарь {courier_id: [polygon_ids...]}.
    """
    # Подготовка словаря order_count по MpId
    order_count_map: Dict[int, int] = {}
    if 'MpId' in polygons_df.columns and 'order_count' in polygons_df.columns:
        for row in polygons_df.select(['MpId', 'order_count']).iter_rows():
            order_count_map[row[0]] = row[1]

    # Инициализация состояний курьеров
    courier_state = {
        cid: {
            'position': warehouse_id,
            'time': 0,
            'assigned': []
        }
        for cid in range(len(couriers_df))
    }

    # Доступные полигоны (исключаем склад)
    available_polygons: List[int] = [row['MpId'] for row in polygons_df.iter_rows(named=True) if row['MpId'] != warehouse_id]

    progress = True
    while progress and available_polygons:
        progress = False
        # Курьер с минимальным временем
        courier_id = min(courier_state.keys(), key=lambda c: courier_state[c]['time'])
        current_pos = courier_state[courier_id]['position']
        current_time = courier_state[courier_id]['time']

        # Оцениваем top-K полигонов по utility
        scored: List[tuple] = []
        for pid in available_polygons:
            svc = courier_service_times.get(courier_id, {}).get(pid, 0)
            try:
                cost = distance_provider.get_polygon_access_cost(current_pos, pid, svc)
            except Exception:
                cost = float('inf')
            if cost >= float('inf'):
                continue
            if current_time + cost > max_time_per_courier:
                continue
            orders = order_count_map.get(pid, 1)
            utility = orders / (cost + 1)
            scored.append((pid, utility, cost))

        if not scored:
            # Этот курьер не может ничего взять, попробуем другого
            # Удаляем курьера из рассмотрения, если он достиг лимита
            done = all(courier_state[c]['time'] >= max_time_per_courier for c in courier_state)
            if done:
                break
            # Иначе просто на следующей итерации возьмем другого курьера
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0]
        pid, _, cost = best

        # Обновляем состояние курьера
        try:
            if hasattr(distance_provider, 'find_best_port_to_polygon'):
                best_port, _ = distance_provider.find_best_port_to_polygon(current_pos, pid)
                courier_state[courier_id]['position'] = best_port if best_port else distance_provider.polygon_info.get(pid, {}).get('portal_id', current_pos)
            else:
                courier_state[courier_id]['position'] = distance_provider.polygon_info.get(pid, {}).get('portal_id', current_pos)
        except Exception:
            pass

        courier_state[courier_id]['time'] += int(cost)
        courier_state[courier_id]['assigned'].append(pid)
        available_polygons.remove(pid)
        progress = True

    assignment = {cid: state['assigned'] for cid, state in courier_state.items()}
    total_assigned = sum(len(v) for v in assignment.values())
    active_couriers = sum(1 for v in assignment.values() if v)
    logger.info(f"Greedy warm-start: назначено {total_assigned} полигонов, активных курьеров {active_couriers}")
    return assignment


