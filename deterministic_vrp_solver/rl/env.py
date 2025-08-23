import logging
from functools import lru_cache
from typing import Dict, List, Tuple
from .reward import calculate_reward
from .encoder import EnvironmentState
from .protocols import DistanceProviderProtocol


logger = logging.getLogger(__name__)


class CourierState:
    def __init__(self, courier_id: int, position: int):
        self.courier_id = courier_id
        self.current_position = position
        self.current_time = 0
        self.assigned_polygons: List[int] = []
        self.total_distance = 0


class PolygonInfo:
    def __init__(self, polygon_id: int, portal_id: int, total_distance: int, order_count: int):
        self.polygon_id = polygon_id
        self.portal_id = portal_id
        self.total_distance = total_distance
        self.order_count = order_count


class VRPEnvironment:
    def __init__(self, polygons_df, couriers_df, max_time_per_courier: int, distance_provider: DistanceProviderProtocol, courier_service_times: Dict):
        self.polygons_df = polygons_df
        self.couriers_df = couriers_df
        self.max_time_per_courier = max_time_per_courier
        self.distance_provider = distance_provider
        self.courier_service_times = courier_service_times or {}
        self.warehouse_id = 0
        self.polygon_info: Dict[int, PolygonInfo] = {}
        self.cost_cache: Dict[Tuple[int, int, int], float] = {}
        self.top_k_actions = 10
        self.candidate_pool_size = 200
        self._static_utility_sorted: List[int] = []  # предсортированный список polygon_id по статической полезности
        for row in polygons_df.iter_rows(named=True):
            mp = row['MpId']
            self.polygon_info[mp] = PolygonInfo(
                polygon_id=mp,
                portal_id=row.get('portal_id', 0),
                total_distance=row.get('total_distance', 0),
                order_count=row.get('order_count', 0) if hasattr(row, 'get') else (row['order_count'] if 'order_count' in row else 0),
            )
        # reset вызывается извне (в шедулере), чтобы не делать побочных эффектов в конструкторе

    def reset(self):
        self.courier_states: Dict[int, CourierState] = {}
        for courier_id in range(len(self.couriers_df)):
            self.courier_states[courier_id] = CourierState(courier_id, self.warehouse_id)
        self.available_polygons: List[int] = [row['MpId'] for row in self.polygons_df.iter_rows(named=True) if row['MpId'] != 0]
        self.current_time = 0
        # Не очищаем cost_cache между эпизодами — сохраняем кэш для ускорения
        # Предрасчёт статической полезности (склад -> полигон)
        static_scores: List[Tuple[int, float]] = []
        for pid in self.available_polygons:
            try:
                base_cost = self.distance_provider.get_polygon_access_cost(self.warehouse_id, pid, 0)
            except Exception:
                base_cost = float('inf')
            orders = self._get_polygon_info(pid).order_count or 1
            util = orders / (base_cost + 1.0) if base_cost < float('inf') else 0.0
            static_scores.append((pid, util))
        static_scores.sort(key=lambda x: x[1], reverse=True)
        self._static_utility_sorted = [pid for pid, _ in static_scores]
        # Инвентаризация для отладки
        logger.debug(f"reset(): доступных полигонов={len(self.available_polygons)}; первые={self.available_polygons[:10]}")

    def _get_polygon_info(self, polygon_id: int) -> PolygonInfo:
        """Безопасный доступ к информации полигона.

        Поддерживает случаи, когда по ошибке self.polygon_info может быть не словарём.
        """
        try:
            if isinstance(self.polygon_info, dict):
                return self.polygon_info.get(polygon_id, PolygonInfo(polygon_id, 0, 0, 0))
            # Если вдруг пришёл одиночный объект
            if isinstance(self.polygon_info, PolygonInfo):
                return self.polygon_info
        except Exception:
            pass
        return PolygonInfo(polygon_id, 0, 0, 0)

    def get_available_actions(self, courier_id: int) -> List[int]:
        courier = self.courier_states[courier_id]
        actions: List[int] = []
        # ограничим кандидатов топом по статической полезности
        candidates = []
        if self._static_utility_sorted:
            pool = self._static_utility_sorted[: min(self.candidate_pool_size, len(self._static_utility_sorted))]
            # оставим только ещё доступные
            pool_set = set(self.available_polygons)
            candidates = [pid for pid in pool if pid in pool_set]
        # если по каким-то причинам нет кандидатов — вернёмся к полному перебору
        scan_source = candidates if candidates else self.available_polygons
        for pid in scan_source:
            if self._can_assign_polygon(courier, pid):
                actions.append(pid)
        if actions:
            scored: List[Tuple[int, float]] = []
            for pid in actions:
                time_cost = self._calculate_polygon_total_time(courier.courier_id, pid, courier.current_position)
                orders = self._get_polygon_info(pid).order_count or 1
                utility = orders / (time_cost + 1)
                scored.append((pid, utility))
            scored.sort(key=lambda x: x[1], reverse=True)
            actions = [pid for pid, _ in scored[: self.top_k_actions]]
        return actions

    def _can_assign_polygon(self, courier: CourierState, polygon_id: int) -> bool:
        total_time = self._calculate_polygon_total_time(courier.courier_id, polygon_id, courier.current_position)
        if total_time >= float('inf'):
            return False
        return courier.current_time + total_time <= self.max_time_per_courier

    @lru_cache(maxsize=10000)
    def _calculate_polygon_total_time_cached(self, courier_id: int, polygon_id: int, from_position: int) -> float:
        svc = 0
        if courier_id in self.courier_service_times and polygon_id in self.courier_service_times[courier_id]:
            svc = self.courier_service_times[courier_id][polygon_id]
        total_cost = self.distance_provider.get_polygon_access_cost(from_position, polygon_id, svc)
        if total_cost >= float('inf'):
            return float('inf')
        return float(total_cost)

    def _calculate_polygon_total_time(self, courier_id: int, polygon_id: int, from_position: int) -> float:
        key = (courier_id, polygon_id, from_position)
        if key in self.cost_cache:
            return self.cost_cache[key]
        res = self._calculate_polygon_total_time_cached(courier_id, polygon_id, from_position)
        self.cost_cache[key] = res
        return res

    def execute_action(self, courier_id: int, polygon_id: int) -> Tuple[float, bool]:
        courier = self.courier_states[courier_id]
        poly = self._get_polygon_info(polygon_id)
        total_time = self._calculate_polygon_total_time(courier.courier_id, polygon_id, courier.current_position)
        if total_time >= float('inf'):
            return -1000.0, False
        best_port, _ = self.distance_provider.find_best_port_to_polygon(courier.current_position, polygon_id)
        courier.current_position = best_port if best_port else poly.portal_id
        courier.current_time += total_time
        courier.assigned_polygons.append(polygon_id)
        courier.total_distance += total_time
        self.available_polygons.remove(polygon_id)
        can_continue = courier.current_time < self.max_time_per_courier
        all_times = [c.current_time for c in self.courier_states.values()]
        orders = getattr(poly, 'order_count', 0) or 0
        reward = calculate_reward(
            courier_time=courier.current_time,
            total_time=total_time,
            max_time=self.max_time_per_courier,
            orders=orders,
            all_courier_times=all_times,
            remaining_polygons=len(self.available_polygons),
            violations=sum(1 for c in self.courier_states.values() if c.current_time > self.max_time_per_courier),
        )
        return reward, not can_continue

    def is_episode_finished(self) -> bool:
        if not self.available_polygons:
            return True
        return all(c.current_time >= self.max_time_per_courier for c in self.courier_states.values())

    def get_environment_state(self) -> EnvironmentState:
        return EnvironmentState(
            courier_states=self.courier_states.copy(),
            available_polygons=self.available_polygons.copy(),
            polygon_info=self.polygon_info,
            current_time=self.current_time,
            max_time_per_courier=self.max_time_per_courier,
        )

    def calculate_global_objective(self) -> float:
        total_time = sum(c.current_time for c in self.courier_states.values())
        unassigned_penalty = len(self.available_polygons) * 3000
        violations = sum(1 for c in self.courier_states.values() if c.current_time > self.max_time_per_courier)
        violation_penalty = violations * 10000
        return total_time + unassigned_penalty + violation_penalty

    def evaluate_assignment(self, assignment: Dict[int, List[int]]) -> float:
        """Симуляция эпизода по фиксированному назначению полигонов.

        Выполняет действия в порядке `assignment[courier_id]` для каждого курьера,
        используя обычную динамику среды, затем возвращает значение глобальной цели.
        """
        self.reset()
        # Перебор по наименьшему текущему времени для баланса
        progress = True
        # Подготовим итераторы по полигоном на курьера
        pending: Dict[int, List[int]] = {cid: lst.copy() for cid, lst in assignment.items()}
        while progress:
            progress = False
            # активные курьеры
            active = [cid for cid in pending if pending[cid] and self.courier_states[cid].current_time < self.max_time_per_courier]
            if not active:
                break
            active.sort(key=lambda cid: self.courier_states[cid].current_time)
            for cid in active:
                if not pending[cid]:
                    continue
                pid = pending[cid][0]
                if pid not in self.available_polygons:
                    # уже взят другим курьером или не существует
                    pending[cid].pop(0)
                    continue
                # проверим ограничение по времени
                cost = self._calculate_polygon_total_time(cid, pid, self.courier_states[cid].current_position)
                if cost >= float('inf') or self.courier_states[cid].current_time + cost > self.max_time_per_courier:
                    # пропускаем этот полигон
                    pending[cid].pop(0)
                    continue
                # выполним действие
                self.execute_action(cid, pid)
                pending[cid].pop(0)
                progress = True
        return self.calculate_global_objective()


