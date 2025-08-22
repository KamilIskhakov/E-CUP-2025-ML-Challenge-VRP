import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
 
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class CourierState:
    courier_id: int
    current_position: int
    current_time: int
    assigned_polygons: List[int]
    total_distance: int

@dataclass
class PolygonInfo:
    polygon_id: int
    portal_id: int
    total_distance: int
    order_count: int
    service_time: int

@dataclass
class EnvironmentState:
    courier_states: Dict[int, CourierState]
    available_polygons: List[int]
    polygon_info: Dict[int, PolygonInfo]
    current_time: int
    max_time_per_courier: int

class IStateEncoder(ABC):
    @abstractmethod
    def encode_state(self, env_state: EnvironmentState, courier_id: int) -> str:
        pass

class IActionSelector(ABC):
    @abstractmethod
    def select_action(self, state: str, available_actions: List[int], q_table: Dict) -> int:
        pass

class IEnvironment(ABC):
    @abstractmethod
    def get_available_actions(self, courier_id: int) -> List[int]:
        pass
    
    @abstractmethod
    def execute_action(self, courier_id: int, polygon_id: int) -> Tuple[float, bool]:
        pass
    
    @abstractmethod
    def is_episode_finished(self) -> bool:
        pass

class SimpleStateEncoder(IStateEncoder):
    def encode_state(self, env_state: EnvironmentState, courier_id: int) -> str:
        courier = env_state.courier_states[courier_id]
        time_bucket = min(courier.current_time // 7200, 5)
        polygon_bucket = min(len(courier.assigned_polygons), 10)
        available_bucket = min(len(env_state.available_polygons), 20)
        accessible_polygons = 0
        for polygon_id in env_state.available_polygons[:10]:
            polygon = env_state.polygon_info[polygon_id]
            direct_path = True
            if direct_path:
                accessible_polygons += 1
        accessibility_bucket = min(accessible_polygons, 5)
        return f"T{time_bucket}_P{polygon_bucket}_A{available_bucket}_AC{accessibility_bucket}"

class EpsilonGreedyActionSelector(IActionSelector):
    def __init__(self, epsilon: float = 0.1, min_epsilon: float = 0.05, decay: float = 0.995):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
    
    def select_action(self, state: str, available_actions: List[int], q_table: Dict) -> int:
        if not available_actions:
            return -1
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
            return action
        best_action = available_actions[0]
        best_value = q_table.get((state, best_action), 0.0)
        for action in available_actions[1:]:
            value = q_table.get((state, action), 0.0)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

class VRPEnvironment(IEnvironment):
    def __init__(self, polygons_df, couriers_df, max_time_per_courier: int, 
                 distance_provider, courier_service_times: Dict):
        self.polygons_df = polygons_df
        self.couriers_df = couriers_df
        self.max_time_per_courier = max_time_per_courier
        self.distance_provider = distance_provider
        self.courier_service_times = courier_service_times
        self.warehouse_id = 0  # Склад имеет ID=0 в матрице расстояний
        self.polygon_info = {}
        self.cost_cache = {}  # Кэш для ускорения вычислений
        self.top_k_actions = 20
        for row in polygons_df.iter_rows(named=True):
            polygon_id = row['MpId']
            self.polygon_info[polygon_id] = PolygonInfo(
                polygon_id=polygon_id,
                portal_id=row.get('portal_id', 0),
                total_distance=row.get('total_distance', 0),
                order_count=row.get('order_count', 0) if hasattr(row, 'get') else (row['order_count'] if 'order_count' in row else 0),
                service_time=0
            )
        self.reset()
    
    def reset(self):
        self.courier_states = {}
        for courier_id in range(len(self.couriers_df)):
            self.courier_states[courier_id] = CourierState(
                courier_id=courier_id,
                current_position=self.warehouse_id,
                current_time=0,
                assigned_polygons=[],
                total_distance=0
            )
        # Исключаем склад (ID=0) из доступных полигонов
        self.available_polygons = [row['MpId'] for row in self.polygons_df.iter_rows(named=True) if row['MpId'] != 0]
        logger.info(f"Доступных полигонов: {len(self.available_polygons)}")
        logger.info(f"Первые 10 полигонов: {self.available_polygons[:10]}")
        if 0 in self.available_polygons:
            logger.error("ОШИБКА: Склад (ID=0) все еще в доступных полигонах!")
        else:
            logger.info("Склад (ID=0) успешно исключен из доступных полигонов")
        self.current_time = 0
        # Очищаем кэш при сбросе для экономии памяти
        if len(self.cost_cache) > 10000:  # Если кэш стал слишком большим
            self.cost_cache.clear()
    
    def get_available_actions(self, courier_id: int) -> List[int]:
        courier = self.courier_states[courier_id]
        available_actions = []
        for polygon_id in self.available_polygons:
            if self._can_assign_polygon(courier, polygon_id):
                available_actions.append(polygon_id)
        # Shortlist top-K по простому utility: orders / time_cost
        if available_actions:
            scored: List[Tuple[int, float]] = []
            for pid in available_actions:
                time_cost = self._calculate_polygon_total_time(courier.courier_id, pid, courier.current_position)
                orders = self.polygon_info.get(pid, PolygonInfo(pid, 0, 0, 0, 0)).order_count or 1
                utility = orders / (time_cost + 1)
                scored.append((pid, utility))
            scored.sort(key=lambda x: x[1], reverse=True)
            available_actions = [pid for pid, _ in scored[: self.top_k_actions]]

        # Отладочная информация (снижаем уровень логирования)
        if courier_id == 0 and len(available_actions) > 0:
            logger.debug(f"Курьер {courier_id}: доступно {len(available_actions)} действий, первые 5: {available_actions[:5]}")
            if 0 in available_actions:
                logger.error(f"ОШИБКА: Курьер {courier_id} может назначить склад (ID=0)!")
        
        if len(available_actions) == 0 and len(self.available_polygons) > 0:
            logger.debug(f"Курьер {courier_id}: завершает работу (нет доступных полигонов в рамках лимита времени)")
            logger.debug(f"  Текущее время курьера: {courier.current_time}/{self.max_time_per_courier}")
            if len(self.available_polygons) > 0:
                sample_polygon = self.available_polygons[0]
                total_time = self._calculate_polygon_total_time(courier_id, sample_polygon, courier.current_position)
                logger.debug(f"  Пример полигона {sample_polygon}: время {total_time} (превышает лимит)")
                
                # Дополнительная отладка для провайдера портов
                if hasattr(self.distance_provider, 'debug_polygon_access'):
                    polygon_info = self.polygon_info[sample_polygon]
                    self.distance_provider.debug_polygon_access(
                        courier.current_position, sample_polygon, polygon_info.total_distance
                    )
        
        return available_actions
    
    def _can_assign_polygon(self, courier: CourierState, polygon_id: int) -> bool:
        total_time = self._calculate_polygon_total_time(courier.courier_id, polygon_id, courier.current_position)
        
        # Проверяем только время доступа к полигону (без возврата к складу)
        # Возврат к складу будет учтен в конце маршрута
        return courier.current_time + total_time <= self.max_time_per_courier
    
    @lru_cache(maxsize=10000)
    def _calculate_polygon_total_time_cached(self, courier_id: int, polygon_id: int, from_position: int) -> int:
        """Кэшированная версия вычисления времени с использованием lru_cache"""
        polygon = self.polygon_info[polygon_id]
        
        # Получаем сервисное время курьера в данном полигоне
        service_time = 0
        if (courier_id in self.courier_service_times and 
            polygon_id in self.courier_service_times[courier_id]):
            service_time = self.courier_service_times[courier_id][polygon_id]
        
        # Используем декомпозированный провайдер
        if hasattr(self.distance_provider, 'get_polygon_access_cost'):
            total_cost = self.distance_provider.get_polygon_access_cost(
                from_position, polygon_id, service_time
            )
            
            if total_cost >= float('inf'):
                return 999999
            else:
                return int(total_cost)
        else:
            # Старый провайдер (обратная совместимость)
            time_to_portal = self._find_path_to_polygon(from_position, polygon.portal_id)
            if time_to_portal >= 999999:
                return 999999
            else:
                tsp_time = polygon.total_distance
            
            time_from_portal_to_warehouse = self.distance_provider.get_distance(polygon.portal_id, self.warehouse_id)
            if time_from_portal_to_warehouse == 0:
                time_from_portal_to_warehouse = 999999
            
            return time_to_portal + tsp_time + service_time + time_from_portal_to_warehouse
    
    def _calculate_polygon_total_time(self, courier_id: int, polygon_id: int, from_position: int) -> int:
        """Оптимизированная версия с двойным кэшированием"""
        # Проверяем локальный кэш
        cache_key = (courier_id, polygon_id, from_position)
        if cache_key in self.cost_cache:
            return self.cost_cache[cache_key]
        
        # Используем lru_cache для часто используемых значений
        result = self._calculate_polygon_total_time_cached(courier_id, polygon_id, from_position)
        
        # Логируем большие значения для отладки
        if result > 100000:  # Больше 27 часов
            logger.warning(f"Большое время для курьера {courier_id}, полигон {polygon_id}: {result} сек ({result/3600:.1f} ч)")
            logger.warning(f"  from_position: {from_position}")
            logger.warning(f"  courier_service_times: {self.courier_service_times.get(courier_id, {}).get(polygon_id, 0)}")
        
        # Сохраняем в локальный кэш
        self.cost_cache[cache_key] = result
        return result
    
    def _find_path_to_polygon(self, from_position: int, to_portal: int) -> int:
        direct_distance = self.distance_provider.get_distance(from_position, to_portal)
        if direct_distance > 0 and direct_distance < 999999:
            return direct_distance
        
        # Отладочная информация
        if from_position == 0:  # Склад
            logger.info(f"Нет прямого пути от склада {from_position} к порталу {to_portal}")
            logger.info(f"Прямое расстояние: {direct_distance}")
        
        intermediate_points = []
        for poly_info in self.polygon_info.values():
            if poly_info.portal_id != to_portal and poly_info.portal_id != from_position:
                intermediate_points.append(poly_info.portal_id)
        
        if not intermediate_points:
            if from_position == 0:  # Склад
                logger.info(f"Нет промежуточных точек для пути к порталу {to_portal}")
            return 999999
        
        min_path = 999999
        for intermediate in intermediate_points:
            dist1 = self.distance_provider.get_distance(from_position, intermediate)
            if dist1 == 0 or dist1 >= 999999:
                continue
            dist2 = self.distance_provider.get_distance(intermediate, to_portal)
            if dist2 == 0 or dist2 >= 999999:
                continue
            path_length = dist1 + dist2
            if path_length < min_path:
                min_path = path_length
        
        if from_position == 0 and min_path == 999999:  # Склад
            logger.info(f"Не найден путь через промежуточные точки к порталу {to_portal}")
            logger.info(f"Доступно промежуточных точек: {len(intermediate_points)}")
        
        return min_path
    
    def execute_action(self, courier_id: int, polygon_id: int) -> Tuple[float, bool]:
        courier = self.courier_states[courier_id]
        polygon = self.polygon_info[polygon_id]
        
        total_time = self._calculate_polygon_total_time(courier.courier_id, polygon_id, courier.current_position)
        
        # Логируем большие времена для отладки
        if total_time > 50000:  # Больше 13 часов
            logger.warning(f"Курьер {courier_id} назначается на полигон {polygon_id} с временем {total_time} сек ({total_time/3600:.1f} ч)")
            logger.warning(f"  Текущее время курьера: {courier.current_time} сек ({courier.current_time/3600:.1f} ч)")
            logger.warning(f"  Лимит времени: {self.max_time_per_courier} сек ({self.max_time_per_courier/3600:.1f} ч)")
        
        if total_time >= 999999:
            return -1000.0, False
        
        # Находим лучший порт для входа в полигон
        if hasattr(self.distance_provider, 'find_best_port_to_polygon'):
            best_port, _ = self.distance_provider.find_best_port_to_polygon(courier.current_position, polygon_id)
            if best_port:
                courier.current_position = best_port
            else:
                courier.current_position = polygon.portal_id  # fallback
        else:
            courier.current_position = polygon.portal_id
        
        courier.current_time += total_time
        courier.assigned_polygons.append(polygon_id)
        courier.total_distance += total_time
        self.available_polygons.remove(polygon_id)
        
        reward = self._calculate_reward(courier_id, polygon_id, total_time)
        can_continue = courier.current_time < self.max_time_per_courier
        
        return reward, not can_continue
    
    def _calculate_reward(self, courier_id: int, polygon_id: int, total_time: int) -> float:
        # Взвешенный ревард: покрытие − время − штрафы + справедливость
        w_cov = 2.0
        w_time = 1.0 / 1000.0
        courier = self.courier_states[courier_id]
        orders = self.polygon_info.get(polygon_id, PolygonInfo(polygon_id, 0, 0, 0, 0)).order_count or 0
        coverage_bonus = w_cov * orders
        time_penalty = w_time * total_time
        # Штраф за превышение лимита
        time_ratio = courier.current_time / self.max_time_per_courier if self.max_time_per_courier > 0 else 0
        if time_ratio > 1.0:
            over_penalty = -1000.0
        elif time_ratio > 0.95:
            over_penalty = -200.0
        elif time_ratio > 0.85:
            over_penalty = -50.0
        else:
            over_penalty = 0.0
        fairness_bonus = self._calculate_balance_bonus(courier_id)
        return coverage_bonus - time_penalty + over_penalty + fairness_bonus
    
    def _calculate_balance_bonus(self, courier_id: int) -> float:
        courier_times = [courier.current_time for courier in self.courier_states.values()]
        if not courier_times:
            return 0.0
        avg_time = sum(courier_times) / len(courier_times)
        time_diff = abs(self.courier_states[courier_id].current_time - avg_time)
        if time_diff < avg_time * 0.1:
            return 10.0
        elif time_diff < avg_time * 0.2:
            return 5.0
        return 0.0
    
    def is_episode_finished(self) -> bool:
        if not self.available_polygons:
            return True
        active_couriers = 0
        for courier in self.courier_states.values():
            if courier.current_time < self.max_time_per_courier:
                active_couriers += 1
        if active_couriers == 0:
            return True
        return False
    
    def get_environment_state(self) -> EnvironmentState:
        return EnvironmentState(
            courier_states=self.courier_states.copy(),
            available_polygons=self.available_polygons.copy(),
            polygon_info=self.polygon_info,
            current_time=self.current_time,
            max_time_per_courier=self.max_time_per_courier
        )
    
    def calculate_global_objective(self) -> float:
        total_time = sum(courier.current_time for courier in self.courier_states.values())
        unassigned_penalty = len(self.available_polygons) * 3000
        violations = sum(1 for courier in self.courier_states.values() 
                        if courier.current_time > self.max_time_per_courier)
        violation_penalty = violations * 10000
        return total_time + unassigned_penalty + violation_penalty

class QLearningAgent:
    def __init__(self, state_encoder: IStateEncoder, action_selector: IActionSelector,
                 learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.state_encoder = state_encoder
        self.action_selector = action_selector
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
    
    def get_q_value(self, state: str, action: int) -> float:
        return self.q_table.get((state, action), 0.0)
    
    def update_q_value(self, state: str, action: int, reward: float, next_state: str):
        current_q = self.get_q_value(state, action)
        max_next_q = 0.0
        if next_state:
            next_actions = [action for (s, action) in self.q_table.keys() if s == next_state]
            if next_actions:
                max_next_q = max(self.get_q_value(next_state, a) for a in next_actions)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def select_action(self, env_state: EnvironmentState, courier_id: int, environment: IEnvironment) -> int:
        state = self.state_encoder.encode_state(env_state, courier_id)
        available_actions = environment.get_available_actions(courier_id)
        action = self.action_selector.select_action(state, available_actions, self.q_table)
        return action

class ReinforcementScheduler:
    def __init__(self, polygons_df, couriers_df, max_time_per_courier: int,
                 distance_provider, courier_service_times: Dict):
        self.environment = VRPEnvironment(
            polygons_df, couriers_df, max_time_per_courier, 
            distance_provider, courier_service_times
        )
        self.state_encoder = SimpleStateEncoder()
        self.action_selector = EpsilonGreedyActionSelector(epsilon=0.3, min_epsilon=0.05, decay=0.995)
        self.agent = QLearningAgent(self.state_encoder, self.action_selector, learning_rate=0.1, discount_factor=0.9)
    
    def _run_episode(self, episode_num: int, max_steps_per_episode: int) -> Tuple[Dict[int, List[int]], float, int, List[Tuple]]:
        env_copy = VRPEnvironment(
            self.environment.polygons_df, 
            self.environment.couriers_df, 
            self.environment.max_time_per_courier,
            self.environment.distance_provider,
            self.environment.courier_service_times
        )
        env_copy.reset()
        episode_reward = 0
        step_count = 0
        experience_buffer = []
        
        while not env_copy.is_episode_finished() and step_count < max_steps_per_episode:
            step_count += 1
            active_couriers = []
            for courier_id, courier in env_copy.courier_states.items():
                if courier.current_time < env_copy.max_time_per_courier:
                    available_actions = env_copy.get_available_actions(courier_id)
                    if available_actions:
                        active_couriers.append(courier_id)
            
            if not active_couriers:
                break
            
            # Выбираем наименее загруженного курьера (по текущему времени)
            courier_id = min(active_couriers, key=lambda cid: env_copy.courier_states[cid].current_time)
            env_state = env_copy.get_environment_state()
            current_state = self.state_encoder.encode_state(env_state, courier_id)
            available_actions = env_copy.get_available_actions(courier_id)
            action = self.action_selector.select_action(current_state, available_actions, self.agent.q_table)
            
            if action == -1:
                continue
            
            reward, done = env_copy.execute_action(courier_id, action)
            episode_reward += reward
            next_env_state = env_copy.get_environment_state()
            next_state = self.state_encoder.encode_state(next_env_state, courier_id)
            experience_buffer.append((current_state, action, reward, next_state))
        
        assignment = {
            courier_id: courier.assigned_polygons.copy()
            for courier_id, courier in env_copy.courier_states.items()
        }
        
        global_objective = env_copy.calculate_global_objective()
        
        return assignment, global_objective, step_count, experience_buffer
    
    def train(self, episodes: int = 100, max_steps_per_episode: int = 2000) -> Dict[int, List[int]]:
        # Используем оптимизированное последовательное обучение для стабильности
        logger.info(f"Используем оптимизированное последовательное обучение")
        return self._train_sequential_optimized(episodes, max_steps_per_episode)
    
    def _train_sequential(self, episodes: int, max_steps_per_episode: int) -> Dict[int, List[int]]:
        """Последовательное обучение для небольшого количества эпизодов"""
        best_assignment = None
        best_objective = float('inf')
        no_improvement_count = 0
        early_stop_threshold = 25  # Менее агрессивная остановка для лучшего обучения
        
        logger.info(f"Последовательное обучение: {episodes} эпизодов")
        
        for episode in range(episodes):
            try:
                assignment, global_objective, step_count, experience_buffer = self._run_episode(episode, max_steps_per_episode)
                
                for current_state, action, reward, next_state in experience_buffer:
                    self.agent.update_q_value(current_state, action, reward, next_state)
                
                total_assigned = sum(len(polygons) for polygons in assignment.values())
                active_couriers = sum(1 for polygons in assignment.values() if polygons)
                
                if global_objective < best_objective:
                    best_objective = global_objective
                    best_assignment = assignment
                    no_improvement_count = 0
                    logger.info(f"Новый лучший результат: {global_objective:.0f} (эпизод {episode + 1})")
                else:
                    no_improvement_count += 1
                
                if (episode + 1) % 100 == 0:  # Реже логируем для максимального ускорения
                    logger.info(f"Эпизод {episode + 1}: назначено {total_assigned} полигонов, активных курьеров {active_couriers}, цель {global_objective:.0f}")
                
                if no_improvement_count >= early_stop_threshold:
                    logger.info(f"Ранняя остановка: нет улучшений {early_stop_threshold} эпизодов подряд")
                    break
                
            except Exception as e:
                logger.error(f"Ошибка в эпизоде {episode + 1}: {e}")
        
        logger.info(f"Последовательное обучение завершено. Лучшая цель: {best_objective:.0f}")
        
        if best_assignment is None:
            return {}
        
        return best_assignment
    
    def _train_sequential_optimized(self, episodes: int, max_steps_per_episode: int) -> Dict[int, List[int]]:
        """Оптимизированное последовательное обучение с максимальным использованием CPU"""
        best_assignment = None
        best_objective = float('inf')
        no_improvement_count = 0
        early_stop_threshold = 30  # Увеличиваем для лучшего обучения
        
        logger.info(f"Оптимизированное последовательное обучение: {episodes} эпизодов")
        
        # Предварительная загрузка всех расстояний в память
        logger.info("Предварительная загрузка расстояний в память...")
        self._preload_distances()
        
        # Векторизованные вычисления
        logger.info("Инициализация векторизованных вычислений...")
        self._init_vectorized_computations()
        
        for episode in range(episodes):
            try:
                assignment, global_objective, step_count, experience_buffer, stats = self._run_episode_optimized(episode, max_steps_per_episode)
                
                # Батчевое обновление Q-таблицы
                self._batch_update_q_values(experience_buffer)
                
                # Плавное уменьшение epsilon
                if hasattr(self.action_selector, 'decay_epsilon'):
                    self.action_selector.decay_epsilon()
                
                total_assigned = sum(len(polygons) for polygons in assignment.values())
                active_couriers = sum(1 for polygons in assignment.values() if polygons)
                
                # Сохраняем метрики
                if not hasattr(self, 'training_metrics'):
                    self.training_metrics = []
                self.training_metrics.append({
                    'episode': stats.get('episode', episode),
                    'objective': float(global_objective),
                    'epsilon': float(getattr(self.action_selector, 'epsilon', 0.0)),
                    'total_assigned': int(stats.get('total_assigned', total_assigned)),
                    'active_couriers': int(stats.get('active_couriers', active_couriers)),
                    'avg_courier_time': float(stats.get('avg_courier_time', 0)),
                    'max_courier_time': float(stats.get('max_courier_time', 0)),
                    'remaining_polygons': int(stats.get('remaining_polygons', 0))
                })
                
                if global_objective < best_objective:
                    best_objective = global_objective
                    best_assignment = assignment
                    no_improvement_count = 0
                    logger.info(f"Новый лучший результат: {global_objective:.0f} (эпизод {episode + 1})")
                else:
                    no_improvement_count += 1
                
                if (episode + 1) % 20 == 0:  # Частое логирование для мониторинга
                    logger.info(f"Эпизод {episode + 1}: назначено {total_assigned} полигонов, активных курьеров {active_couriers}, цель {global_objective:.0f}")
                
                if no_improvement_count >= early_stop_threshold:
                    logger.info(f"Ранняя остановка: нет улучшений {early_stop_threshold} эпизодов подряд")
                    break
                
            except Exception as e:
                logger.error(f"Ошибка в эпизоде {episode + 1}: {e}")
        
        logger.info(f"Оптимизированное последовательное обучение завершено. Лучшая цель: {best_objective:.0f}")
        
        if best_assignment is None:
            return {}
        
        # Сохраняем метрики в файлы
        try:
            import polars as pl
            df = pl.DataFrame(self.training_metrics)
            df.write_parquet('rl_training_metrics.parquet', compression='zstd')
            df.write_csv('rl_training_metrics.csv')
            logger.info("Метрики обучения сохранены: rl_training_metrics.parquet, rl_training_metrics.csv")
        except Exception as e:
            logger.error(f"Не удалось сохранить метрики обучения: {e}")
        
        return best_assignment
    
    # Удалено: нестабильные многопроцессные тренировки с SQLite
    
    # Удалено: многопоточное обучение с SQLite (нестабильно)
    
    # Удалено: параллельное обучение (нестабильно с SQLite и локальными функциями)
    
    # Удалено: параллельный вариант обучения (см. причины выше)
    
    def _train_batch_worker(self, episodes: int, max_steps_per_episode: int, worker_id: int,
                           polygons_data: List[Dict], couriers_data: List[Dict], 
                           polygon_info: Dict, courier_service_times: Dict) -> Dict:
        """Рабочая функция для параллельного обучения"""
        # Создаем новые соединения с базой данных для этого процесса
        import sqlite3
        conn = sqlite3.connect('durations.sqlite')
        
        # Создаем DataFrame из переданных данных
        import polars as pl
        polygons_df = pl.DataFrame(polygons_data)
        couriers_df = pl.DataFrame(couriers_data)
        
        # Создаем новый провайдер расстояний для этого процесса
        from decomposed_distance_provider import DecomposedDistanceProvider
        distance_provider = DecomposedDistanceProvider(
            conn, polygon_info, courier_service_times
        )
        
        # Создаем локальную копию окружения для этого процесса
        env_copy = VRPEnvironment(
            polygons_df, couriers_df, self.max_time_per_courier,
            distance_provider, courier_service_times
        )
        
        agent_copy = QLearningAgent(learning_rate=0.1, discount_factor=0.9)
        state_encoder = SimpleStateEncoder()
        action_selector = EpsilonGreedyActionSelector(epsilon=0.1)
        
        best_assignment = None
        best_objective = float('inf')
        
        logger.info(f"Процесс {worker_id}: начинаем {episodes} эпизодов")
        
        for episode in range(episodes):
            try:
                assignment, global_objective, step_count, experience_buffer = self._run_episode_worker(
                    env_copy, agent_copy, state_encoder, action_selector, episode, max_steps_per_episode
                )
                
                # Обновляем Q-таблицу
                for current_state, action, reward, next_state in experience_buffer:
                    agent_copy.update_q_value(current_state, action, reward, next_state)
                
                if global_objective < best_objective:
                    best_objective = global_objective
                    best_assignment = assignment
                
                if (episode + 1) % 10 == 0:
                    total_assigned = sum(len(polygons) for polygons in assignment.values())
                    active_couriers = sum(1 for polygons in assignment.values() if polygons)
                    logger.info(f"Процесс {worker_id}, эпизод {episode + 1}: назначено {total_assigned} полигонов, цель {global_objective:.0f}")
                
            except Exception as e:
                logger.error(f"Ошибка в процессе {worker_id}, эпизод {episode + 1}: {e}")
        
        logger.info(f"Процесс {worker_id} завершен. Лучшая цель: {best_objective:.0f}")
        
        return {
            'assignment': best_assignment,
            'objective': best_objective,
            'worker_id': worker_id
        }
    
    def _run_episode_worker(self, env_copy, agent_copy, state_encoder, action_selector, episode, max_steps_per_episode):
        """Запуск эпизода для рабочего процесса"""
        env_copy.reset()
        episode_reward = 0
        step_count = 0
        experience_buffer = []
        
        while not env_copy.is_episode_finished() and step_count < max_steps_per_episode:
            step_count += 1
            active_couriers = []
            for courier_id, courier in env_copy.courier_states.items():
                if courier.current_time < env_copy.max_time_per_courier:
                    available_actions = env_copy.get_available_actions(courier_id)
                    if available_actions:
                        active_couriers.append(courier_id)
            
            if not active_couriers:
                break
            
            courier_id = random.choice(active_couriers)
            env_state = env_copy.get_environment_state()
            current_state = state_encoder.encode_state(env_state, courier_id)
            available_actions = env_copy.get_available_actions(courier_id)
            action = action_selector.select_action(current_state, available_actions, agent_copy.q_table)
            
            if action == -1:
                continue
            
            reward, done = env_copy.execute_action(courier_id, action)
            episode_reward += reward
            next_env_state = env_copy.get_environment_state()
            next_state = state_encoder.encode_state(next_env_state, courier_id)
            experience_buffer.append((current_state, action, reward, next_state))
        
        assignment = {
            courier_id: courier.assigned_polygons.copy()
            for courier_id, courier in env_copy.courier_states.items()
        }
        
        global_objective = env_copy.calculate_global_objective()
        
        return assignment, global_objective, step_count, experience_buffer
    
    def solve(self, polygons_df, couriers_df, max_time_per_courier: int) -> Dict[int, List[int]]:
        logger.info("Начинаем решение VRP с обучением с подкреплением")
        
        total_polygons = len(polygons_df)
        total_couriers = len(couriers_df)
        
        # Агрессивные параметры для максимального использования CPU
        episodes = min(500, max(100, total_polygons // 50))  # Больше эпизодов для параллелизации
        max_steps = min(5000, max(2000, total_polygons * 2))  # Больше шагов для лучшего обучения
        
        logger.info(f"Параметры обучения: {episodes} эпизодов, {max_steps} шагов на эпизод")
        logger.info(f"Всего полигонов: {total_polygons}, курьеров: {total_couriers}")
        
        # Сохраняем данные для передачи в параллельные процессы
        self.polygons_df = polygons_df
        self.couriers_df = couriers_df
        self.max_time_per_courier = max_time_per_courier
        
        # Получаем polygon_info и courier_service_times из distance_provider окружения
        dp = getattr(self.environment, 'distance_provider', None)
        if dp and hasattr(dp, 'polygon_info'):
            self.polygon_info = dp.polygon_info
        else:
            self.polygon_info = {}
        
        if dp and hasattr(dp, 'courier_service_times'):
            self.courier_service_times = dp.courier_service_times
        else:
            self.courier_service_times = {}
        
        # Greedy warm-start
        try:
            from .greedy_initializer import greedy_initialize_assignment
        except Exception:
            from greedy_initializer import greedy_initialize_assignment
        warm_assignment = greedy_initialize_assignment(
            polygons_df, couriers_df, max_time_per_courier,
            self.environment.distance_provider,
            getattr(self.environment, 'courier_service_times', {}),
            warehouse_id=self.environment.warehouse_id,
            top_k=50,
        )
        # Можно интегрировать warm_assignment в начальные Q-значения (простым бустом)
        for cid, pids in warm_assignment.items():
            state = self.state_encoder.encode_state(self.environment.get_environment_state(), cid)
            for pid in pids:
                self.agent.q_table[(state, pid)] = self.agent.q_table.get((state, pid), 0.0) + 1.0

        assignment = self.train(episodes=episodes, max_steps_per_episode=max_steps)
        
        total_assigned = sum(len(polygons) for polygons in assignment.values())
        active_couriers = sum(1 for polygons in assignment.values() if polygons)
        
        logger.info(f"Результат: назначено {total_assigned} полигонов, активных курьеров {active_couriers}")
        
        return assignment
    
    def _preload_distances(self):
        """Предварительная загрузка всех расстояний в память для ускорения"""
        logger.info("Начинаем предварительную загрузку расстояний...")
        
        # Загружаем все расстояния между полигонами
        polygon_ids = list(self.environment.polygon_info.keys())
        warehouse_id = self.environment.warehouse_id
        
        # Создаем матрицу расстояний в памяти
        self.distance_matrix = {}
        
        for i, polygon_id in enumerate(polygon_ids):
            logger.info(f"Загружаем расстояния для полигона {polygon_id} ({i+1}/{len(polygon_ids)})")
            
            # Расстояния от склада
            if hasattr(self.environment.distance_provider, 'get_polygon_access_cost'):
                cost = self.environment.distance_provider.get_polygon_access_cost(
                    warehouse_id, polygon_id, 0
                )
                self.distance_matrix[(warehouse_id, polygon_id)] = cost
            
            # Расстояния между полигонами
            for j, other_polygon_id in enumerate(polygon_ids):
                if i != j:
                    if hasattr(self.environment.distance_provider, 'get_polygon_access_cost'):
                        cost = self.environment.distance_provider.get_polygon_access_cost(
                            polygon_id, other_polygon_id, 0
                        )
                        self.distance_matrix[(polygon_id, other_polygon_id)] = cost
        
        logger.info(f"Загружено {len(self.distance_matrix)} расстояний в память")
    
    def _init_vectorized_computations(self):
        """Инициализация векторизованных вычислений"""
        logger.info("Инициализация векторизованных вычислений...")
        
        # Создаем numpy массивы для быстрых вычислений
        polygon_ids = list(self.environment.polygon_info.keys())
        courier_ids = list(self.environment.courier_states.keys())
        
        self.polygon_ids_array = np.array(polygon_ids)
        self.courier_ids_array = np.array(courier_ids)
        
        # Создаем матрицу состояний
        self.state_matrix = np.zeros((len(courier_ids), len(polygon_ids)), dtype=np.float32)
        
        logger.info(f"Создана матрица состояний: {self.state_matrix.shape}")
    
    def _run_episode_optimized(self, episode_num: int, max_steps_per_episode: int) -> Tuple[Dict[int, List[int]], float, int, List[Tuple], Dict[str, Any]]:
        """Оптимизированная версия запуска эпизода"""
        env_copy = VRPEnvironment(
            self.environment.polygons_df, 
            self.environment.couriers_df, 
            self.environment.max_time_per_courier,
            self.environment.distance_provider,
            self.environment.courier_service_times
        )
        env_copy.reset()
        episode_reward = 0
        step_count = 0
        experience_buffer = []
        
        # Используем векторизованные вычисления
        while not env_copy.is_episode_finished() and step_count < max_steps_per_episode:
            step_count += 1
            
            # Быстрое определение активных курьеров
            active_couriers = [
                courier_id for courier_id, courier in env_copy.courier_states.items()
                if courier.current_time < env_copy.max_time_per_courier
            ]
            
            if not active_couriers:
                break
            
            # Выбираем курьера с минимальной загрузкой (по текущему времени)
            courier_id = min(active_couriers, key=lambda cid: env_copy.courier_states[cid].current_time)
            
            env_state = env_copy.get_environment_state()
            current_state = self.state_encoder.encode_state(env_state, courier_id)
            available_actions = env_copy.get_available_actions(courier_id)
            
            if not available_actions:
                continue
            
            action = self.action_selector.select_action(current_state, available_actions, self.agent.q_table)
            
            if action == -1:
                continue
            
            reward, done = env_copy.execute_action(courier_id, action)
            episode_reward += reward
            next_env_state = env_copy.get_environment_state()
            next_state = self.state_encoder.encode_state(next_env_state, courier_id)
            experience_buffer.append((current_state, action, reward, next_state))
        
        assignment = {
            courier_id: courier.assigned_polygons.copy()
            for courier_id, courier in env_copy.courier_states.items()
        }
        
        global_objective = env_copy.calculate_global_objective()
        
        courier_times = [c.current_time for c in env_copy.courier_states.values()]
        stats = {
            'episode': episode_num,
            'courier_times': courier_times,
            'avg_courier_time': (sum(courier_times)/len(courier_times)) if courier_times else 0,
            'max_courier_time': max(courier_times) if courier_times else 0,
            'total_assigned': sum(len(v) for v in assignment.values()),
            'active_couriers': sum(1 for t in courier_times if t > 0),
            'remaining_polygons': len(env_copy.available_polygons)
        }
        return assignment, global_objective, step_count, experience_buffer, stats
    
    def _batch_update_q_values(self, experience_buffer: List[Tuple]):
        """Батчевое обновление Q-таблицы для ускорения"""
        if not experience_buffer:
            return
        
        # Группируем опыт по состояниям для батчевого обновления
        state_action_updates = {}
        
        for current_state, action, reward, next_state in experience_buffer:
            key = (current_state, action)
            if key not in state_action_updates:
                state_action_updates[key] = []
            state_action_updates[key].append((reward, next_state))
        
        # Батчевое обновление
        for (state, action), experiences in state_action_updates.items():
            avg_reward = sum(exp[0] for exp in experiences) / len(experiences)
            next_states = [exp[1] for exp in experiences]
            
            # Обновляем Q-значение
            self.agent.update_q_value(state, action, avg_reward, next_states[0])  # Берем первый next_state
    
    def _prepare_process_data(self) -> Dict:
        """Подготовка данных для многопроцессного обучения"""
        logger.info("Подготовка данных для многопроцессного обучения...")
        
        # Сериализуем данные для передачи в процессы
        process_data = {
            'polygons_df': self.environment.polygons_df.to_dicts(),
            'couriers_df': self.environment.couriers_df.to_dicts(),
            'max_time_per_courier': self.environment.max_time_per_courier,
            'polygon_info': self.environment.polygon_info,
            'courier_service_times': self.environment.courier_service_times,
            'warehouse_id': self.environment.warehouse_id
        }
        
        logger.info("Данные подготовлены для многопроцессного обучения")
        return process_data
    
    def _train_batch_worker_optimized(self, batch_episodes: int, max_steps_per_episode: int, 
                                    process_data: Dict, process_id: int) -> Dict:
        """Оптимизированный рабочий процесс для многопроцессного обучения"""
        logger.info(f"Запуск оптимизированного процесса {process_id}: {batch_episodes} эпизодов")
        
        # Восстанавливаем данные в процессе
        import polars as pl
        
        polygons_df = pl.DataFrame(process_data['polygons_df'])
        couriers_df = pl.DataFrame(process_data['couriers_df'])
        max_time_per_courier = process_data['max_time_per_courier']
        polygon_info = process_data['polygon_info']
        courier_service_times = process_data['courier_service_times']
        warehouse_id = process_data['warehouse_id']
        
        # Создаем новые соединения с БД для процесса
        from decomposed_distance_provider import DecomposedDistanceProvider
        
        distance_provider = DecomposedDistanceProvider(
            durations_db_path=self.environment.distance_provider.durations_db_path,
            ports_db_path=self.environment.distance_provider.ports_db_path,
            warehouse_ports_db_path=self.environment.distance_provider.warehouse_ports_db_path
        )
        distance_provider.__enter__()
        distance_provider.set_polygon_info(polygon_info)
        
        # Создаем окружение для процесса
        env = VRPEnvironment(
            polygons_df, couriers_df, max_time_per_courier, 
            distance_provider, courier_service_times
        )
        
        # Создаем агента и другие компоненты
        state_encoder = SimpleStateEncoder()
        action_selector = EpsilonGreedyActionSelector(epsilon=0.1)
        agent = QLearningAgent(state_encoder, action_selector, learning_rate=0.1, discount_factor=0.95)
        
        best_assignment = None
        best_objective = float('inf')
        
        for episode in range(batch_episodes):
            try:
                assignment, global_objective, step_count, experience_buffer = self._run_episode_worker(
                    env, agent, state_encoder, action_selector, episode, max_steps_per_episode
                )
                
                # Обновляем Q-таблицу
                for current_state, action, reward, next_state in experience_buffer:
                    agent.update_q_value(current_state, action, reward, next_state)
                
                if global_objective < best_objective:
                    best_objective = global_objective
                    best_assignment = assignment
                
                if (episode + 1) % 10 == 0:
                    total_assigned = sum(len(polygons) for polygons in assignment.values())
                    active_couriers = sum(1 for polygons in assignment.values() if polygons)
                    logger.info(f"Процесс {process_id}, эпизод {episode + 1}: назначено {total_assigned} полигонов, цель {global_objective:.0f}")
                
            except Exception as e:
                logger.error(f"Ошибка в процессе {process_id}, эпизод {episode + 1}: {e}")
        
        # Закрываем соединения
        distance_provider.close()
        
        logger.info(f"Процесс {process_id} завершен. Лучшая цель: {best_objective:.0f}")
        
        return {
            'episodes': batch_episodes,
            'best_assignment': best_assignment,
            'best_objective': best_objective,
            'process_id': process_id
        }
