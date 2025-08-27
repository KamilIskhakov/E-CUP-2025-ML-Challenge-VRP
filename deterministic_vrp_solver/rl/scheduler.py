import logging
from typing import Dict, List, Tuple, Any
import numpy as np

from .encoder import SimpleStateEncoder, EnvironmentState
from .selector import EpsilonGreedyActionSelector
from .agent import QLearningAgent
from .warm_start import greedy_initialize_assignment
from .env import VRPEnvironment
from .metrics import MetricsRecorder
from .trainer import RLTrainer


logger = logging.getLogger(__name__)


class ReinforcementScheduler:
    def __init__(self, polygons_df, couriers_df, max_time_per_courier: int,
                 distance_provider, courier_service_times: Dict, use_parallel: bool = False,
                 num_workers: int = 2):
        self.environment = VRPEnvironment(
            polygons_df, couriers_df, max_time_per_courier,
            distance_provider, courier_service_times
        )
                                                       
        try:
            wh_id = int(getattr(distance_provider, 'warehouse_id', 0))
            self.environment.warehouse_id = wh_id
        except Exception:
            pass
                              
        self.environment.reset()
        self.use_parallel = use_parallel
        self.num_workers = num_workers
        self.state_encoder = SimpleStateEncoder()
        self.action_selector = EpsilonGreedyActionSelector(epsilon=0.3, min_epsilon=0.05, decay=0.995)
        self.agent = QLearningAgent(self.state_encoder, self.action_selector, learning_rate=0.1, discount_factor=0.9)
        self.trainer = RLTrainer(self.environment)

    def _run_episode(self, episode_num: int, max_steps_per_episode: int) -> Tuple[Dict[int, List[int]], float, int, List[Tuple]]:
        env_copy = VRPEnvironment(
            self.environment.polygons_df,
            self.environment.couriers_df,
            self.environment.max_time_per_courier,
            self.environment.distance_provider,
            self.environment.courier_service_times
        )
                                                
        try:
            env_copy.warehouse_id = int(getattr(self.environment, 'warehouse_id', 0))
        except Exception:
            pass
        env_copy.reset()
        episode_reward = 0
        step_count = 0
        experience_buffer: List[Tuple[str, int, float, str]] = []

        while not env_copy.is_episode_finished() and step_count < max_steps_per_episode:
            step_count += 1
            active_couriers = [cid for cid, c in env_copy.courier_states.items() if c.current_time < env_copy.max_time_per_courier and env_copy.get_available_actions(cid)]
            if not active_couriers:
                break
            courier_id = min(active_couriers, key=lambda cid: env_copy.courier_states[cid].current_time)
            result = self.agent.act(env_copy, courier_id)
            if result is None:
                continue
            current_state, action, reward, next_state = result
            episode_reward += reward
            experience_buffer.append((current_state, action, reward, next_state))

        assignment = {cid: c.assigned_polygons.copy() for cid, c in env_copy.courier_states.items()}
        global_objective = env_copy.calculate_global_objective()
        return assignment, global_objective, step_count, experience_buffer

    def train(self, episodes: int = 100, max_steps_per_episode: int = 2000) -> Dict[int, List[int]]:
        if self.use_parallel and episodes >= self.num_workers * 2:
            logger.info("Запуск параллельного обучения (без разделения sqlite-соединений)")
            return self._train_parallel(episodes, max_steps_per_episode)
        logger.info("Используем оптимизированное последовательное обучение")
        return self._train_sequential_optimized(episodes, max_steps_per_episode)

    def _train_sequential_optimized(self, episodes: int, max_steps_per_episode: int) -> Dict[int, List[int]]:
        best_assignment = None
        best_objective = float('inf')
        no_improvement_count = 0
        early_stop_threshold = 30

        logger.info(f"Оптимизированное последовательное обучение: {episodes} эпизодов")

        logger.debug("Пропускаем предварительную загрузку расстояний (используем SQLite провайдер)")

        logger.info("Инициализация векторизованных вычислений...")
        self._init_vectorized_computations()

        self.metrics = MetricsRecorder()

        for episode in range(episodes):
            try:
                assignment, global_objective, step_count, experience_buffer, stats = self.trainer._run_episode_optimized(episode, max_steps_per_episode)

                self.trainer._batch_update_q_values(experience_buffer)

                if hasattr(self.action_selector, 'decay_epsilon'):
                    self.action_selector.decay_epsilon()

                total_assigned = sum(len(polygons) for polygons in assignment.values())
                active_couriers = sum(1 for polygons in assignment.values() if polygons)

                self.metrics.append(**{
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

                if (episode + 1) % 20 == 0:
                    logger.info(f"Эпизод {episode + 1}: назначено {total_assigned} полигонов, активных курьеров {active_couriers}, цель {global_objective:.0f}")

                if no_improvement_count >= early_stop_threshold:
                    logger.info(f"Ранняя остановка: нет улучшений {early_stop_threshold} эпизодов подряд")
                    break
            except Exception as e:
                logger.error(f"Ошибка в эпизоде {episode + 1}: {e}")

        logger.info(f"Оптимизированное последовательное обучение завершено. Лучшая цель: {best_objective:.0f}")

        if best_assignment is None:
            return {}

        try:
            self.metrics.save('rl_training_metrics.parquet', 'rl_training_metrics.csv')
        except Exception as e:
            logger.error(f"Не удалось сохранить метрики обучения: {e}")

        return best_assignment

    def _preload_distances(self):
        logger.info("Начинаем предварительную загрузку расстояний...")
        polygon_ids = list(self.environment.polygon_info.keys())
        warehouse_id = self.environment.warehouse_id
        self.distance_matrix: Dict[Tuple[int, int], float] = {}
        for polygon_id in polygon_ids:
            if hasattr(self.environment.distance_provider, 'get_polygon_access_cost'):
                cost = self.environment.distance_provider.get_polygon_access_cost(warehouse_id, polygon_id, 0)
                self.distance_matrix[(warehouse_id, polygon_id)] = cost
            for other_polygon_id in polygon_ids:
                if other_polygon_id == polygon_id:
                    continue
                if hasattr(self.environment.distance_provider, 'get_polygon_access_cost'):
                    cost = self.environment.distance_provider.get_polygon_access_cost(polygon_id, other_polygon_id, 0)
                    self.distance_matrix[(polygon_id, other_polygon_id)] = cost
        logger.info(f"Загружено {len(self.distance_matrix)} расстояний в память")

    def _init_vectorized_computations(self):
        logger.info("Инициализация векторизованных вычислений...")
        polygon_ids = list(self.environment.polygon_info.keys())
        courier_ids = list(self.environment.courier_states.keys())
        self.polygon_ids_array = np.array(polygon_ids)
        self.courier_ids_array = np.array(courier_ids)
        self.state_matrix = np.zeros((len(courier_ids), len(polygon_ids)), dtype=np.float32)
        logger.info(f"Создана матрица состояний: {self.state_matrix.shape}")

    def _train_parallel(self, episodes: int, max_steps_per_episode: int) -> Dict[int, List[int]]:
                                                    
        base_context: Dict[str, Any] = {
            'polygons_df': self.environment.polygons_df.to_dicts(),
            'couriers_df': self.environment.couriers_df.to_dicts(),
            'max_time_per_courier': self.environment.max_time_per_courier,
                                                                                           
                                                                                     
            'polygon_info': getattr(self.environment.distance_provider, 'polygon_info', {}),
            'courier_service_times': self.environment.courier_service_times,
            'durations_db_path': getattr(self.environment.distance_provider, 'durations_db_path', ''),
            'ports_db_path': getattr(self.environment.distance_provider, 'ports_db_path', ''),
            'warehouse_ports_db_path': getattr(self.environment.distance_provider, 'warehouse_ports_db_path', ''),
        }
        from .parallel import ParallelTrainer
        trainer = ParallelTrainer(base_context)
        best_assignment, best_objective = trainer.run(episodes, max_steps_per_episode, num_workers=self.num_workers)
        if not best_assignment:
            logger.warning("Параллельное обучение не дало результата, падаем назад на последовательное")
            return self._train_sequential_optimized(episodes, max_steps_per_episode)
        return best_assignment

    def _run_episode_optimized(self, episode_num: int, max_steps_per_episode: int) -> Tuple[Dict[int, List[int]], float, int, List[Tuple], Dict[str, Any]]:
        env_copy = VRPEnvironment(
            self.environment.polygons_df,
            self.environment.couriers_df,
            self.environment.max_time_per_courier,
            self.environment.distance_provider,
            self.environment.courier_service_times
        )
        try:
            env_copy.warehouse_id = int(getattr(self.environment, 'warehouse_id', 0))
        except Exception:
            pass
        env_copy.reset()
        episode_reward = 0
        step_count = 0
        experience_buffer: List[Tuple[str, int, float, str]] = []

        while not env_copy.is_episode_finished() and step_count < max_steps_per_episode:
            step_count += 1
            active_couriers = [cid for cid, c in env_copy.courier_states.items() if c.current_time < env_copy.max_time_per_courier]
            if not active_couriers:
                break
            if all(len(env_copy.get_available_actions(cid)) == 0 for cid in active_couriers):
                break
            courier_id = min(active_couriers, key=lambda cid: env_copy.courier_states[cid].current_time)
            result = self.agent.act(env_copy, courier_id)
            if result is None:
                continue
            current_state, action, reward, next_state = result
            episode_reward += reward
            experience_buffer.append((current_state, action, reward, next_state))

        assignment = {cid: c.assigned_polygons.copy() for cid, c in env_copy.courier_states.items()}
        global_objective = env_copy.calculate_global_objective()
        courier_times = [c.current_time for c in env_copy.courier_states.values()]
        stats = {
            'episode': episode_num,
            'courier_times': courier_times,
            'avg_courier_time': (sum(courier_times) / len(courier_times)) if courier_times else 0,
            'max_courier_time': max(courier_times) if courier_times else 0,
            'total_assigned': sum(len(v) for v in assignment.values()),
            'active_couriers': sum(1 for t in courier_times if t > 0),
            'remaining_polygons': len(env_copy.available_polygons)
        }
        return assignment, global_objective, step_count, experience_buffer, stats

    def _batch_update_q_values(self, experience_buffer: List[Tuple[str, int, float, str]]):
        if not experience_buffer:
            return
        state_action_updates: Dict[Tuple[str, int], List[Tuple[float, str]]] = {}
        for current_state, action, reward, next_state in experience_buffer:
            key = (current_state, action)
            state_action_updates.setdefault(key, []).append((reward, next_state))
        for (state, action), experiences in state_action_updates.items():
            avg_reward = sum(r for r, _ in experiences) / len(experiences)
            next_state = experiences[0][1]
            self.agent.update_q_value(state, action, avg_reward, next_state)

    def solve(self, polygons_df, couriers_df, max_time_per_courier: int) -> Dict[int, List[int]]:
        logger.info("Начинаем решение VRP с обучением с подкреплением")

        total_polygons = len(polygons_df)
        total_couriers = len(couriers_df)

        episodes = min(500, max(100, total_polygons // 50))
        max_steps = min(5000, max(2000, total_polygons * 2))

        logger.info(f"Параметры обучения: {episodes} эпизодов, {max_steps} шагов на эпизод")
        logger.info(f"Всего полигонов: {total_polygons}, курьеров: {total_couriers}")

        self.polygons_df = polygons_df
        self.couriers_df = couriers_df
        self.max_time_per_courier = max_time_per_courier

        dp = getattr(self.environment, 'distance_provider', None)
        if dp and hasattr(dp, 'polygon_info'):
            self.polygon_info = dp.polygon_info
        else:
            self.polygon_info = {}
        if dp and hasattr(dp, 'courier_service_times'):
            self.courier_service_times = dp.courier_service_times
        else:
            self.courier_service_times = {}

        warm_assignment = greedy_initialize_assignment(
            polygons_df, couriers_df, max_time_per_courier,
            self.environment.distance_provider,
            getattr(self.environment, 'courier_service_times', {}),
            warehouse_id=self.environment.warehouse_id,
            top_k=50,
        )
        for cid, pids in warm_assignment.items():
                                                                                     
                                                                        
            if cid not in self.environment.courier_states:
                try:
                    env_ids = list(self.environment.courier_states.keys())
                    mapped_cid = env_ids[cid] if 0 <= cid < len(env_ids) else cid
                except Exception:
                    mapped_cid = cid
            else:
                mapped_cid = cid
            state = self.state_encoder.encode_state(self.environment.get_environment_state(), mapped_cid)
            for pid in pids:
                self.agent.q_table[(state, pid)] = self.agent.q_table.get((state, pid), 0.0) + 1.0

        assignment = self.train(episodes=episodes, max_steps_per_episode=max_steps)

        total_assigned = sum(len(polygons) for polygons in assignment.values())
        active_couriers = sum(1 for polygons in assignment.values() if polygons)
        logger.info(f"Результат: назначено {total_assigned} полигонов, активных курьеров {active_couriers}")
        return assignment


