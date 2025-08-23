import logging
from typing import Dict, List, Tuple, Any
import numpy as np

from .encoder import SimpleStateEncoder
from .selector import EpsilonGreedyActionSelector
from .agent import QLearningAgent
from .env import VRPEnvironment
from .metrics import MetricsRecorder


logger = logging.getLogger(__name__)


class RLTrainer:
    def __init__(self, environment: VRPEnvironment):
        self.environment = environment
        self.state_encoder = SimpleStateEncoder()
        self.action_selector = EpsilonGreedyActionSelector(epsilon=0.3, min_epsilon=0.05, decay=0.995)
        self.agent = QLearningAgent(self.state_encoder, self.action_selector, learning_rate=0.1, discount_factor=0.9)
        self.metrics = MetricsRecorder()

    def _init_vectorized_computations(self):
        polygon_ids = list(self.environment.polygon_info.keys())
        courier_ids = list(self.environment.courier_states.keys())
        self.polygon_ids_array = np.array(polygon_ids)
        self.courier_ids_array = np.array(courier_ids)
        self.state_matrix = np.zeros((len(courier_ids), len(polygon_ids)), dtype=np.float32)
        logger.info(f"Создана матрица состояний: {self.state_matrix.shape}")

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

    def _run_episode_optimized(self, episode_num: int, max_steps_per_episode: int) -> Tuple[Dict[int, List[int]], float, int, List[Tuple], Dict[str, Any]]:
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
        experience_buffer: List[Tuple[str, int, float, str]] = []

        # Избегаем множественных вызовов get_available_actions на каждом шаге для всех курьеров
        no_action_couriers = set()
        while not env_copy.is_episode_finished() and step_count < max_steps_per_episode:
            step_count += 1
            # Курьеры, которые ещё не исчерпаны по времени и не помечены как бездействующие
            active_couriers = [cid for cid, c in env_copy.courier_states.items() if c.current_time < env_copy.max_time_per_courier and cid not in no_action_couriers]
            if not active_couriers:
                break
            # Берём наименее загруженного по времени
            courier_id = min(active_couriers, key=lambda cid: env_copy.courier_states[cid].current_time)
            # Проверяем доступные действия только для выбранного курьера
            actions = env_copy.get_available_actions(courier_id)
            if not actions:
                # Больше нечего делать этому курьеру в рамках ограничений
                no_action_couriers.add(courier_id)
                continue
            result = self.agent.act(env_copy, courier_id)
            if result is None:
                # На случай гонки состояний
                no_action_couriers.add(courier_id)
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

    def train(self, episodes: int, max_steps_per_episode: int) -> Dict[int, List[int]]:
        best_assignment = None
        best_objective = float('inf')
        no_improvement_count = 0
        early_stop_threshold = 30

        logger.info(f"Оптимизированное последовательное обучение: {episodes} эпизодов")
        self._init_vectorized_computations()

        for episode in range(episodes):
            try:
                assignment, global_objective, step_count, experience_buffer, stats = self._run_episode_optimized(episode, max_steps_per_episode)

                self._batch_update_q_values(experience_buffer)

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


