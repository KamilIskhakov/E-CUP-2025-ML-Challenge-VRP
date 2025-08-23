from typing import Dict


class QLearningAgent:
    def __init__(self, state_encoder, action_selector, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.state_encoder = state_encoder
        self.action_selector = action_selector
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict = {}

    def get_q_value(self, state: str, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state: str, action: int, reward: float, next_state: str):
        current_q = self.get_q_value(state, action)
        max_next_q = 0.0
        if next_state:
            next_actions = [a for (s, a) in self.q_table.keys() if s == next_state]
            if next_actions:
                max_next_q = max(self.get_q_value(next_state, a) for a in next_actions)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def select_action(self, env_state, courier_id, environment):
        state = self.state_encoder.encode_state(env_state, courier_id)
        available_actions = environment.get_available_actions(courier_id)
        action = self.action_selector.select_action(state, available_actions, self.q_table)
        return action

    def act(self, environment, courier_id: int):
        """Высокоуровневый выбор действия и его исполнение в среде.

        Возвращает кортеж (state, action, reward, next_state) или None, если действий нет.
        """
        env_state = environment.get_environment_state()
        state = self.state_encoder.encode_state(env_state, courier_id)
        available_actions = environment.get_available_actions(courier_id)
        if not available_actions:
            return None
        action = self.action_selector.select_action(state, available_actions, self.q_table)
        if action == -1:
            return None
        reward, _ = environment.execute_action(courier_id, action)
        next_state = self.state_encoder.encode_state(environment.get_environment_state(), courier_id)
        return state, action, reward, next_state


