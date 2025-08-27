import random


class IActionSelector:
    def select_action(self, state: str, available_actions: list, q_table: dict) -> int:
        raise NotImplementedError


class EpsilonGreedyActionSelector(IActionSelector):
    def __init__(self, epsilon: float = 0.1, min_epsilon: float = 0.05, decay: float = 0.995):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def select_action(self, state: str, available_actions: list, q_table: dict) -> int:
        if not available_actions:
            return -1
        if random.random() < self.epsilon:
            return random.choice(available_actions)
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


