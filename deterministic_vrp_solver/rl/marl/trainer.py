from typing import Dict, List

from .graph_encoder import GraphEncoder
from .policy import SharedPolicy, CentralCritic


class CTDETrainer:
    """Скелет CTDE-тренера (без ML-фреймворков) для интеграции в оффлайн пайплайн.

    Не подключается к инференсу. Предполагает внешнюю сборку траекторий и обучение
    параметров (в полноценной версии — через torch/ppo/qmix).
    """

    def __init__(self):
        self.encoder = GraphEncoder()
        self.policy = SharedPolicy(epsilon=0.1)
        self.critic = CentralCritic()

    def rollout_step(self, courier_states: Dict[int, Dict], polygon_info: Dict[int, Dict], available_actions: Dict[int, List[int]]):
        embeddings = self.encoder.encode(courier_states, polygon_info)
        actions: Dict[int, int] = {}
        for cid, acts in available_actions.items():
            actions[cid] = self.policy.act(cid, acts, embeddings)
        value = self.critic.value(embeddings)
        return actions, value


