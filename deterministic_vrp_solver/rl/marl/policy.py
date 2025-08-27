from typing import Dict, List
import random


class SharedPolicy:
    """Децентрализованная shared-политика поверх эмбеддингов.

    На вход: эмбеддинг курьера и доступных полигонов (Top-K), маска допустимости.
    На выход: выбранное действие (id полигона).
    """

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def act(self, courier_id: int, available_polygons: List[int], embeddings: Dict[str, Dict[int, List[float]]]) -> int:
        if not available_polygons:
            return -1
        if random.random() < self.epsilon:
            return random.choice(available_polygons)
                                                                                   
        pe = embeddings.get("polygons", {})
        best = max(available_polygons, key=lambda pid: (pe.get(pid, [0.0])[0] if pid in pe else 0.0))
        return best


class CentralCritic:
    """Централизованный критик (заглушка). Возвращает оценку состояния.
    В полноценной версии сюда пойдёт torch и обучение value-функции.
    """

    def value(self, embeddings: Dict[str, Dict[int, List[float]]]) -> float:
                                                  
        total = 0.0
        for v in embeddings.get("couriers", {}).values():
            total += sum(v)
        for v in embeddings.get("polygons", {}).values():
            total += sum(v)
        return total


