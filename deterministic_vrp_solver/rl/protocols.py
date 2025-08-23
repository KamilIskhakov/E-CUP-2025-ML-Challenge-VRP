from typing import Protocol, Tuple


class DistanceProviderProtocol(Protocol):
    """Протокол провайдера расстояний, необходимый для среды RL.

    Требуем минимум две операции:
    - get_polygon_access_cost: время доступа к полигону из текущей позиции с учётом сервисного времени
    - find_best_port_to_polygon: выбор лучшего порта входа для обновления позиции курьера
    """

    def get_polygon_access_cost(self, from_position: int, polygon_id: int, service_time: float = 0) -> float:
        ...

    def find_best_port_to_polygon(self, from_position: int, target_polygon_id: int) -> Tuple[int, float]:
        ...


