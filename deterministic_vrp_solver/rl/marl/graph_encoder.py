from typing import Dict, List


class GraphEncoder:
    """Лёгкий скелет графового энкодера (без torch-зависимостей).

    Ожидает двудольный граф курьеры–полигоны и агрегированные признаки.
    Возвращает эмбеддинги курьеров и полигонов (словарями id->вектор признаков).
    """

    def __init__(self, courier_feature_size: int = 16, polygon_feature_size: int = 16):
        self.courier_feature_size = courier_feature_size
        self.polygon_feature_size = polygon_feature_size

    def encode(self, courier_states: Dict[int, Dict], polygon_info: Dict[int, Dict]) -> Dict[str, Dict[int, List[float]]]:
                                                                                     
        courier_embeddings: Dict[int, List[float]] = {}
        polygon_embeddings: Dict[int, List[float]] = {}

        for cid, c in courier_states.items():
            t = float(getattr(c, 'current_time', c.get('current_time', 0)))
            util = min(1.0, t / float(c.get('max_time', 43200))) if isinstance(c, dict) else 0.0
            courier_embeddings[cid] = [util] * self.courier_feature_size

        for pid, p in polygon_info.items():
            orders = float(p.get('order_count', 0))
            cost = float(p.get('total_distance', p.get('cost', 0)))
            s1 = orders / (cost + 1.0) if cost > 0 else 0.0
            polygon_embeddings[pid] = [s1] * self.polygon_feature_size

        return {"couriers": courier_embeddings, "polygons": polygon_embeddings}


