from typing import Dict, List

from ..rl.warm_start import greedy_initialize_assignment


class WarmStartAssignment:
    def __init__(self, epsilon: float = 0.05, top_k: int = 12):
        self.epsilon = float(epsilon)
        self.top_k = int(top_k)

    def assign(
        self,
        polygons_df,
        couriers_df,
        provider,
        courier_service_times: Dict[int, Dict[int, int]],
        mp_to_order_count: Dict[int, int],
        warehouse_id: int,
        max_time_per_courier: int,
    ) -> Dict[int, List[int]]:
        return greedy_initialize_assignment(
            polygons_df,
            couriers_df,
            provider,
            courier_service_times,
            mp_to_order_count,
            warehouse_id=warehouse_id,
            epsilon=self.epsilon,
            top_k=self.top_k,
        )


