from typing import Dict, List

from ..rcsp_assigner import compute_rcsp_assignment


class RCSPAssignment:
    def __init__(self, top_k: int = 12, refill_pool_size: int = 2000):
        self.top_k = int(top_k)
        self.refill_pool_size = int(refill_pool_size)

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
        return compute_rcsp_assignment(
            polygons_df,
            couriers_df,
            provider,
            courier_service_times,
            max_time_per_courier=max_time_per_courier,
            warehouse_id=warehouse_id,
            per_order_penalty=3000,
            top_k_candidates=self.top_k,
            refill_pool_size=self.refill_pool_size,
        )


