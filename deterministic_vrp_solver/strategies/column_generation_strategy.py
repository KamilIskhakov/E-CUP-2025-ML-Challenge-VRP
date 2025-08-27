from typing import Dict, List

from ..column_generation import run_column_generation_assignment


class ColumnGenerationAssignment:
    def __init__(self, iterations: int = 3, use_espprc: bool = True):
        self.iterations = int(iterations)
        self.use_espprc = bool(use_espprc)

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
        return run_column_generation_assignment(
            polygons_df,
            couriers_df,
            provider,
            courier_service_times,
            mp_to_order_count,
            warehouse_id,
            max_time_per_courier,
            iterations=self.iterations,
            use_espprc=self.use_espprc,
        )


