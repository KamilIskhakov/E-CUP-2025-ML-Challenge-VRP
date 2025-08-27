from typing import Protocol, Dict, List, Any


class AssignmentStrategy(Protocol):
    def assign(
        self,
        polygons_df: Any,
        couriers_df: Any,
        provider: Any,
        courier_service_times: Dict[int, Dict[int, int]],
        mp_to_order_count: Dict[int, int],
        warehouse_id: int,
        max_time_per_courier: int,
    ) -> Dict[int, List[int]]: ...


class PostRoutingImprover(Protocol):
    def improve(
        self,
        route_optimizer: Any,
        routes: Dict[int, Dict],
        assignment: Dict[int, List[int]],
        polygons_df: Any,
        provider: Any,
        time_cap: int,
    ) -> Dict[int, Dict]: ...


