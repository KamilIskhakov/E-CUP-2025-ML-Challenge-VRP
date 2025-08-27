from typing import Dict, List, Any


class InferenceOrchestrator:
    def __init__(self) -> None:
        pass

    def assign(
        self,
        strategy_name: str,
        polygons_df: Any,
        couriers_df: Any,
        provider: Any,
        courier_service_times: Dict[int, Dict[int, int]],
        mp_to_order_count: Dict[int, int],
        warehouse_id: int,
        max_time_per_courier: int,
    ) -> Dict[int, List[int]]:
        from .strategies.factory import build_assignment_strategy
        strategy = build_assignment_strategy(strategy_name)
        return strategy.assign(
            polygons_df,
            couriers_df,
            provider,
            courier_service_times,
            mp_to_order_count,
            warehouse_id,
            max_time_per_courier,
        )

    def post_improve(
        self,
        chain: str,
        route_optimizer: Any,
        routes: Dict[int, Dict],
        assignment: Dict[int, List[int]],
        polygons_df: Any,
        provider: Any,
        time_cap: int,
    ) -> Dict[int, Dict]:
        from .strategies.factory import build_post_improver
        improver = build_post_improver(chain)
        return improver.improve(route_optimizer, routes, assignment, polygons_df, provider, time_cap)


