from typing import Dict
from dataclasses import dataclass


@dataclass
class EnvironmentState:
    courier_states: Dict
    available_polygons: list
    polygon_info: Dict
    current_time: int
    max_time_per_courier: int


class IStateEncoder:
    def encode_state(self, env_state: EnvironmentState, courier_id: int) -> str:
        raise NotImplementedError


class SimpleStateEncoder(IStateEncoder):
    def encode_state(self, env_state: EnvironmentState, courier_id: int) -> str:
        courier = env_state.courier_states[courier_id]
        time_bucket = min(courier.current_time // 7200, 5)
        polygon_bucket = min(len(courier.assigned_polygons), 10)
        available_bucket = min(len(env_state.available_polygons) // 20, 10)
                                              
        utilization_bucket = min(int((courier.current_time / max(1, env_state.max_time_per_courier)) * 10), 10)
                                                    
        accessible_polygons = min(10, len(env_state.available_polygons))
                    
        remaining_bucket = min(len(env_state.available_polygons) // 50, 10)
        return f"T{time_bucket}_P{polygon_bucket}_A{available_bucket}_U{utilization_bucket}_AC{accessible_polygons}_R{remaining_bucket}"


