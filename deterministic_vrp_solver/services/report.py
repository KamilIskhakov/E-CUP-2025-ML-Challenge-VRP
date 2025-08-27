from typing import Dict, List
import json


def build_validation_report(
    exec_time: float,
    total_couriers: int,
    active_couriers: int,
    num_polygons: int,
    max_route_time: int,
    avg_route_time: int,
    sum_route_time: int,
    unassigned_orders: int,
) -> Dict:
    return {
        'exec_time_sec': exec_time,
        'total_couriers': total_couriers,
        'active_couriers': active_couriers,
        'polygons': int(num_polygons),
        'max_route_time_sec': int(max_route_time),
        'avg_route_time_sec': int(avg_route_time),
        'sum_route_time_sec': int(sum_route_time),
        'unassigned_orders': int(unassigned_orders),
        'penalty_unassigned_sec': int(3000 * unassigned_orders),
    }


def save_report(path: str, report: Dict) -> None:
    with open(path, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)


