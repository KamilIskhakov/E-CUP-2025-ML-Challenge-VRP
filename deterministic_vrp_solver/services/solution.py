from typing import Dict, List
import json
from pathlib import Path
import polars as pl


def generate_solution_structure(route_optimizer: Dict[int, Dict], optimized_polygons: pl.DataFrame) -> Dict:
    routes = []
    for courier_id, info in route_optimizer.items():
        if not info or not info.get('polygon_order'):
            continue
        route_entry = {
            'courier_id': int(courier_id),
            'polygon_order': [int(pid) for pid in info['polygon_order']],
            'total_time': int(info.get('total_time', 0)),
            'route_details': info.get('route_details', []),
        }
        routes.append(route_entry)
    return {'routes': routes}


def save_solution(output_path: Path, final_routes: List[Dict]) -> None:
    solution = {'routes': final_routes}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(solution, f, indent=2)


