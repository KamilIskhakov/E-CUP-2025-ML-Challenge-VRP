import multiprocessing as mp
from typing import Dict, List, Tuple, Any


def _worker_train(process_data: Dict) -> Dict[str, Any]:
    # Локальные импорты внутри процесса
    import polars as pl
    from deterministic_vrp_solver.decomposed_distance_provider import DecomposedDistanceProvider
    from .scheduler import ReinforcementScheduler

    polygons_df = pl.DataFrame(process_data['polygons_df'])
    couriers_df = pl.DataFrame(process_data['couriers_df'])
    max_time = int(process_data['max_time_per_courier'])
    episodes = int(process_data['episodes'])
    max_steps = int(process_data['max_steps'])

    provider = DecomposedDistanceProvider(
        durations_db_path=process_data['durations_db_path'],
        ports_db_path=process_data['ports_db_path'],
        warehouse_ports_db_path=process_data['warehouse_ports_db_path'],
    )
    provider.__enter__()
    provider.set_polygon_info(process_data['polygon_info'])

    scheduler = ReinforcementScheduler(
        polygons_df, couriers_df, max_time,
        distance_provider=provider,
        courier_service_times=process_data['courier_service_times'],
        use_parallel=False,
    )

    try:
        assignment = scheduler.train(episodes=episodes, max_steps_per_episode=max_steps)
        objective = float('inf')
        # Точная оценка: симулируем эпизод по assignment
        if assignment:
            env = scheduler.environment
            objective = float(env.evaluate_assignment(assignment))
        return {'assignment': assignment, 'objective': float(objective)}
    finally:
        provider.close()


class ParallelTrainer:
    def __init__(self, base_context: Dict):
        self.base_context = base_context

    def run(self, episodes: int, max_steps: int, num_workers: int = 2) -> Tuple[Dict[int, List[int]], float]:
        if num_workers <= 1 or episodes <= num_workers:
            # Нечего параллелить
            return {}, float('inf')
        batch = max(1, episodes // num_workers)
        tasks: List[Dict] = []
        for _ in range(num_workers):
            pd = dict(self.base_context)
            pd['episodes'] = batch
            pd['max_steps'] = max_steps
            tasks.append(pd)

        with mp.get_context('spawn').Pool(processes=num_workers) as pool:
            results = pool.map(_worker_train, tasks)

        best_assignment: Dict[int, List[int]] = {}
        best_objective = float('inf')
        for r in results:
            if r and r.get('assignment') and r.get('objective', float('inf')) < best_objective:
                best_objective = float(r['objective'])
                best_assignment = r['assignment']
        return best_assignment, best_objective


