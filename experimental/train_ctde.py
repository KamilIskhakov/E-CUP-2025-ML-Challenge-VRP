                      

import logging
from typing import Dict, List

from deterministic_vrp_solver.rl.marl.trainer import CTDETrainer


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                                                                   
    courier_states: Dict[int, Dict] = {i: {"current_time": 0, "max_time": 43200} for i in range(4)}
    polygon_info: Dict[int, Dict] = {i: {"order_count": 5, "total_distance": 1000} for i in range(10)}
    available_actions: Dict[int, List[int]] = {i: list(polygon_info.keys())[:5] for i in courier_states}

    trainer = CTDETrainer()
    actions, value = trainer.rollout_step(courier_states, polygon_info, available_actions)
    logging.info(f"Sample rollout actions: {actions}; value_estimate={value:.3f}")


if __name__ == "__main__":
    main()


