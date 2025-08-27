                      

import logging
from typing import Dict, List

import torch

from deterministic_vrp_solver.rl.marl_torch.mappo_trainer import MAPPOTrainer


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    trainer = MAPPOTrainer()

                                                                            
    courier_states: Dict[int, Dict] = {i: {"current_time": 0.0, "max_time": 43200.0, "assigned_polygons": []} for i in range(8)}
    polygon_info: Dict[int, Dict] = {i: {"order_count": 3 + (i % 5), "total_distance": 500 + 10 * i} for i in range(100)}

    c_feats, p_feats = trainer.compute_features(courier_states, polygon_info)
    c_emb, p_emb = trainer.encoder(c_feats, p_feats)

                                                                    
    K = 10
    candidate_ids: List[List[int]] = [list(range(K)) for _ in range(c_emb.shape[0])]
    probs, mask = trainer.policy_step(c_emb, candidate_ids, p_emb)
    v = trainer.value_step(c_emb, p_emb)
    logging.info(f"policy probs shape={tuple(probs.shape)} mask_sum={int(mask.sum())} value={float(v.item()):.3f}")


if __name__ == "__main__":
    main()


