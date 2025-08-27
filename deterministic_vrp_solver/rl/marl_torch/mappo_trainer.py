import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple

from .graph_encoder_torch import GraphEncoderTorch
from .policy_value import ActorPolicy, CentralValue


class MAPPOTrainer:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = GraphEncoderTorch().to(self.device)
        self.actor = ActorPolicy().to(self.device)
        self.value_net = CentralValue().to(self.device)
        self.opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.value_net.parameters()),
            lr=3e-4,
        )

    def compute_features(self, courier_states: Dict, polygon_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
                                                         
        c_feats = []
        for c in courier_states.values():
            t = float(getattr(c, 'current_time', c.get('current_time', 0)))
            max_t = float(getattr(c, 'max_time', c.get('max_time', 43200)))
            util = min(1.0, t / max(1.0, max_t))
            assigned = len(getattr(c, 'assigned_polygons', c.get('assigned_polygons', [])))
            c_feats.append([util, float(assigned), 0.0, 1.0])
        p_feats = []
        for p in polygon_info.values():
            orders = float(p.get('order_count', 0))
            cost = float(p.get('total_distance', p.get('cost', 0)))
            util = orders / (cost + 1.0) if cost > 0 else 0.0
            p_feats.append([orders, cost, util, 1.0])
        return torch.tensor(c_feats, device=self.device), torch.tensor(p_feats, device=self.device)

    def policy_step(self, courier_emb: torch.Tensor, candidate_ids: List[List[int]], polygon_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                                                
        max_k = max(len(x) for x in candidate_ids) if candidate_ids else 0
        if max_k == 0:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)
        batch = len(candidate_ids)
        mask = torch.zeros((batch, max_k), dtype=torch.bool, device=self.device)
        cand = torch.zeros((batch, max_k, polygon_emb.shape[-1]), device=self.device)
        for i, ids in enumerate(candidate_ids):
            if not ids:
                continue
            mask[i, : len(ids)] = 1
            cand[i, : len(ids)] = polygon_emb[torch.tensor(ids, device=self.device)]
        probs = self.actor(courier_emb, cand, mask)
        return probs, mask

    def value_step(self, courier_emb: torch.Tensor, polygon_emb: torch.Tensor) -> torch.Tensor:
        pooled_c = courier_emb.mean(dim=0, keepdim=True)
        pooled_p = polygon_emb.mean(dim=0, keepdim=True)
        return self.value_net(pooled_c, pooled_p)


