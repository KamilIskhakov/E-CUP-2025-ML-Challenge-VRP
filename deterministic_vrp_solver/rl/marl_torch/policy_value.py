import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorPolicy(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, courier_emb: torch.Tensor, candidate_embs: torch.Tensor, mask: torch.Tensor):
                                                                                  
        E, K, D = candidate_embs.shape
        courier_expanded = courier_emb.unsqueeze(1).expand(-1, K, -1)
        logits = self.fc(torch.cat([courier_expanded, candidate_embs], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(mask == 0, -1e9)
        probs = F.softmax(logits, dim=-1)
        return probs


class CentralValue(nn.Module):
    def __init__(self, courier_dim: int = 16, polygon_dim: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(courier_dim + polygon_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, pooled_courier: torch.Tensor, pooled_polygon: torch.Tensor):
        x = torch.cat([pooled_courier, pooled_polygon], dim=-1)
        return self.fc(x).squeeze(-1)


