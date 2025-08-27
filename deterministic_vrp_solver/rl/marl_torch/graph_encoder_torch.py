import torch
import torch.nn as nn


class GraphEncoderTorch(nn.Module):
    def __init__(self, courier_dim: int = 16, polygon_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.courier_proj = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, courier_dim))
        self.polygon_proj = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, polygon_dim))

    def forward(self, courier_feats: torch.Tensor, polygon_feats: torch.Tensor):
                                                                        
                                                            
        ce = self.courier_proj(courier_feats)
        pe = self.polygon_proj(polygon_feats)
        return ce, pe


