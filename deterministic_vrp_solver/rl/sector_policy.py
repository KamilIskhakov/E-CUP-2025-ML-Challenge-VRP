from typing import List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SectorPolicy(nn.Module):
    """Простая модель, которая по угловой гистограмме заказов предсказывает K центров секторов.

    Вход: histogram (B, H) — распределение заказов по H угловым бинам [0..360).
    Выход: logits (B, K, H) — распределения по бинам для K центров; берём argmax по H для центров.
    """

    def __init__(self, num_bins: int = 72, num_sectors: int = 24, hidden_dim: int = 128):
        super().__init__()
        self.num_bins = int(num_bins)
        self.num_sectors = int(num_sectors)
        self.encoder = nn.Sequential(
            nn.Linear(self.num_bins, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, self.num_sectors * self.num_bins)

    def forward(self, histogram: torch.Tensor) -> torch.Tensor:
        x = self.encoder(histogram)
        logits = self.head(x)
        return logits.view(-1, self.num_sectors, self.num_bins)


def _angle_of(lat: float, lon: float, wh_lat: float, wh_lon: float) -> float:
    dy = float(lat) - float(wh_lat)
    dx = float(lon) - float(wh_lon)
    return (math.degrees(math.atan2(dy, dx)) % 360.0)


def build_angle_histogram(orders_df, wh_lat: float, wh_lon: float, num_bins: int = 72) -> List[float]:
    counts = [0.0] * int(num_bins)
    if not {'Lat', 'Long', 'MpId', 'ID'}.issubset(set(orders_df.columns)):
        return counts
    bin_size = 360.0 / float(num_bins)
    for row in orders_df.iter_rows(named=True):
        ang = _angle_of(float(row['Lat']), float(row['Long']), wh_lat, wh_lon)
        idx = int(ang // bin_size)
        if 0 <= idx < num_bins:
            counts[idx] += 1.0
    s = sum(counts) or 1.0
    return [c / s for c in counts]


def select_sector_centers(
    model: SectorPolicy,
    histogram: List[float],
    top_per_sector: int = 1,
) -> List[int]:
    """Возвращает индексы бинов центров секторов длиной K=model.num_sectors."""
    model.eval()
    with torch.no_grad():
        h = torch.tensor(histogram, dtype=torch.float32).unsqueeze(0)
        logits = model(h)[0]          
        centers: List[int] = []
        for k in range(logits.shape[0]):
            probs = F.softmax(logits[k], dim=-1)
            topk = torch.topk(probs, k=top_per_sector).indices.tolist()
            centers.append(int(topk[0]))
        return centers


