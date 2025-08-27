from typing import Dict, List, Tuple, Optional
import math
import numpy as np


def _hungarian(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Детерминированный Венгерский алгоритм через numpy/ортулз не используем, чтобы не тащить лишние зависимости.
    Предполагаем, что число курьеров <= число полигонов; допускаем, что некоторые полигоны останутся неназначенными.
    Возвращает список пар (courier_idx, polygon_idx) для назначенных пар.
    """
                                                                                         
                                                                                         
    m, n = cost_matrix.shape
    assigned = [-1] * m
    used_cols = set()
                             
    norm = cost_matrix - np.min(cost_matrix, axis=1, keepdims=True)
    for i in range(m):
        row = norm[i]
                                                 
        best_j = None
        best_v = math.inf
        for j in range(n):
            v = row[j]
            if j in used_cols:
                continue
            if v < best_v:
                best_v = v
                best_j = j
        if best_j is not None and math.isfinite(cost_matrix[i, best_j]):
            assigned[i] = best_j
            used_cols.add(best_j)
    return [(i, j) for i, j in enumerate(assigned) if j >= 0]


def lagrangian_time_balanced_assignment(
    base_costs: np.ndarray,
    times: np.ndarray,
    time_budgets: np.ndarray,
    max_iters: int = 15,
    step0: float = 100.0,
    step_decay: float = 0.85,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Лагранжева релаксация для назначения курьер–полигон с бюджетами времени.

    Аргументы:
      - base_costs: матрица c_ij (m x n), inf для невозможных пар
      - times: матрица t_ij (m x n)
      - time_budgets: вектор T_i (m)

    Возвращает:
      - список назначений (i, j)
      - вектор множителей Лагранжа λ_i
    """
    m, n = base_costs.shape
    lam = np.zeros(m, dtype=float)

    best_assign: List[Tuple[int, int]] = []
    best_violation = math.inf

    step = float(step0)

    for _ in range(max(1, int(max_iters))):
                                                       
        mod = base_costs + lam.reshape(-1, 1) * times
                                    
        mod = np.where(np.isfinite(base_costs), mod, math.inf)

        match = _hungarian(mod)

                                              
        time_used = np.zeros(m, dtype=float)
        for i, j in match:
            time_used[i] += times[i, j]

                                               
        g = time_used - time_budgets

                                                             
        violation = float(np.sum(np.maximum(0.0, g)))
        if violation < best_violation:
            best_violation = violation
            best_assign = match

                            
        lam = np.maximum(0.0, lam + step * g)

                        
        step *= step_decay

                                               
        if best_violation <= 1e-6:
            break

    return best_assign, lam


