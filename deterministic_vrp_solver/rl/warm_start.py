from typing import Dict, List, Tuple, Optional
import heapq
import math
import random

import polars as pl
import logging
import numpy as np

logger = logging.getLogger(__name__)


def greedy_initialize_assignment(
    polygons_df: pl.DataFrame,
    couriers_df: pl.DataFrame,
    max_time_per_courier: int,
    distance_provider,
    courier_service_times: Dict[int, Dict[int, int]],
    warehouse_id: int = 0,
    top_k: int = 50,
    candidate_pool_size: int = 200,
    per_order_penalty: int = 3000,
    epsilon: float = 0.15,
    ucb_c: float = 1.0,
    rng_seed: Optional[int] = None,
    use_graph_embedding: bool = False,
    embedding_dim: int = 8,
    embedding_beta: float = 0.3,
    graph_distance_threshold: int = 1200,
    fairness_beta: float = 0.2,
) -> Dict[int, List[int]]:
    order_count_map: Dict[int, int] = {}
    if 'MpId' in polygons_df.columns and 'order_count' in polygons_df.columns:
        for row in polygons_df.select(['MpId', 'order_count']).iter_rows():
            order_count_map[row[0]] = row[1]

    courier_ids: List[int] = couriers_df['ID'].to_list() if 'ID' in couriers_df.columns else list(range(len(couriers_df)))
    courier_state = {
        int(cid): {'position': warehouse_id, 'time': 0, 'assigned': []}
        for cid in courier_ids
    }

    available_polygons: List[int] = [row['MpId'] for row in polygons_df.iter_rows(named=True) if row['MpId'] != warehouse_id]
    total_to_assign = len(available_polygons)

                                                                                    
    static_cost_hint: Dict[int, float] = {}
    for pid in available_polygons:
        try:
            static_cost_hint[pid] = distance_provider.get_polygon_access_cost(warehouse_id, pid, 0)
        except Exception:
            static_cost_hint[pid] = float('inf')
                                                     
    available_polygons.sort(key=lambda p: static_cost_hint.get(p, float('inf')))

                                                
    available_set = set(available_polygons)

                                                                                        
    access_cost_cache: Dict[Tuple[int, int, int], float] = {}

    def get_access_cost(from_pos: int, polygon_id: int, svc_time: int) -> float:
        key = (from_pos, polygon_id, svc_time)
        if key in access_cost_cache:
            return access_cost_cache[key]
        try:
            cost_val = distance_provider.get_polygon_access_cost(from_pos, polygon_id, svc_time)
        except Exception:
            cost_val = float('inf')
        access_cost_cache[key] = float(cost_val)
        return access_cost_cache[key]

    rng = random.Random(rng_seed)

                                                                                                    
    rep_embeddings: Dict[int, np.ndarray] = {}
    courier_embeddings: Dict[int, np.ndarray] = {}
    if use_graph_embedding and hasattr(distance_provider, 'polygon_info'):
        try:
            rep_set = set(int(p) for p in available_polygons)
            reps = sorted(list(rep_set))
            idx_of: Dict[int, int] = {int(p): i for i, p in enumerate(reps)}
            n = len(reps)
                                                            
            if n >= 3 and embedding_dim > 0 and n <= 400:
                                                        
                W = np.zeros((n, n), dtype=float)
                portals = [int(distance_provider.polygon_info.get(int(p), {}).get('portal_id', 0)) for p in reps]
                                           
                for i in range(n):
                    pi = portals[i]
                    if not pi:
                        continue
                    for j in range(i + 1, n):
                        pj = portals[j]
                        if not pj:
                            continue
                        try:
                            d = float(distance_provider.get_distance_between_ports(int(pi), int(pj)))
                        except Exception:
                            d = float('inf')
                        if 0 < d < float(graph_distance_threshold):
                            w = 1.0 / (1.0 + d)
                            W[i, j] = w
                            W[j, i] = w
                D = np.diag(np.sum(W, axis=1))
                L = D - W
                                                                      
                with np.errstate(divide='ignore'):
                    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.sum(W, axis=1), 1e-9)))
                Lsym = D_inv_sqrt @ L @ D_inv_sqrt
                                                                         
                try:
                    vals, vecs = np.linalg.eigh(Lsym)
                                                                
                    k = min(embedding_dim + 1, n)
                    emb = vecs[:, 1:k]
                               
                    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
                    for i, p in enumerate(reps):
                        rep_embeddings[int(p)] = emb_norm[i]
                except Exception:
                    rep_embeddings = {}
                                                                                             
            for cid in courier_ids:
                weights = []
                vectors = []
                st_map = courier_service_times.get(int(cid), {})
                for p in reps:
                    if p not in rep_embeddings:
                        continue
                    st = st_map.get(int(p))
                    if st is None or st <= 0:
                        continue
                    w = 1.0 / float(st)
                    weights.append(w)
                    vectors.append(rep_embeddings[int(p)])
                if vectors and sum(weights) > 0:
                    V = np.stack(vectors, axis=0)
                    w = np.array(weights).reshape(-1, 1)
                    v = np.sum(V * w, axis=0) / (np.sum(w) + 1e-9)
                    v = v / (np.linalg.norm(v) + 1e-9)
                    courier_embeddings[int(cid)] = v
                else:
                    courier_embeddings[int(cid)] = np.zeros((max(1, embedding_dim),), dtype=float)
        except Exception:
            rep_embeddings = {}
            courier_embeddings = {}

                                                                
                                            
    courier_heaps: Dict[int, List[Tuple[float, int]]] = {int(cid): [] for cid in courier_ids}
                                                      
    bandit_counts: Dict[int, Dict[int, int]] = {int(cid): {} for cid in courier_ids}
    bandit_means: Dict[int, Dict[int, float]] = {int(cid): {} for cid in courier_ids}
    plays_total: Dict[int, int] = {int(cid): 0 for cid in courier_ids}

                                                                       
    seed_pool = available_polygons[:candidate_pool_size] if candidate_pool_size and candidate_pool_size > 0 else available_polygons
    for cid in courier_ids:
        current_pos = warehouse_id
        for pid in seed_pool:
            svc = courier_service_times.get(int(cid), {}).get(int(pid), 0)
            cost = get_access_cost(current_pos, int(pid), int(svc))
            if cost >= float('inf'):
                continue
            orders = order_count_map.get(int(pid), 1)
            penalty = float(per_order_penalty) * float(orders)
            reward_prior = max(0.0, penalty - float(cost))
            bandit_counts[int(cid)][int(pid)] = 0
            bandit_means[int(cid)][int(pid)] = float(reward_prior)
            ucb = bandit_means[int(cid)][int(pid)] + ucb_c * math.sqrt(math.log(plays_total[int(cid)] + 1.0) / (bandit_counts[int(cid)][int(pid)] + 1.0))
            if use_graph_embedding and int(pid) in rep_embeddings and int(cid) in courier_embeddings:
                ve = rep_embeddings[int(pid)]
                vc = courier_embeddings[int(cid)]
                sim = float(np.dot(ve, vc))
                ucb += float(embedding_beta) * sim
            heapq.heappush(courier_heaps[int(cid)], (-ucb, int(pid)))

    inactive_couriers = set()

    while available_set:
                                                           
        active_couriers = [c for c in courier_state.keys() if c not in inactive_couriers]
        if not active_couriers:
            break
        courier_id = min(active_couriers, key=lambda c: courier_state[c]['time'])
        current_pos = courier_state[courier_id]['position']
        current_time = courier_state[courier_id]['time']

        chosen_pid = None
        chosen_cost = None

        heap_ref = courier_heaps.get(int(courier_id))
                                                                      
        valid_candidates: List[Tuple[int, float]] = []               
        temp_buffer: List[Tuple[float, int]] = []
        while heap_ref and len(valid_candidates) < max(1, int(top_k)):
            score, pid = heapq.heappop(heap_ref)
            if pid not in available_set:
                continue
            svc = courier_service_times.get(int(courier_id), {}).get(int(pid), 0)
            cost = get_access_cost(int(current_pos), int(pid), int(svc))
            if cost >= float('inf') or current_time + cost > max_time_per_courier:
                continue
            orders = order_count_map.get(int(pid), 1)
            penalty = float(per_order_penalty) * float(orders)
            delta = float(cost) - penalty
            if delta >= 0:
                continue
            valid_candidates.append((int(pid), float(cost)))
            temp_buffer.append((score, int(pid)))
                                                                                
        for item in temp_buffer:
            heapq.heappush(heap_ref, item)

                                                    
        if valid_candidates:
            if rng.random() < float(epsilon) and len(valid_candidates) > 1:
                pid, cost = rng.choice(valid_candidates)
                chosen_pid = int(pid)
                chosen_cost = int(cost)
            else:
                                                                       
                best_tuple = None
                best_score = -float('inf')
                for pid, cost in valid_candidates:
                    orders = order_count_map.get(int(pid), 1)
                    penalty = float(per_order_penalty) * float(orders)
                    utility = max(0.0, penalty - float(cost))
                                                           
                    sim = 0.0
                    if use_graph_embedding and int(pid) in rep_embeddings and int(courier_id) in courier_embeddings:
                        ve = rep_embeddings[int(pid)]
                        vc = courier_embeddings[int(courier_id)]
                        sim = float(np.dot(ve, vc))
                                                                            
                    time_ratio = (float(current_time) + float(cost)) / float(max(1, max_time_per_courier))
                    fairness_penalty = float(fairness_beta) * max(0.0, time_ratio - 0.8)
                    score = utility + float(embedding_beta) * sim - fairness_penalty
                    if score > best_score:
                        best_score = score
                        best_tuple = (int(pid), float(cost))
                if best_tuple is not None:
                    chosen_pid = int(best_tuple[0])
                    chosen_cost = int(best_tuple[1])

                                                                                                      
        if chosen_pid is None:
            refill_added = 0
            for pid in seed_pool:
                if pid not in available_set:
                    continue
                svc = courier_service_times.get(int(courier_id), {}).get(int(pid), 0)
                cost = get_access_cost(int(current_pos), int(pid), int(svc))
                if cost >= float('inf') or current_time + cost > max_time_per_courier:
                    continue
                orders = order_count_map.get(int(pid), 1)
                penalty = float(per_order_penalty) * float(orders)
                delta = float(cost) - penalty
                if delta >= 0:
                    continue
                utility = -delta
                heapq.heappush(heap_ref, (-utility, int(pid)))
                refill_added += 1
                if refill_added >= max(50, min(candidate_pool_size, 500)):
                    break
                                                          
            while heap_ref and heap_ref:
                _, pid = heapq.heappop(heap_ref)
                if pid not in available_set:
                    continue
                svc = courier_service_times.get(int(courier_id), {}).get(int(pid), 0)
                cost = get_access_cost(int(current_pos), int(pid), int(svc))
                if cost >= float('inf') or current_time + cost > max_time_per_courier:
                    continue
                chosen_pid = int(pid)
                chosen_cost = int(cost)
                break

        if chosen_pid is None:
                                                                                   
            inactive_couriers.add(courier_id)
                                            
            if len(inactive_couriers) == len(courier_state):
                break
            continue

                                                      
        pid = chosen_pid
        cost = chosen_cost
                                                             
        try:
            orders = order_count_map.get(int(pid), 1)
            penalty = float(per_order_penalty) * float(orders)
            reward = max(0.0, penalty - float(cost))
            bandit_counts[int(courier_id)][int(pid)] = bandit_counts[int(courier_id)].get(int(pid), 0) + 1
            n = bandit_counts[int(courier_id)][int(pid)]
            m = bandit_means[int(courier_id)].get(int(pid), 0.0)
            bandit_means[int(courier_id)][int(pid)] = m + (reward - m) / float(n)
            plays_total[int(courier_id)] += 1
        except Exception:
            pass
        try:
            if hasattr(distance_provider, 'find_best_port_to_polygon'):
                best_port, _ = distance_provider.find_best_port_to_polygon(current_pos, pid)
                courier_state[courier_id]['position'] = best_port if best_port else distance_provider.polygon_info.get(pid, {}).get('portal_id', current_pos)
            else:
                courier_state[courier_id]['position'] = distance_provider.polygon_info.get(pid, {}).get('portal_id', current_pos)
        except Exception:
            pass
        courier_state[courier_id]['time'] += int(cost)
        courier_state[courier_id]['assigned'].append(pid)
        if pid in available_set:
            available_set.remove(pid)
        assigned_so_far = total_to_assign - len(available_set)
        if assigned_so_far % 100 == 0:
            logger.info(f"Warm-start: назначено {assigned_so_far}/{total_to_assign} полигонов")

    return {int(cid): st['assigned'] for cid, st in courier_state.items()}



def assignment_local_search(
    assignment: Dict[int, List[int]],
    couriers_df: pl.DataFrame,
    distance_provider,
    courier_service_times: Dict[int, Dict[int, int]],
    mp_to_order_count: Dict[int, int],
    warehouse_id: int,
    max_time_per_courier: int,
    max_iters: int = 2,
    sample_per_courier: int = 40,
) -> Dict[int, List[int]]:
    """Простой локальный поиск по назначению: relocate и swap между курьерами.
    Стоимость оцениваем как суммарный доступ от склада к полигону (портал+внутри+сервис).
    """
    courier_ids: List[int] = couriers_df['ID'].to_list() if 'ID' in couriers_df.columns else list(assignment.keys())

    def polygon_cost_for_courier(cid: int, pid: int) -> float:
        st = courier_service_times.get(int(cid), {}).get(int(pid))
        if st is None:
            return float('inf')
        svc = int(st) * int(mp_to_order_count.get(int(pid), 0))
        try:
            return float(distance_provider.get_polygon_access_cost(int(warehouse_id), int(pid), int(svc)))
        except Exception:
            return float('inf')

                            
    courier_time: Dict[int, float] = {int(cid): 0.0 for cid in assignment.keys()}
    for cid, plist in assignment.items():
        for pid in plist:
            courier_time[int(cid)] += polygon_cost_for_courier(int(cid), int(pid))

    for _ in range(max_iters):
        improved = False
                                                                
        for a in courier_ids:
            plist = assignment.get(int(a), [])
            if not plist:
                continue
                                             
            scored = [(int(p), polygon_cost_for_courier(int(a), int(p))) for p in plist]
            scored.sort(key=lambda x: x[1], reverse=True)
            for pid, cost_a in scored[:sample_per_courier]:
                best_b = None
                best_delta = 0.0
                best_new_time_a = 0.0
                best_new_time_b = 0.0
                for b in courier_ids:
                    if int(b) == int(a):
                        continue
                    cost_b = polygon_cost_for_courier(int(b), int(pid))
                    if cost_b >= float('inf'):
                        continue
                    new_time_a = courier_time[int(a)] - float(cost_a)
                    new_time_b = courier_time[int(b)] + float(cost_b)
                    if new_time_a <= max_time_per_courier and new_time_b <= max_time_per_courier:
                        delta = float(cost_b) - float(cost_a)
                        if delta < best_delta:
                            best_delta = float(delta)
                            best_b = int(b)
                            best_new_time_a = float(new_time_a)
                            best_new_time_b = float(new_time_b)
                if best_b is not None and best_delta < 0.0:
                                       
                    assignment[int(a)].remove(int(pid))
                    assignment.setdefault(int(best_b), []).append(int(pid))
                    courier_time[int(a)] = best_new_time_a
                    courier_time[int(best_b)] = best_new_time_b
                    improved = True
        if improved:
            continue
                                          
        for a in courier_ids:
            plist_a = assignment.get(int(a), [])
            if not plist_a:
                continue
            scored_a = [(int(p), polygon_cost_for_courier(int(a), int(p))) for p in plist_a]
            scored_a.sort(key=lambda x: x[1], reverse=True)
            for b in courier_ids:
                if int(b) == int(a):
                    continue
                plist_b = assignment.get(int(b), [])
                if not plist_b:
                    continue
                scored_b = [(int(q), polygon_cost_for_courier(int(b), int(q))) for q in plist_b]
                scored_b.sort(key=lambda x: x[1], reverse=True)
                best_pair = None
                best_delta = 0.0
                best_new_time_a = 0.0
                best_new_time_b = 0.0
                for pid, cost_a in scored_a[: min(sample_per_courier, len(scored_a))]:
                    for qid, cost_b in scored_b[: min(sample_per_courier, len(scored_b))]:
                        cost_a_q = polygon_cost_for_courier(int(a), int(qid))
                        cost_b_p = polygon_cost_for_courier(int(b), int(pid))
                        if cost_a_q >= float('inf') or cost_b_p >= float('inf'):
                            continue
                        new_time_a = courier_time[int(a)] - float(cost_a) + float(cost_a_q)
                        new_time_b = courier_time[int(b)] - float(cost_b) + float(cost_b_p)
                        if new_time_a <= max_time_per_courier and new_time_b <= max_time_per_courier:
                            delta = (float(cost_a_q) - float(cost_a)) + (float(cost_b_p) - float(cost_b))
                            if delta < best_delta:
                                best_delta = float(delta)
                                best_pair = (int(pid), int(qid))
                                best_new_time_a = float(new_time_a)
                                best_new_time_b = float(new_time_b)
                if best_pair is not None and best_delta < 0.0:
                    pid, qid = best_pair
                    assignment[int(a)].remove(int(pid))
                    assignment[int(b)].remove(int(qid))
                    assignment[int(a)].append(int(qid))
                    assignment[int(b)].append(int(pid))
                    courier_time[int(a)] = best_new_time_a
                    courier_time[int(b)] = best_new_time_b
                    improved = True
        if not improved:
            break

    return assignment

