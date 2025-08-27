from typing import Dict, List, Tuple
import polars as pl


def assignment_objective(
    ass: Dict[int, List[int]],
    provider,
    courier_service_times: Dict[int, Dict[int, int]],
    mp_to_order_count: Dict[int, int],
    optimized_polygons: pl.DataFrame,
    warehouse_id: int,
) -> float:
    ct: Dict[int, float] = {int(cid): 0.0 for cid in ass.keys()}
    for cid, pids in ass.items():
        for pid in pids:
            st = courier_service_times.get(int(cid), {}).get(int(pid))
            if st is None:
                continue
            svc = int(st) * int(mp_to_order_count.get(int(pid), 0))
            try:
                ct[int(cid)] += float(provider.get_polygon_access_cost(warehouse_id, int(pid), int(svc)))
            except Exception:
                pass
    sum_time = sum(ct.values())
    all_p = set(int(r['MpId']) for r in optimized_polygons.iter_rows(named=True) if int(r['MpId']) != warehouse_id)
    assigned_p = set(int(p) for lst in ass.values() for p in lst)
    unassigned_orders = sum(int(mp_to_order_count.get(int(p), 0)) for p in (all_p - assigned_p))
    return float(sum_time) + 3000.0 * float(unassigned_orders)


def compute_assignment(
    assign_strategy: str,
    use_column_gen: bool,
    cg_use_espprc: bool,
    fast: bool,
    optimized_polygons: pl.DataFrame,
    orders_df: pl.DataFrame,
    couriers_df,
    provider,
    courier_service_times: Dict[int, Dict[int, int]],
    mp_to_order_count: Dict[int, int],
    warehouse_id: int,
    max_time_per_courier: int,
    mp_to_macro: Dict[int, int],
    representative_map: Dict[int, int],
    clusters: List[Dict],
    polygon_info_dict: Dict[int, Dict],
) -> Dict[int, List[int]]:
    from deterministic_vrp_solver.strategies.factory import build_assignment_strategy

    assignment = build_assignment_strategy(assign_strategy).assign(
        optimized_polygons,
        couriers_df,
        provider,
        courier_service_times,
        mp_to_order_count,
        warehouse_id,
        int(max_time_per_courier),
    )

    if use_column_gen:
        from deterministic_vrp_solver.column_generation import run_column_generation_assignment
        assignment_cg = run_column_generation_assignment(
            optimized_polygons,
            couriers_df,
            provider,
            courier_service_times,
            mp_to_order_count,
            warehouse_id,
            int(max_time_per_courier),
            per_order_penalty=3000,
            iterations=(2 if fast else 3),
            use_espprc=bool(cg_use_espprc),
        )
        if assignment is None or assignment_objective(
            assignment_cg, provider, courier_service_times, mp_to_order_count, optimized_polygons, warehouse_id
        ) < assignment_objective(
            assignment, provider, courier_service_times, mp_to_order_count, optimized_polygons, warehouse_id
        ):
            assignment = assignment_cg

    initial_assignment: Dict[int, List[int]] = {int(cid): [] for cid in couriers_df['ID'].to_list()}
    remaining_polygons = set(int(r['MpId']) for r in optimized_polygons.iter_rows(named=True) if int(r['MpId']) != warehouse_id)
    for cid in initial_assignment.keys():
        best_pid = None
        best_cost = float('inf')
        for pid in list(remaining_polygons):
            per_order_st = courier_service_times.get(int(cid), {}).get(int(pid))
            if per_order_st is None:
                continue
            svc = int(per_order_st) * int(mp_to_order_count.get(int(pid), 0))
            try:
                cost = provider.get_polygon_access_cost(warehouse_id, pid, svc)
            except Exception:
                cost = float('inf')
            if cost < best_cost and cost < max_time_per_courier:
                best_cost = cost
                best_pid = pid
        if best_pid is not None:
            mid = mp_to_macro.get(int(best_pid))
            rep = representative_map.get(int(mid)) if mid else None
            initial_assignment[cid].append(int(rep if rep is not None else best_pid))
            remaining_polygons.discard(best_pid)

    rep_set = set(representative_map.values())
    optimized_polygons_reps = optimized_polygons.filter(pl.col('MpId').is_in(list(rep_set)))

    from deterministic_vrp_solver.rl.warm_start import greedy_initialize_assignment, assignment_local_search
    def run_seeded(seed: int, eps: float):
        return greedy_initialize_assignment(
            optimized_polygons_reps,
            couriers_df,
            max_time_per_courier,
            provider,
            courier_service_times,
            warehouse_id=warehouse_id,
            top_k=(10 if fast else 8),
            candidate_pool_size=(50 if fast else 60),
            per_order_penalty=3000,
            epsilon=eps,
            ucb_c=1.0,
            rng_seed=seed,
        )

    seeds = [11, 23, 37]
    eps_list = [0.15, 0.25, 0.05]
    best_assignment = None
    best_objective = float('inf')
    for s, e in zip(seeds, eps_list):
        cand = run_seeded(s, e)
        obj = assignment_objective(cand, provider, courier_service_times, mp_to_order_count, optimized_polygons_reps, warehouse_id)
        if obj < best_objective:
            best_objective = obj
            best_assignment = cand

    assignment = best_assignment if best_assignment is not None else run_seeded(11, 0.15)

    assignment = assignment_local_search(
        assignment,
        couriers_df,
        provider,
        courier_service_times,
        mp_to_order_count,
        warehouse_id,
        int(max_time_per_courier),
        max_iters=(1 if fast else 2),
        sample_per_courier=30,
    )

    from deterministic_vrp_solver.alns import improve_assignment_alns
    assignment = improve_assignment_alns(
        assignment,
        provider,
        courier_service_times,
        mp_to_order_count,
        warehouse_id,
        int(max_time_per_courier),
        per_order_penalty=3000,
        iterations=(20 if fast else 35),
        overloaded_top=12,
        remove_per_overloaded=2,
    )

    for cid, seed_pids in initial_assignment.items():
        if not seed_pids:
            continue
        if cid not in assignment:
            assignment[cid] = []
        merged = list(dict.fromkeys([*seed_pids, *[int(x) for x in assignment[cid]]]))
        assignment[cid] = merged

    rep_to_best: Dict[int, int] = {}
    rep_to_best_cost: Dict[int, float] = {}
    for cid, pids in assignment.items():
        for pid in pids:
            ipid = int(pid)
            mid = mp_to_macro.get(ipid)
            rep = representative_map.get(int(mid), ipid) if mid else ipid
            per_order_st = courier_service_times.get(int(cid), {}).get(int(rep))
            if per_order_st is None:
                cst = float('inf')
            else:
                svc = int(per_order_st) * int(mp_to_order_count.get(int(rep), 0))
                try:
                    cst = float(provider.get_polygon_access_cost(warehouse_id, int(rep), int(svc)))
                except Exception:
                    cst = float('inf')
            if rep not in rep_to_best or cst < rep_to_best_cost.get(rep, float('inf')):
                rep_to_best[rep] = int(cid)
                rep_to_best_cost[rep] = cst

    for cid in list(assignment.keys()):
        filtered: List[int] = []
        for pid in assignment[cid]:
            ipid = int(pid)
            mid = mp_to_macro.get(ipid)
            rep = representative_map.get(int(mid), ipid) if mid else ipid
            if rep_to_best.get(int(rep)) == int(cid):
                filtered.append(int(rep))
        assignment[cid] = list(dict.fromkeys(filtered))

    all_reps = set(representative_map.values())
    assigned_reps = set(int(r) for reps in assignment.values() for r in reps)
    remaining_reps = [int(r) for r in all_reps if int(r) not in assigned_reps]

    courier_time: Dict[int, float] = {int(cid): 0.0 for cid in assignment.keys()}
    for cid, reps in assignment.items():
        for rep in reps:
            per_order_st = courier_service_times.get(int(cid), {}).get(int(rep))
            if per_order_st is None:
                continue
            svc = int(per_order_st) * int(mp_to_order_count.get(int(rep), 0))
            try:
                courier_time[int(cid)] += float(provider.get_polygon_access_cost(warehouse_id, int(rep), int(svc)))
            except Exception:
                continue

    for rep in remaining_reps:
        best_cid = None
        best_new_time = float('inf')
        for cid in assignment.keys():
            per_order_st = courier_service_times.get(int(cid), {}).get(int(rep))
            if per_order_st is None:
                continue
            svc = int(per_order_st) * int(mp_to_order_count.get(int(rep), 0))
            try:
                add = float(provider.get_polygon_access_cost(warehouse_id, int(rep), int(svc)))
            except Exception:
                add = float('inf')
            orders_rep = int(mp_to_order_count.get(int(rep), 0))
            penalty_gain = 3000.0 * float(orders_rep)
            if not (add < penalty_gain):
                continue
            new_time = courier_time[int(cid)] + add
            if add < float('inf') and new_time <= float(max_time_per_courier) and new_time < best_new_time:
                best_new_time = new_time
                best_cid = int(cid)
        if best_cid is not None:
            assignment[best_cid].append(int(rep))
            courier_time[best_cid] = best_new_time

    portal_route_by_macro = {c['macro_id']: [int(p) for p in c['micro_route_portals']] for c in clusters}
                                                      
    macro_to_members = {c['macro_id']: [int(x) for x in c['members']] for c in clusters}
    for cid, pids in list(assignment.items()):
        expanded: List[int] = []
        for pid in pids:
            mid = mp_to_macro.get(int(pid))
            if not mid:
                expanded.append(int(pid))
                continue
            route_portals = portal_route_by_macro.get(int(mid), [])
            full_members = [int(m) for m in macro_to_members.get(int(mid), [])]
            members_by_portal: Dict[int, List[int]] = {}
            for m in full_members:
                prt = int(polygon_info_dict.get(int(m), {}).get('portal_id', 0))
                members_by_portal.setdefault(prt, []).append(int(m))
            ordered: List[int] = []
            seen_local = set()
            for prt in route_portals:
                for m in members_by_portal.get(int(prt), []):
                    if m not in seen_local:
                        ordered.append(int(m))
                        seen_local.add(int(m))
            for m in full_members:
                if int(m) not in seen_local:
                    ordered.append(int(m))
                    seen_local.add(int(m))
            expanded.extend(int(m) for m in ordered)
        assignment[cid] = expanded

    seen_polygons: set[int] = set()
    for cid in list(assignment.keys()):
        unique_list: List[int] = []
        for pid in assignment[cid]:
            ipid = int(pid)
            if ipid in seen_polygons:
                continue
            seen_polygons.add(ipid)
            unique_list.append(ipid)
        assignment[cid] = unique_list

    all_polygons_set = set(int(r['MpId']) for r in optimized_polygons.iter_rows(named=True) if int(r['MpId']) != warehouse_id)
    assigned_polygons_set = set(int(p) for plist in assignment.values() for p in plist)
    leftover_polygons = [int(p) for p in all_polygons_set if int(p) not in assigned_polygons_set]

    courier_time_fill: Dict[int, float] = {int(cid): 0.0 for cid in assignment.keys()}
    for cid, plist in assignment.items():
        for pid in plist:
            per_order_st = courier_service_times.get(int(cid), {}).get(int(pid))
            if per_order_st is None:
                continue
            svc = int(per_order_st) * int(mp_to_order_count.get(int(pid), 0))
            try:
                courier_time_fill[int(cid)] += float(provider.get_polygon_access_cost(warehouse_id, int(pid), int(svc)))
            except Exception:
                continue

    for pid in leftover_polygons:
        best_cid = None
        best_new_time = float('inf')
        for cid in assignment.keys():
            per_order_st = courier_service_times.get(int(cid), {}).get(int(pid))
            if per_order_st is None:
                continue
            svc = int(per_order_st) * int(mp_to_order_count.get(int(pid), 0))
            try:
                add = float(provider.get_polygon_access_cost(warehouse_id, int(pid), int(svc)))
            except Exception:
                add = float('inf')
            orders_pid = int(mp_to_order_count.get(int(pid), 0))
            penalty_gain = 3000.0 * float(orders_pid)
            if not (add < penalty_gain):
                continue
            new_time = courier_time_fill[int(cid)] + add
            if add < float('inf') and new_time <= float(max_time_per_courier) and new_time < best_new_time:
                best_new_time = new_time
                best_cid = int(cid)
        if best_cid is not None:
            assignment[best_cid].append(int(pid))
            courier_time_fill[best_cid] = best_new_time

    return assignment


