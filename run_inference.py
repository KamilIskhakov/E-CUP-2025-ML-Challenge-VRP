import argparse
import os
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List

import polars as pl
import statistics

from deterministic_vrp_solver.utils import load_couriers_data, aggregate_orders_by_polygon_lazy, calculate_polygon_portal
from deterministic_vrp_solver.polygon.optimizer import optimize_all_polygons_hybrid
from deterministic_vrp_solver.decomposed_distance_provider import DecomposedDistanceProvider
from deterministic_vrp_solver.rl.lagrangian_assignment import lagrangian_time_balanced_assignment
from deterministic_vrp_solver.aggregation.super_polygons import build_super_polygons
from deterministic_vrp_solver.services.db import get_fast_db_connection
from deterministic_vrp_solver.services.orders import load_orders_lazy
from deterministic_vrp_solver.services.ports_db import build_ports_database, build_warehouse_ports_database
from deterministic_vrp_solver.services.polygons import build_polygon_info
from deterministic_vrp_solver.pipeline.assignment import compute_assignment
from deterministic_vrp_solver.pipeline.routes import optimize_and_improve_routes
from deterministic_vrp_solver.services.report import build_validation_report, save_report


logger = logging.getLogger("inference")


def get_fast_db_connection(db_path: str) -> sqlite3.Connection:
    from deterministic_vrp_solver.services.db import get_fast_db_connection as _get
    return _get(db_path)


def build_ports_database(ports_db_path: Path, polygon_ports: Dict[int, List[int]], durations_conn: sqlite3.Connection) -> None:
    from deterministic_vrp_solver.services.ports_db import build_ports_database as _build
    _build(ports_db_path, polygon_ports, durations_conn)


def build_warehouse_ports_database(warehouse_ports_db_path: Path, polygon_ports: Dict[int, List[int]], durations_conn: sqlite3.Connection, warehouse_id: int) -> None:
    from deterministic_vrp_solver.services.ports_db import build_warehouse_ports_database as _build_wh
    _build_wh(warehouse_ports_db_path, polygon_ports, durations_conn, warehouse_id)


def load_orders_lazy(orders_path: Path) -> pl.LazyFrame:
    from deterministic_vrp_solver.services.orders import load_orders_lazy as _load
    return _load(orders_path)


def build_polygon_info(optimized_polygons: pl.DataFrame, polygon_ports: Dict[int, List[int]], conn: sqlite3.Connection) -> Dict[int, Dict]:
    from deterministic_vrp_solver.services.polygons import build_polygon_info as _build_info
    return _build_info(optimized_polygons, polygon_ports, conn)


def generate_solution_structure(route_optimizer: Dict[int, Dict], optimized_polygons: pl.DataFrame, conn: sqlite3.Connection) -> Dict:
    from deterministic_vrp_solver.services.solution import generate_solution_structure as _gen
    return _gen(route_optimizer, optimized_polygons)


def main():
    parser = argparse.ArgumentParser(description="Deterministic VRP inference (no RL)")
    parser.add_argument("--orders", type=str, required=True, help="Path to dataSetOrders.json")
    parser.add_argument("--couriers", type=str, required=True, help="Path to dataSetCouriers.json")
    parser.add_argument("--durations_sqlite", type=str, required=True, help="Path to durations.sqlite")
    parser.add_argument("--output", type=str, default="solution.json", help="Output solution JSON path")
    parser.add_argument("--max_polygons", type=int, default=0, help="Max polygons to process (0 = all)")
    parser.add_argument("--max_time_per_courier", type=int, default=43200, help="Time limit per courier (sec)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Parallel workers for polygon TSP")
    parser.add_argument("--max_couriers", type=int, default=0, help="Max couriers to use (0 = all)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: aggressive aggregation, skip Lagrange, reuse port DBs if present")
    parser.add_argument("--print_lb", action="store_true", help="Print quick lower bound estimate before assignment")
    parser.add_argument("--reuse_port_dbs", action="store_true", help="Reuse existing ports/warehouse DBs if present (skip rebuild)")
    parser.add_argument("--use_column_gen", action="store_true", help="Use column generation (pricing+RCSP) for macro assignment")
    parser.add_argument("--assign_strategy", type=str, default="warm", help="Assignment strategy: warm|rcsp|cg")
    parser.add_argument("--post_improvers", type=str, default="cross,alns", help="Comma-separated post improvers: cross,alns,portals")
    parser.add_argument("--sectors", type=int, default=0, help="Split polygons by angle sectors (0=off)")
    parser.add_argument("--sector_policy_ckpt", type=str, default="", help="Path to torch ckpt for sector policy (optional)")
    parser.add_argument("--cg_use_espprc", action="store_true", help="Use ESPPRC pricing inside column generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    start_time = time.time()

    orders_path = Path(args.orders)
    couriers_path = Path(args.couriers)
    db_path = Path(args.durations_sqlite)
    output_path = Path(args.output)

    if not orders_path.exists() or not couriers_path.exists() or not db_path.exists():
        raise FileNotFoundError("Input files not found. Check --orders, --couriers, --durations_sqlite")

    logger.info("=== Шаг 1: Загрузка данных ===")
    orders_lf = load_orders_lazy(orders_path)
    couriers_df, warehouse_info = load_couriers_data(str(couriers_path))
    if args.max_couriers and args.max_couriers > 0 and len(couriers_df) > args.max_couriers:
        couriers_df = couriers_df.head(args.max_couriers)

    conn = get_fast_db_connection(str(db_path))

                                 
    logger.info("=== Шаг 2: Агрегация полигонов ===")
    orders_lf_filtered = orders_lf.filter(pl.col('MpId') != 0)
    polygon_stats = aggregate_orders_by_polygon_lazy(orders_lf_filtered)
    if args.max_polygons and args.max_polygons > 0:
        polygon_stats = polygon_stats.head(args.max_polygons)

                                                                                        
                                                                                                                
    polygon_ports: Dict[int, List[int]] = {}
    for row in polygon_stats.iter_rows(named=True):
        order_ids = row['order_ids']
        if isinstance(order_ids, list):
            ports = list(order_ids)
        else:
            ports = []
        polygon_ports[row['MpId']] = ports

    work_dir = Path("deterministic_vrp_solver") / "data"
    work_dir.mkdir(parents=True, exist_ok=True)
    ports_db_path = work_dir / "ports_database.sqlite"
    warehouse_ports_db_path = work_dir / "warehouse_ports_database.sqlite"
    logger.info("=== Шаг 3: Построение БД портов ===")
    reuse = args.reuse_port_dbs or args.fast
    if reuse and ports_db_path.exists() and warehouse_ports_db_path.exists():
        logger.info("Переиспользуем существующие ports_database.sqlite и warehouse_ports_database.sqlite")
    else:
        build_ports_database(ports_db_path, polygon_ports, conn)
                                                                                
        build_warehouse_ports_database(warehouse_ports_db_path, polygon_ports, conn, 0)
                                                   
    warehouse_id = 0

    logger.info("=== Шаг 4: Расчёт TSP внутри полигонов ===")
    optimized_polygons = optimize_all_polygons_hybrid(polygon_stats, str(db_path), service_times={}, max_workers=int(args.workers))
                                           
    optimized_polygons = optimized_polygons.filter(pl.col('total_cost') <= args.max_time_per_courier)

                                
    logger.info("=== Шаг 5: Расчёт порталов полигонов ===")
    orders_df = orders_lf.collect()
    if 'portal_id' not in optimized_polygons.columns:
        optimized_polygons = optimized_polygons.with_columns(pl.lit(0).alias('portal_id'))
    for row in optimized_polygons.iter_rows(named=True):
        mp_id = row['MpId']
        order_ids = row['order_ids']
        polygon_orders = orders_df.filter(pl.col('MpId') == mp_id)
        lats = polygon_orders['Lat'].to_list()
        longs = polygon_orders['Long'].to_list()
        portal_id = calculate_polygon_portal(conn, order_ids, lats, longs)
        optimized_polygons = optimized_polygons.with_columns(
            pl.when(pl.col('MpId') == mp_id).then(pl.lit(portal_id)).otherwise(pl.col('portal_id')).alias('portal_id')
        )

                                                                                     
    mp_to_order_count: Dict[int, int] = {}
    for row in optimized_polygons.iter_rows(named=True):
        mp_id_i = int(row['MpId'])
        order_ids = row.get('order_ids')
        cnt = len(order_ids) if isinstance(order_ids, list) else int(row.get('order_count', 0) or 0)
        mp_to_order_count[mp_id_i] = int(cnt)

                                                                     
    with open(couriers_path, 'r') as f:
        couriers_data = json.load(f)
    courier_service_times: Dict[int, Dict[int, int]] = {}
    per_order_times_by_mp: Dict[int, List[int]] = {}
    for courier in couriers_data['Couriers']:
        cid = int(courier['ID'])
        courier_service_times[cid] = {}
        st_list = courier.get('ServiceTimeInMps') or []
        for mp in st_list:
            mpid = int(mp['MpID'])
            st = int(mp['ServiceTime'])
            courier_service_times[cid][mpid] = st
            per_order_times_by_mp.setdefault(mpid, []).append(st)

                                                                                                  
    service_time_by_mp_total: Dict[int, int] = {}
    for mpid, times in per_order_times_by_mp.items():
        if not times:
            continue
        try:
            median_st = statistics.median(times)
        except Exception:
            median_st = sum(times) / len(times)
        service_time_by_mp_total[int(mpid)] = int(median_st) * int(mp_to_order_count.get(int(mpid), 0))

    logger.info("=== Шаг 6: Агрегация в супер-полигоны ===")
                                                          
    clusters = build_super_polygons(
        optimized_polygons,
        ports_db_path=str(ports_db_path),
        d_portal_max=(700 if args.fast else 900),
        k_neighbors=(8 if args.fast else 10),
        max_polygons_per_cluster=(4 if args.fast else 5),
        max_orders_per_cluster=120,
        max_internal_distance=(1200 if args.fast else 1500),
        service_time_by_mp=service_time_by_mp_total,
    )

                              
    macro_to_members = {c['macro_id']: [int(x) for x in c['members']] for c in clusters}
                                     
    mp_to_macro = {}
    for mid, members in macro_to_members.items():
        for mp in members:
            mp_to_macro[int(mp)] = int(mid)
                                                                            
    representative_map = {}
    for mid, members in macro_to_members.items():
        if members:
            representative_map[mid] = int(members[0])

    logger.info(f"Собрано супер-полигонов: {len(macro_to_members)} (из {len(optimized_polygons)})")

    logger.info("=== Шаг 7: Подготовка информации для декомпозиции ===")

                                                                                           
    if args.print_lb:
        try:
            lb_total = 0
                                                    
            mp_min_svc: Dict[int, int] = {}
            for cid, mp_map in courier_service_times.items():
                for mp_id, st in mp_map.items():
                    cur = mp_min_svc.get(int(mp_id))
                    val = int(st)
                    if cur is None or val < cur:
                        mp_min_svc[int(mp_id)] = val
            counted = 0
            for row in optimized_polygons.iter_rows(named=True):
                mp_id = int(row.get('MpId'))
                if mp_id == warehouse_id:
                    continue
                base_cost = int(row.get('total_distance', 0) or 0)
                orders = 0
                try:
                    orders_list = row.get('order_ids')
                    if isinstance(orders_list, list):
                        orders = len(orders_list)
                    else:
                        orders = int(row.get('order_count', 0) or 0)
                except Exception:
                    orders = int(row.get('order_count', 0) or 0)
                svc_min = int(mp_min_svc.get(int(mp_id), 0))
                lb_total += int(base_cost) + int(svc_min) * int(orders)
                counted += 1
            logger.info(f"Быстрый нижний предел (LB): {int(lb_total)} сек по {counted} полигонам")
        except Exception as e:
            logger.warning(f"Не удалось вычислить нижний предел: {e}")
                                                                                                       
    polygon_info_dict = build_polygon_info(optimized_polygons, polygon_ports, conn)

    provider = DecomposedDistanceProvider(str(db_path), str(ports_db_path), str(warehouse_ports_db_path))
    provider.__enter__()
                                                 
    provider.set_warehouse_id(0)
    provider.set_polygon_info(polygon_info_dict)

                                                                  

    logger.info("=== Шаг 8: Инициализация назначений и жадное дораспределение (на макро-узлах) ===")
                                                                           
    polygon_to_sector = None
    sector_to_polygons = None
    if int(args.sectors) and int(args.sectors) > 0:
        try:
            from deterministic_vrp_solver.sectorization import split_into_sectors_by_angle
                                                                                                  
            centers = None
            if args.sector_policy_ckpt:
                try:
                    import importlib.util, os as _os
                    if _os.path.exists(args.sector_policy_ckpt) and importlib.util.find_spec("torch") is not None:
                        import torch                
                        from deterministic_vrp_solver.rl.sector_policy import SectorPolicy, build_angle_histogram, select_sector_centers                
                        wh_lat = float(warehouse_info['Warehouse']['Lat'])
                        wh_lon = float(warehouse_info['Warehouse']['Long'])
                        hist = build_angle_histogram(orders_df, wh_lat, wh_lon, num_bins=72)
                        model = SectorPolicy(num_bins=72, num_sectors=int(args.sectors))
                        model.load_state_dict(torch.load(args.sector_policy_ckpt, map_location='cpu'))
                        centers = select_sector_centers(model, hist, top_per_sector=1)
                        logger.info(f"Сектор-политика загрузилась; центры бинов: {centers}")
                    else:
                        logger.info("Torch не найден или чекпойнт отсутствует — используем геометрическую секторизацию.")
                except Exception as e:
                    centers = None
                    logger.warning(f"Сектор-политика не применена: {e}")
                                                             
            wh_lat = float(warehouse_info['Warehouse']['Lat'])
            wh_lon = float(warehouse_info['Warehouse']['Long'])
            polygon_to_sector, sector_to_polygons, sector_orders = split_into_sectors_by_angle(
                optimized_polygons,
                orders_df,
                wh_lat,
                wh_lon,
                target_sectors=int(args.sectors),
            )
            logger.info(f"Секторизация активна: {int(args.sectors)} секторов, распределение заказов: {sector_orders}")
        except Exception as e:
            logger.warning(f"Секторизация отключена из-за ошибки: {e}")
                                        
    assignment = compute_assignment(
        assign_strategy=str(args.assign_strategy),
        use_column_gen=bool(args.use_column_gen),
        cg_use_espprc=bool(args.cg_use_espprc),
        fast=bool(args.fast),
        optimized_polygons=optimized_polygons,
        orders_df=orders_df,
        couriers_df=couriers_df,
        provider=provider,
        courier_service_times=courier_service_times,
        mp_to_order_count=mp_to_order_count,
        warehouse_id=warehouse_id,
        max_time_per_courier=int(args.max_time_per_courier),
        mp_to_macro=mp_to_macro,
        representative_map=representative_map,
        clusters=clusters,
        polygon_info_dict=polygon_info_dict,
    )

                                     
                                               

                                                       
    if not args.fast:
        try:
            logger.info("=== Шаг 8: Лагранжев ребаланс назначений ===")
            courier_ids = couriers_df['ID'].to_list()
            polygon_ids = [int(r['MpId']) for r in optimized_polygons.iter_rows(named=True)]
            m = len(courier_ids)
            n = len(polygon_ids)
            import numpy as np
            base_costs = np.full((m, n), np.inf, dtype=float)
            times = np.full((m, n), np.inf, dtype=float)
            time_budgets = np.full((m,), float(args.max_time_per_courier))
                                                             
            for i, cid in enumerate(courier_ids):
                for j, pid in enumerate(polygon_ids):
                    svc = courier_service_times.get(int(cid), {}).get(int(pid), 0)
                    cost = provider.get_polygon_access_cost(warehouse_id, int(pid), int(svc))
                    if cost < float('inf') and cost <= args.max_time_per_courier:
                        base_costs[i, j] = float(cost)
                        times[i, j] = float(cost)                                                   
            match, _ = lagrangian_time_balanced_assignment(base_costs, times, time_budgets, max_iters=8, step0=50.0, step_decay=0.9)
                                                                                    
            reb_assignment: Dict[int, List[int]] = {int(cid): [] for cid in courier_ids}
            for i, j in match:
                reb_assignment[int(courier_ids[i])].append(int(polygon_ids[j]))
                                                                                                    
            for cid in reb_assignment:
                if reb_assignment[cid]:
                    assignment[cid] = reb_assignment[cid]
            logger.info("Лагранжев ребаланс применён")
        except Exception as e:
            logger.warning(f"Лагранжев ребаланс пропущен: {e}")
    else:
        logger.info("Fast режим: пропускаем лагранжев ребаланс")

    logger.info("=== Шаг 9: Оптимизация маршрутов курьеров ===")
                                            
                                                                                                      
    improved_routes, final_routes = optimize_and_improve_routes(
        optimized_polygons=optimized_polygons,
        assignment=assignment,
        courier_service_times=courier_service_times,
        provider=provider,
        conn=conn,
        warehouse_id=warehouse_id,
        max_time_per_courier=int(args.max_time_per_courier),
        post_improvers=str(args.post_improvers),
        fast=bool(args.fast),
    )

    from deterministic_vrp_solver.services.solution import save_solution
    save_solution(output_path, final_routes)
    logger.info(f"Solution saved to {output_path} (routes with full order sequence)")

                        
    try:
        route_times = [int(v.get('total_time', 0)) for v in improved_routes.values() if v]
        max_route_time = max(route_times) if route_times else 0
        total_route_time = sum(route_times)
        avg_route_time = int(total_route_time / len(route_times)) if route_times else 0
        active_couriers = sum(1 for v in improved_routes.values() if v and v.get('polygon_order'))
        total_couriers = len(couriers_df)
        exec_time = time.time() - start_time

        logger.info("=== Результаты ===")
        logger.info(f"Время выполнения: {exec_time:.2f} секунд")
        logger.info(f"Курьеров: {total_couriers}")
        logger.info(f"Полигонов: {len(optimized_polygons)}")
        logger.info(f"Активных курьеров: {active_couriers}")
        logger.info(f"Максимальное время маршрута: {int(max_route_time)} сек")
        logger.info(f"Среднее время маршрута: {int(avg_route_time)} сек")
        logger.info(f"Общее время маршрутов: {int(total_route_time)} сек")

                                                      
        all_order_ids = set()
        for row in orders_df.iter_rows(named=True):
            all_order_ids.add(int(row['ID']))
        assigned_order_ids = set()
        for r in final_routes:
                                        
            assigned_order_ids.update(int(x) for x in r['route'] if int(x) != 0)
        unassigned_orders = sorted(list(all_order_ids - assigned_order_ids))
        penalty_unassigned = 3000 * len(unassigned_orders)
        from deterministic_vrp_solver.services.report import build_validation_report, save_report
        report = build_validation_report(exec_time, total_couriers, active_couriers, len(optimized_polygons), int(max_route_time), int(avg_route_time), int(total_route_time), len(unassigned_orders))
        save_report('validation_report.json', report)
        logger.info("Отчёт сохранён в validation_report.json")
    except Exception:
        pass

    provider.close()
    conn.close()


if __name__ == "__main__":
    main()
