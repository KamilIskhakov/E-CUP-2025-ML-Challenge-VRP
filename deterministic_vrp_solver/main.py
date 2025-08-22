#!/usr/bin/env python3

import os
import sys
import time
import logging
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import polars as pl
from utils import (
    load_orders_data, load_couriers_data, get_distance_matrix,
    aggregate_orders_by_polygon, calculate_polygon_portal,
    calculate_polygon_statistics, filter_polygons_by_size, optimize_polygon_processing_order,
    load_orders_data_lazy, aggregate_orders_by_polygon_lazy
)
from polygon_optimizer import optimize_all_polygons, optimize_all_polygons_mp, optimize_all_polygons_hybrid
from distance_provider import SQLiteDistanceProvider
from solver_factory import AssignmentSolverFactory
from polygon_validator import PolygonValidator
from route_optimizer import optimize_courier_routes
from solution_generator import generate_solution

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deterministic_vrp.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Запуск детерминированного VRP решателя")
    start_time = time.time()
    
    try:
        data_dir = Path("../ml_ozon_logistic")
        orders_file = data_dir / "ml_ozon_logistic_dataSetOrders.json"
        couriers_file = data_dir / "ml_ozon_logistic_dataSetCouriers.json"
        db_file = Path("../durations.sqlite")
        
        if not orders_file.exists():
            raise FileNotFoundError(f"Файл заказов не найден: {orders_file}")
        if not couriers_file.exists():
            raise FileNotFoundError(f"Файл курьеров не найден: {couriers_file}")
        if not db_file.exists():
            raise FileNotFoundError(f"База данных не найдена: {db_file}")
        
        logger.info("=== Шаг 1: Загрузка данных ===")
        
        orders_lf = load_orders_data_lazy(str(orders_file))
        couriers_df, warehouse_info = load_couriers_data(str(couriers_file))
        conn = get_distance_matrix(str(db_file))
        
        logger.info(f"Загружено {len(couriers_df)} курьеров")
        
        distance_provider = SQLiteDistanceProvider(conn)
        
        polygon_stats = aggregate_orders_by_polygon_lazy(orders_lf)
        
        filtered_polygons = filter_polygons_by_size(polygon_stats, min_size=1, max_size=20)
        
        optimized_polygon_order = optimize_polygon_processing_order(filtered_polygons)
        
        logger.info("=== Шаг 2: Оптимизация полигонов ===")
        
        orders_df = load_orders_data(str(orders_file))
        service_times = {}
        
        optimized_polygons = optimize_all_polygons_hybrid(
            optimized_polygon_order, 
            str(db_file), 
            service_times,
            max_workers=None
        )
        
        polygon_validator = PolygonValidator(max_time_per_courier=43200)
        polygon_stats_info = polygon_validator.get_polygon_statistics(optimized_polygons)
        
        cost_distribution = polygon_validator.get_polygon_cost_distribution(optimized_polygons)
        
        logger.info(f"Статистика стоимости полигонов:")
        logger.info(f"   Q25: {cost_distribution['q25']:.0f} сек")
        logger.info(f"   Q50: {cost_distribution['q50']:.0f} сек")
        logger.info(f"   Q75: {cost_distribution['q75']:.0f} сек")
        logger.info(f"   Q90: {cost_distribution['q90']:.0f} сек")
        logger.info(f"   Q95: {cost_distribution['q95']:.0f} сек")
        logger.info(f"   Q99: {cost_distribution['q99']:.0f} сек")
        
        expensive_polygons = optimized_polygons.filter(pl.col('total_cost') > 100000)
        if len(expensive_polygons) > 0:
            logger.warning(f"Найдено {len(expensive_polygons)} дорогих полигонов (>100000 сек)")
            logger.warning(f"   Максимальная стоимость: {expensive_polygons['total_cost'].max()} сек")
            logger.warning(f"   Средняя стоимость дорогих: {expensive_polygons['total_cost'].mean():.0f} сек")
        
        unassignable_polygons = polygon_validator.get_unassignable_polygons(optimized_polygons)
        if unassignable_polygons:
            logger.warning(f"Найдено {len(unassignable_polygons)} неназначаемых полигонов")
            optimized_polygons = polygon_validator.filter_assignable_polygons(optimized_polygons)
            logger.info(f"После фильтрации осталось {len(optimized_polygons)} полигонов")
        
        expensive_polygons = optimized_polygons.filter(pl.col('total_cost') > 43200)
        if len(expensive_polygons) > 0:
            logger.warning(f"Найдено {len(expensive_polygons)} полигонов дороже лимита курьера (>43200 сек)")
            logger.warning(f"   Максимальная стоимость: {expensive_polygons['total_cost'].max()} сек")
            logger.warning(f"   Средняя стоимость дорогих: {expensive_polygons['total_cost'].mean():.0f} сек")
            logger.warning(f"   Эти полигоны могут не быть назначены курьерам")
        
        logger.info(f"Оптимизировано {len(optimized_polygons)} полигонов")
        
        logger.info("Вычисление порталов полигонов...")
        for row in optimized_polygons.iter_rows(named=True):
            mp_id = row['MpId']
            order_ids = row['order_ids']
            
            polygon_orders = orders_df.filter(pl.col('MpId') == mp_id)
            lats = polygon_orders['Lat'].to_list()
            longs = polygon_orders['Long'].to_list()
            
            portal_id = calculate_polygon_portal(conn, order_ids, lats, longs)
            
            optimized_polygons = optimized_polygons.with_columns(
                pl.when(pl.col('MpId') == mp_id)
                .then(pl.lit(portal_id))
                .otherwise(pl.col('portal_id'))
                .alias('portal_id')
            )
        
        logger.info("=== Шаг 3: Назначение полигонов курьерам ===")
        
        courier_service_times = {}
        
        with open(couriers_file, 'r') as f:
            couriers_data = json.load(f)
        
        for courier in couriers_data['Couriers']:
            courier_id = courier['ID']
            courier_service_times[courier_id] = {}
            
            if 'ServiceTimeInMps' in courier and courier['ServiceTimeInMps'] is not None:
                for mp_service in courier['ServiceTimeInMps']:
                    mp_id = mp_service['MpID']
                    service_time = mp_service['ServiceTime']
                    courier_service_times[courier_id][mp_id] = service_time
        
        logger.info(f"Загружены сервисные времена для {len(courier_service_times)} курьеров")
        
        total_service_times = sum(len(times) for times in courier_service_times.values())
        logger.info(f"Всего сервисных времен: {total_service_times}")
        
        if courier_service_times:
            sample_courier = list(courier_service_times.keys())[0]
            sample_times = courier_service_times[sample_courier]
            logger.info(f"Пример курьера {sample_courier}: {len(sample_times)} сервисных времен")
            if sample_times:
                sample_mp = list(sample_times.keys())[0]
                logger.info(f"  Полигон {sample_mp}: {sample_times[sample_mp]} сек")
        
        solver = AssignmentSolverFactory.create_solver(
            solver_type='reinforcement',
            distance_provider=distance_provider,
            warehouse_id=warehouse_info['ID'],
            courier_service_times=courier_service_times
        )
        
        max_time_per_courier = 43200
        
        assignment = solver.solve(optimized_polygons, couriers_df, max_time_per_courier=max_time_per_courier)
        
        logger.info(f"Назначено {sum(len(polygons) for polygons in assignment.values())} полигонов")
        
        logger.info("=== Шаг 4: Оптимизация маршрутов курьеров ===")
        
        optimized_routes = optimize_courier_routes(
            assignment, 
            optimized_polygons, 
            conn,
            courier_service_times
        )
        
        logger.info(f"Оптимизировано {len(optimized_routes)} маршрутов")
        
        logger.info("=== Шаг 5: Генерация финального решения ===")
        
        result = generate_solution(
            optimized_routes,
            optimized_polygons,
            orders_df,
            conn,
            "solution.json"
        )
        
        logger.info("=== Результаты ===")
        logger.info(f"Время выполнения: {time.time() - start_time:.2f} секунд")
        
        if result['saved']:
            logger.info("Решение успешно сгенерировано и сохранено")
            
            stats = result['statistics']
            logger.info(f"Статистика:")
            logger.info(f"   Курьеров: {stats['total_couriers']}")
            logger.info(f"   Заказов: {stats['assigned_orders']}/{stats['total_orders']} ({stats['assignment_rate']:.2%})")
            logger.info(f"   Среднее время маршрута: {stats.get('avg_route_time', 0):.0f} сек")
            logger.info(f"   Максимальное время маршрута: {stats.get('max_route_time', 0)} сек")
            logger.info(f"   Общее время: {stats.get('total_route_time', 0)} сек")
        else:
            logger.error("Решение не прошло валидацию")
            
            validation = result['validation']
            if validation['unassigned_orders']:
                logger.error(f"Неназначенные заказы: {len(validation['unassigned_orders'])}")
            if validation['duplicate_orders']:
                logger.error(f"Дубли заказов: {len(validation['duplicate_orders'])}")
            if validation['violations']:
                logger.error(f"Нарушения: {validation['violations']}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
        raise
    finally:
        total_time = time.time() - start_time
        logger.info(f"Общее время выполнения: {total_time:.2f} секунд")

if __name__ == "__main__":
    main()
