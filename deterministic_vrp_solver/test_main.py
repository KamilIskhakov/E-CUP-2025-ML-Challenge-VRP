#!/usr/bin/env python3
"""
Тестовый запуск VRP алгоритма с микротестовыми данными
"""

import os
import sys
import time
import logging
import json
from pathlib import Path

# Добавляем текущую директорию в путь для импортов
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
    """Основная функция запуска алгоритма с тестовыми данными"""
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_deterministic_vrp.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Запуск VRP с микротестовыми данными...")
    start_time = time.time()
    
    try:
        # Пути к тестовым данным
        orders_file = Path("test_data/test_orders.json")
        couriers_file = Path("test_data/test_couriers.json")
        db_file = Path("test_data/test_durations.sqlite")
        
        # Проверяем наличие файлов
        if not orders_file.exists():
            raise FileNotFoundError(f"Тестовый файл заказов не найден: {orders_file}")
        if not couriers_file.exists():
            raise FileNotFoundError(f"Тестовый файл курьеров не найден: {couriers_file}")
        if not db_file.exists():
            raise FileNotFoundError(f"Тестовая база данных не найдена: {db_file}")
        
        logger.info("=== Шаг 1: Загрузка тестовых данных ===")
        
        # Загружаем данные с использованием lazy API
        orders_lf = load_orders_data_lazy(str(orders_file))
        couriers_df, warehouse_info = load_couriers_data(str(couriers_file))
        conn = get_distance_matrix(str(db_file))
        
        logger.info(f"Загружено {len(couriers_df)} курьеров")
        
        # Создаем провайдер расстояний
        distance_provider = SQLiteDistanceProvider(conn)
        
        # Агрегируем заказы по микрополигонам (полностью lazy)
        polygon_stats = aggregate_orders_by_polygon_lazy(orders_lf)
        
        # Фильтруем полигоны по размеру для оптимизации
        filtered_polygons = filter_polygons_by_size(polygon_stats, min_size=1, max_size=20)
        
        # Оптимизируем порядок обработки
        optimized_polygon_order = optimize_polygon_processing_order(filtered_polygons)
        
        logger.info("=== Шаг 2: Оптимизация полигонов ===")
        
        # Оптимизируем полигоны только по расстояниям
        orders_df = load_orders_data(str(orders_file))
        service_times = {}
        
        optimized_polygons = optimize_all_polygons_hybrid(
            optimized_polygon_order, 
            str(db_file), 
            service_times,
            max_workers=None
        )
        
        # Валидация полигонов
        polygon_validator = PolygonValidator(max_time_per_courier=43200)
        polygon_stats_info = polygon_validator.get_polygon_statistics(optimized_polygons)
        
        # Анализируем распределение стоимости
        cost_distribution = polygon_validator.get_polygon_cost_distribution(optimized_polygons)
        
        logger.info(f"📊 Статистика стоимости полигонов:")
        logger.info(f"   Q25: {cost_distribution['q25']:.0f} сек")
        logger.info(f"   Q50: {cost_distribution['q50']:.0f} сек")
        logger.info(f"   Q75: {cost_distribution['q75']:.0f} сек")
        logger.info(f"   Q90: {cost_distribution['q90']:.0f} сек")
        logger.info(f"   Q95: {cost_distribution['q95']:.0f} сек")
        logger.info(f"   Q99: {cost_distribution['q99']:.0f} сек")
        
        # Проверяем неназначаемые полигоны
        unassignable_polygons = polygon_validator.get_unassignable_polygons(optimized_polygons)
        if unassignable_polygons:
            logger.warning(f"Найдено {len(unassignable_polygons)} неназначаемых полигонов")
            optimized_polygons = polygon_validator.filter_assignable_polygons(optimized_polygons)
            logger.info(f"После фильтрации осталось {len(optimized_polygons)} полигонов")
        
        logger.info(f"Оптимизировано {len(optimized_polygons)} полигонов")
        
        # Вычисляем порталы для полигонов
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
        
        # Создаем словарь сервисных времен для каждого курьера
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
        
        # Создаем решатель назначения
        solver = AssignmentSolverFactory.create_solver(
            solver_type='reinforcement',  # Используем RL для микротеста
            distance_provider=distance_provider,
            warehouse_id=warehouse_info['ID'],
            courier_service_times=courier_service_times
        )
        
        # Лимит времени 12 часов (43200 секунд)
        max_time_per_courier = 43200
        
        # Назначаем полигоны курьерам
        assignment = solver.solve(optimized_polygons, couriers_df, max_time_per_courier=max_time_per_courier)
        
        logger.info(f"Назначено {sum(len(polygons) for polygons in assignment.values())} полигонов")
        
        logger.info("=== Шаг 4: Оптимизация маршрутов курьеров ===")
        
        # Оптимизируем маршруты курьеров
        optimized_routes = optimize_courier_routes(
            assignment, 
            optimized_polygons, 
            conn,
            courier_service_times
        )
        
        logger.info(f"Оптимизировано {len(optimized_routes)} маршрутов")
        
        logger.info("=== Шаг 5: Генерация финального решения ===")
        
        # Генерируем финальное решение
        result = generate_solution(
            optimized_routes,
            optimized_polygons,
            orders_df,
            conn,
            "test_solution.json"
        )
        
        # Выводим результаты
        logger.info("=== Результаты ===")
        logger.info(f"Время выполнения: {time.time() - start_time:.2f} секунд")
        
        if result['saved']:
            logger.info("✅ Тестовое решение успешно сгенерировано")
            
            stats = result['statistics']
            logger.info(f"📊 Статистика:")
            logger.info(f"   Курьеров: {stats['total_couriers']}")
            logger.info(f"   Заказов: {stats['assigned_orders']}/{stats['total_orders']} ({stats['assignment_rate']:.2%})")
            logger.info(f"   Среднее время маршрута: {stats.get('avg_route_time', 0):.0f} сек")
            logger.info(f"   Максимальное время маршрута: {stats.get('max_route_time', 0)} сек")
            logger.info(f"   Общее время: {stats.get('total_route_time', 0)} сек")
        else:
            logger.error("❌ Тестовое решение не прошло валидацию")
            
            validation = result['validation']
            if validation['unassigned_orders']:
                logger.error(f"Неназначенные заказы: {len(validation['unassigned_orders'])}")
            if validation['duplicate_orders']:
                logger.error(f"Дубли заказов: {len(validation['duplicate_orders'])}")
            if validation['violations']:
                logger.error(f"Нарушения: {validation['violations']}")
        
        # Закрываем соединение с базой данных
        conn.close()
        
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
        raise
    finally:
        total_time = time.time() - start_time
        logger.info(f"Общее время выполнения: {total_time:.2f} секунд")

if __name__ == "__main__":
    main()
