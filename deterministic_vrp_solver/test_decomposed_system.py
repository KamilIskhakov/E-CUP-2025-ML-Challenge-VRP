#!/usr/bin/env python3

import os
import sys
import time
import logging
import json
import gc
import sqlite3
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import polars as pl
from deterministic_vrp_solver.utils import (
    load_orders_data_lazy, load_couriers_data,
    aggregate_orders_by_polygon_lazy, calculate_polygon_portal,
    filter_polygons_by_size, optimize_polygon_processing_order
)
from deterministic_vrp_solver.polygon.optimizer import optimize_all_polygons_hybrid
from deterministic_vrp_solver.route_optimizer import optimize_courier_routes
from deterministic_vrp_solver.solution_generator import generate_solution
from deterministic_vrp_solver.decomposed_distance_provider import DecomposedDistanceProvider
from deterministic_vrp_solver.reinforcement_scheduler import ReinforcementScheduler

def get_fast_db_connection(db_path: str) -> sqlite3.Connection:
    """Быстрое подключение к SQLite"""
    conn = sqlite3.connect(db_path, timeout=30.0)
    
    # Минимальная оптимизация
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=50000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=134217728")  # 128MB mmap
    conn.execute("PRAGMA page_size=4096")
    
    return conn

def build_ports_database(ports_db_path: str, polygon_ports: dict, durations_conn: sqlite3.Connection) -> None:
    """Создает БД расстояний между портами для выбранных портов полигонов."""
    if os.path.exists(ports_db_path):
        os.remove(ports_db_path)
    conn = sqlite3.connect(ports_db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS port_distances (from_port INTEGER, to_port INTEGER, distance REAL)")
    # Собираем уникальные порты
    all_ports = sorted({p for ports in polygon_ports.values() for p in ports})
    # Заполняем расстояния для всех пар портов
    dcur = durations_conn.cursor()
    for i, fp in enumerate(all_ports):
        for tp in all_ports:
            dcur.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (fp, tp))
            row = dcur.fetchone()
            dist = row[0] if row and row[0] and row[0] > 0 else 0
            cur.execute("INSERT INTO port_distances (from_port, to_port, distance) VALUES (?, ?, ?)", (fp, tp, dist))
    conn.commit()
    conn.close()

def build_warehouse_ports_database(warehouse_ports_db_path: str, polygon_ports: dict, durations_conn: sqlite3.Connection) -> None:
    """Создает БД расстояний от склада (ID=0) до каждого порта."""
    if os.path.exists(warehouse_ports_db_path):
        os.remove(warehouse_ports_db_path)
    conn = sqlite3.connect(warehouse_ports_db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS warehouse_port_distances (port_id INTEGER PRIMARY KEY, distance REAL)")
    all_ports = sorted({p for ports in polygon_ports.values() for p in ports})
    dcur = durations_conn.cursor()
    for port_id in all_ports:
        dcur.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (0, port_id))
        row = dcur.fetchone()
        dist = row[0] if row and row[0] and row[0] > 0 else 0
        cur.execute("INSERT OR REPLACE INTO warehouse_port_distances (port_id, distance) VALUES (?, ?)", (port_id, dist))
    conn.commit()
    conn.close()

def create_decomposed_test():
    """Создает тест с декомпозированной системой"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_decomposed_system.log'),
            logging.StreamHandler()
        ]
    )
    # Устанавливаем уровень DEBUG только для reinforcement_scheduler
    logging.getLogger('reinforcement_scheduler').setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Запуск теста с декомпозированной системой")
    start_time = time.time()
    
    try:
        root = Path(__file__).resolve().parent.parent
        data_dir = root / "ml_ozon_logistic"
        orders_file = data_dir / "ml_ozon_logistic_dataSetOrders.json"
        couriers_file = data_dir / "ml_ozon_logistic_dataSetCouriers.json"
        db_file = root / "durations.sqlite"
        
        if not orders_file.exists():
            raise FileNotFoundError(f"Файл заказов не найден: {orders_file}")
        if not couriers_file.exists():
            raise FileNotFoundError(f"Файл курьеров не найден: {couriers_file}")
        if not db_file.exists():
            raise FileNotFoundError(f"База данных не найдена: {db_file}")
        
        logger.info("=== Шаг 1: Загрузка данных ===")
        
        logger.info("Загрузка заказов...")
        orders_lf = load_orders_data_lazy(str(orders_file))
        
        # Отладочная информация
        logger.info("=== ОТЛАДКА: Проверка данных ===")
        orders_df_sample = orders_lf.collect()
        logger.info(f"Количество заказов: {len(orders_df_sample)}")
        logger.info(f"Количество уникальных полигонов: {orders_df_sample['MpId'].n_unique()}")
        logger.info(f"Примеры полигонов: {orders_df_sample['MpId'].head(10).to_list()}")
        
        logger.info("Загрузка курьеров...")
        couriers_df, warehouse_info = load_couriers_data(str(couriers_file))
        
        # Ограничиваем количество курьеров до 100 для теста
        if len(couriers_df) > 100:
            logger.info(f"Ограничиваем количество курьеров с {len(couriers_df)} до 100")
            couriers_df = couriers_df.head(100)
        
        logger.info("Подключение к базе данных...")
        conn = get_fast_db_connection(str(db_file))
        
        logger.info(f"Загружено {len(couriers_df)} курьеров")
        
        logger.info("Агрегация полигонов...")
        # Исключаем заказы со склада (MpId=0) из агрегации
        orders_lf_filtered = orders_lf.filter(pl.col('MpId') != 0)
        polygon_stats = aggregate_orders_by_polygon_lazy(orders_lf_filtered)
        
        logger.info(f"Найдено {len(polygon_stats)} полигонов в полных данных")
        
        # Выбираем первые 300 полигонов
        polygon_stats_300 = polygon_stats.head(300)
        
        logger.info(f"Выбрано {len(polygon_stats_300)} полигонов для теста")
        
        filtered_polygons = filter_polygons_by_size(polygon_stats_300, min_size=1, max_size=20)
        optimized_polygon_order = optimize_polygon_processing_order(filtered_polygons)
        
        logger.info(f"Отфильтровано {len(filtered_polygons)} полигонов для обработки")
        
        logger.info("=== Шаг 2: Выбор портов полигонов ===")
        # Простая стратегия: берем до 3 точек-заказов как порты для каждого полигона
        polygon_ports = {}
        for row in filtered_polygons.iter_rows(named=True):
            order_ids = row['order_ids']
            ports = order_ids[:3] if isinstance(order_ids, list) else []
            polygon_ports[row['MpId']] = ports
        ports_db_path = str(Path(__file__).resolve().parent / "data" / "ports_database.sqlite")
        build_ports_database(ports_db_path, polygon_ports, conn)

        logger.info("=== Шаг 3: Создание БД расстояний от склада к портам ===")
        warehouse_ports_db_path = str(Path(__file__).resolve().parent / "data" / "warehouse_ports_database.sqlite")
        build_warehouse_ports_database(warehouse_ports_db_path, polygon_ports, conn)
        
        logger.info("=== Шаг 4: Оптимизация полигонов ===")
        
        service_times = {}
        optimized_polygons = optimize_all_polygons_hybrid(
            optimized_polygon_order, 
            str(db_file), 
            service_times,
            max_workers=1
        )
        
        # Статистика и фильтрация без PolygonValidator
        q25 = optimized_polygons['total_cost'].quantile(0.25)
        q50 = optimized_polygons['total_cost'].quantile(0.50)
        q75 = optimized_polygons['total_cost'].quantile(0.75)
        q90 = optimized_polygons['total_cost'].quantile(0.90)
        logger.info(f"Статистика стоимости полигонов:")
        logger.info(f"   Q25: {q25:.0f} сек")
        logger.info(f"   Q50: {q50:.0f} сек")
        logger.info(f"   Q75: {q75:.0f} сек")
        logger.info(f"   Q90: {q90:.0f} сек")
        too_expensive = optimized_polygons.filter(pl.col('total_cost') > 43200)
        if len(too_expensive) > 0:
            logger.warning(f"Найдено {len(too_expensive)} полигонов дороже лимита")
            optimized_polygons = optimized_polygons.filter(pl.col('total_cost') <= 43200)
            logger.info(f"После фильтрации осталось {len(optimized_polygons)} полигонов")
        
        logger.info(f"Оптимизировано {len(optimized_polygons)} полигонов")
        
        logger.info("Вычисление порталов полигонов...")
        # Инициализируем колонку portal_id, т.к. оптимизация полигонов её не добавляет
        if 'portal_id' not in optimized_polygons.columns:
            optimized_polygons = optimized_polygons.with_columns(pl.lit(0).alias('portal_id'))
        
        # Загружаем только заказы для выбранных полигонов, исключая склад (MpId=0)
        selected_mp_ids = set(polygon_stats_300['MpId'].to_list())
        orders_df = load_orders_data_lazy(str(orders_file)).filter(
            pl.col('MpId').is_in(selected_mp_ids) & (pl.col('MpId') != 0)
        ).collect()
        
        logger.info(f"Загружено {len(orders_df)} заказов для {len(selected_mp_ids)} полигонов")
        
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
        
        # Не удаляем orders_df, так как она нужна позже для генерации решения
        # del orders_df
        # gc.collect()
        
        logger.info("=== Шаг 5: Подготовка информации о полигонах для декомпозиции ===")
        
        logger.info(f"Количество оптимизированных полигонов: {len(optimized_polygons)}")
        logger.info(f"Количество полигонов с портами: {len(polygon_ports)}")
        
        # Создаем структуру информации о полигонах для декомпозированного провайдера
        polygon_info_for_decomposition = {}
        
        for row in optimized_polygons.iter_rows(named=True):
            polygon_id = row['MpId']
            
            # Исключаем склад (ID=0) из обработки
            if polygon_id == 0:
                continue
                
            ports = polygon_ports.get(polygon_id, [])
            cost = row['total_cost']
            portal_id = row['portal_id']
            
            # Вычисляем расстояния от портов до центрального пункта (портала)
            portal_distances = {}
            for port_id in ports:
                # Используем durations.sqlite для расстояния от порта до портала
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT d FROM dists WHERE f = ? AND t = ?",
                    (port_id, portal_id)
                )
                result = cursor.fetchone()
                if result and result[0] is not None and result[0] > 0:
                    distance = result[0]
                else:
                    # Если расстояние не найдено, используем 0 (порт совпадает с порталом)
                    distance = 0
                portal_distances[port_id] = distance
            
            polygon_info_for_decomposition[polygon_id] = {
                'ports': ports,
                'cost': cost,
                'portal_distances': portal_distances,
                'portal_id': portal_id
            }
        
        logger.info(f"Создано информации о полигонах: {len(polygon_info_for_decomposition)}")
        if len(polygon_info_for_decomposition) > 0:
            sample_polygon = list(polygon_info_for_decomposition.keys())[0]
            sample_info = polygon_info_for_decomposition[sample_polygon]
            logger.info(f"Пример полигона {sample_polygon}:")
            logger.info(f"  Порты: {sample_info['ports']}")
            logger.info(f"  Стоимость: {sample_info['cost']}")
            logger.info(f"  Portal ID: {sample_info['portal_id']}")
            
            # Проверяем, есть ли полигон 9340
            if 9340 in polygon_info_for_decomposition:
                logger.info(f"Полигон 9340 найден в polygon_info_for_decomposition")
                info_9340 = polygon_info_for_decomposition[9340]
                logger.info(f"  Порты: {info_9340['ports']}")
                logger.info(f"  Стоимость: {info_9340['cost']}")
            else:
                logger.warning(f"Полигон 9340 НЕ найден в polygon_info_for_decomposition")
                logger.info(f"Доступные полигоны: {list(polygon_info_for_decomposition.keys())[:10]}")
        else:
            logger.error("polygon_info_for_decomposition пустой!")
        
        logger.info("=== Шаг 6: Назначение полигонов курьерам с декомпозированной системой ===")
        
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
        
        # Создаем декомпозированный провайдер
        decomposed_provider = DecomposedDistanceProvider(
            durations_db_path=str(db_file),
            ports_db_path=ports_db_path,
            warehouse_ports_db_path=warehouse_ports_db_path
        )
        
        # Инициализируем провайдер
        decomposed_provider.__enter__()
        decomposed_provider.set_polygon_info(polygon_info_for_decomposition)
        
        # Получаем статистику
        stats = decomposed_provider.get_statistics()
        logger.info(f"Статистика декомпозированной системы:")
        logger.info(f"   Полигонов: {stats['total_polygons']}")
        logger.info(f"   Всего портов: {stats['total_ports']}")
        logger.info(f"   Среднее портов на полигон: {stats['avg_ports_per_polygon']:.2f}")
        
        logger.info("Запуск RL алгоритма с декомпозированной системой...")
        optimized_polygons_filtered = optimized_polygons.filter(pl.col('MpId') != 0)
        logger.info(f"Исключен склад из полигонов: {len(optimized_polygons)} -> {len(optimized_polygons_filtered)}")
        scheduler = ReinforcementScheduler(
            optimized_polygons_filtered,
            couriers_df,
            max_time_per_courier=43200,
            distance_provider=decomposed_provider,
            courier_service_times=courier_service_times,
            use_parallel=True,
            num_workers=4,
        )
        assignment = scheduler.solve(optimized_polygons_filtered, couriers_df, max_time_per_courier=43200)
        
        total_assigned = sum(len(polygons) for polygons in assignment.values())
        active_couriers = sum(1 for polygons in assignment.values() if polygons)
        
        logger.info(f"Назначено {total_assigned} полигонов")
        logger.info(f"Активных курьеров: {active_couriers}")
        
        logger.info("=== Шаг 7: Оптимизация маршрутов курьеров ===")
        
        # Пропускаем оптимизацию маршрутов, если нет назначений
        if total_assigned > 0:
            route_optimizer = optimize_courier_routes(
                assignment, optimized_polygons, conn, courier_service_times
            )
            logger.info("Оптимизировано маршрутов")
        else:
            logger.warning("Пропускаем оптимизацию маршрутов - нет назначений")
        
        logger.info("=== Шаг 8: Генерация финального решения ===")
        
        # Используем результат оптимизации маршрутов
        if total_assigned > 0:
            solution = generate_solution(
                route_optimizer, optimized_polygons, orders_df, 
                conn, 'test_decomposed_system_solution.json'
            )
        else:
            logger.warning("Пропускаем генерацию решения - нет назначений")
            solution = None
        
        if solution:
            with open('test_decomposed_system_solution.json', 'w') as f:
                json.dump({"routes": solution.get('routes', [])}, f, indent=2)
            logger.info("Решение сохранено в test_decomposed_system_solution.json")
        
        execution_time = time.time() - start_time
        logger.info("=== Результаты ===")
        logger.info(f"Время выполнения: {execution_time:.2f} секунд")
        logger.info(f"📊 Статистика:")
        logger.info(f"   Курьеров: {len(couriers_df)}")
        logger.info(f"   Полигонов: {len(optimized_polygons)}")
        logger.info(f"   Назначено полигонов: {total_assigned}")
        logger.info(f"   Активных курьеров: {active_couriers}")
        logger.info(f"   Неактивных курьеров: {len(couriers_df) - active_couriers}")
        
        conn.close()
        decomposed_provider.close()
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Закрываем провайдер в случае ошибки
        if 'decomposed_provider' in locals():
            decomposed_provider.close()
        
        raise

if __name__ == "__main__":
    create_decomposed_test()
