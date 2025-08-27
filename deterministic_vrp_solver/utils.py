import json
import logging
import sqlite3
from typing import Dict, List, Tuple, Optional
import polars as pl
import numpy as np
from pathlib import Path

                       
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deterministic_vrp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_orders_data(file_path: str) -> pl.DataFrame:
    """Загрузка данных о заказах с использованием lazy API"""
    logger.info(f"Загрузка данных о заказах из {file_path}")
    
                                                              
    orders_df = (pl.read_json(file_path)
        .explode('Orders')
        .unnest('Orders')
    )
    
    logger.info(f"Загружено {len(orders_df)} заказов")
    
    return orders_df

def load_couriers_data(file_path: str) -> Tuple[pl.DataFrame, Dict]:
    """Загрузка данных о курьерах и складе"""
    logger.info(f"Загрузка данных о курьерах из {file_path}")
    
                                                      
    data_df = pl.read_json(file_path)
    
                        
    couriers_df = data_df.select('Couriers').explode('Couriers').unnest('Couriers')
    
                                   
    warehouse_info = data_df.select('Warehouse').item(0, 'Warehouse')
    
    logger.info(f"Загружено {len(couriers_df)} курьеров")
    logger.info(f"Склад: ID={warehouse_info['ID']}, координаты=({warehouse_info['Lat']}, {warehouse_info['Long']})")
    
    return couriers_df, warehouse_info

def get_distance_matrix(db_path: str) -> sqlite3.Connection:
    """Подключение к базе данных с матрицей расстояний"""
    logger.info(f"Подключение к базе данных расстояний: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=1000000")                   
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=1073741824")            
    conn.execute("PRAGMA page_size=65536")                               
    
                                    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_from_to'")
    if not cursor.fetchone():
        logger.info("Создание индексов для ускорения запросов...")
        conn.execute("CREATE INDEX idx_from_to ON dists(f, t)")
        conn.execute("CREATE INDEX idx_to_from ON dists(t, f)")
        logger.info("Индексы созданы")
    else:
        logger.info("Индексы уже существуют")
    
    return conn

def get_distance(conn: sqlite3.Connection, from_id: int, to_id: int) -> int:
    """Получение расстояния между двумя точками"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT d FROM dists WHERE f = ? AND t = ?",
        (from_id, to_id)
    )
    result = cursor.fetchone()
    distance = result[0] if result else 0
                                                                      
    return distance if distance > 0 else 999999

def get_distances_batch(conn: sqlite3.Connection, from_ids: List[int], to_ids: List[int]) -> List[int]:
    """Получение расстояний для батча точек"""
    cursor = conn.cursor()
    
                                                   
    cursor.execute("""
        CREATE TEMP TABLE IF NOT EXISTS temp_from (id INTEGER PRIMARY KEY)
    """)
    cursor.execute("""
        CREATE TEMP TABLE IF NOT EXISTS temp_to (id INTEGER PRIMARY KEY)
    """)
    
                      
    cursor.executemany("INSERT OR REPLACE INTO temp_from (id) VALUES (?)", [(id,) for id in from_ids])
    cursor.executemany("INSERT OR REPLACE INTO temp_to (id) VALUES (?)", [(id,) for id in to_ids])
    
                      
    cursor.execute("""
        SELECT d.d 
        FROM dists d
        INNER JOIN temp_from f ON d.f = f.id
        INNER JOIN temp_to t ON d.t = t.id
        ORDER BY f.id, t.id
    """)
    
    results = [row[0] for row in cursor.fetchall()]
    
                               
    cursor.execute("DELETE FROM temp_from")
    cursor.execute("DELETE FROM temp_to")
    
    return results

def aggregate_orders_by_polygon(orders_df: pl.DataFrame) -> pl.DataFrame:
    """Агрегация заказов по микрополигонам с использованием lazy API"""
    logger.info("Агрегация заказов по микрополигонам (lazy API)")
    
                                                         
    polygon_stats = (orders_df
        .lazy()
        .group_by("MpId")
        .agg([
            pl.count().alias("order_count"),
            pl.col("ID").alias("order_ids"),
            pl.col("Lat").alias("lats"),
            pl.col("Long").alias("longs")
        ])
        .collect()
    )
    
    logger.info(f"Найдено {len(polygon_stats)} уникальных микрополигонов")
    
    return polygon_stats

def aggregate_orders_by_polygon_lazy(orders_lf: pl.LazyFrame) -> pl.DataFrame:
    """Агрегация заказов по микрополигонам (полностью lazy)"""
    logger.info("Агрегация заказов по микрополигонам (полностью lazy)")
    
                              
    polygon_stats = (orders_lf
        .group_by("MpId")
        .agg([
            pl.count().alias("order_count"),
            pl.col("ID").alias("order_ids"),
            pl.col("Lat").alias("lats"),
            pl.col("Long").alias("longs")
        ])
        .collect()
    )
    
    logger.info(f"Найдено {len(polygon_stats)} уникальных микрополигонов")
    
    return polygon_stats

def calculate_polygon_portal(conn: sqlite3.Connection, order_ids: List[int], lats: List[float], longs: List[float]) -> int:
    """Вычисление портала полигона (репрезентативной точки)"""
    if len(order_ids) == 1:
        return order_ids[0]
    
                                                                                                     
    reachable_candidates = []
    unreachable_candidates = []
    for pid in order_ids:
        d0 = get_distance(conn, 0, pid)
        if d0 > 0 and d0 < 10**12 and d0 != 999999:
            reachable_candidates.append(pid)
        else:
            unreachable_candidates.append(pid)

    def best_by_internal_avg(candidates: List[int]) -> int:
        min_avg_distance = float('inf')
        best_pid = candidates[0]
        for i, point_id in enumerate(candidates):
            total_distance = 0.0
            count = 0
            for other_id in order_ids:
                if other_id == point_id:
                    continue
                distance = get_distance(conn, point_id, other_id)
                total_distance += distance
                count += 1
            avg_distance = total_distance / count if count > 0 else float('inf')
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                best_pid = point_id
        return best_pid

    if reachable_candidates:
                                                                                                              
        portal = best_by_internal_avg(reachable_candidates)
                             
        best = portal
        best_d0 = get_distance(conn, 0, portal)
        for pid in reachable_candidates:
            if pid == best:
                continue
            d0 = get_distance(conn, 0, pid)
            if d0 < best_d0:
                best = pid
                best_d0 = d0
        return best
    else:
                                                                                        
        return best_by_internal_avg(order_ids)

def validate_solution(routes: List[Dict], orders_df: pl.DataFrame, max_time: int = 43200) -> bool:
    """Валидация решения"""
    logger.info("Валидация решения")
    
                                         
    assigned_orders = set()
    for route in routes:
        assigned_orders.update(route['route'][1:-1])                       
    
    all_orders = set(orders_df['ID'].to_list())
    unassigned = all_orders - assigned_orders
    
    if unassigned:
        logger.warning(f"Неназначенные заказы: {len(unassigned)}")
        return False
    
                     
    if len(assigned_orders) != len(all_orders):
        logger.error("Обнаружены дубли заказов")
        return False
    
    logger.info("Решение валидно")
    return True

def validate_solution_lazy(routes: List[Dict], orders_lf: pl.LazyFrame, max_time: int = 43200) -> bool:
    """Валидация решения с использованием lazy API"""
    logger.info("Валидация решения (lazy)")
    
                                            
    all_orders = (orders_lf
        .select('ID')
        .collect()
        .get_column('ID')
        .to_list()
    )
    
                                         
    assigned_orders = set()
    for route in routes:
        assigned_orders.update(route['route'][1:-1])                       
    
    unassigned = set(all_orders) - assigned_orders
    
    if unassigned:
        logger.warning(f"Неназначенные заказы: {len(unassigned)}")
        return False
    
                     
    if len(assigned_orders) != len(all_orders):
        logger.error("Обнаружены дубли заказов")
        return False
    
    logger.info("Решение валидно")
    return True

def save_solution(routes: List[Dict], output_path: str = "solution.json"):
    """Сохранение решения в файл"""
    logger.info(f"Сохранение решения в {output_path}")
    
    solution = {"routes": routes}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(solution, f, indent=2)
    
    logger.info(f"Решение сохранено: {len(routes)} маршрутов")

def calculate_route_time(route: List[int], conn: sqlite3.Connection, service_times: Dict[int, Dict[int, int]]) -> int:
    """Вычисление времени маршрута"""
    if len(route) < 3:                
        return 0
    
    total_time = 0
    
                                     
    for i in range(len(route) - 1):
        from_id = route[i]
        to_id = route[i + 1]
        distance = get_distance(conn, from_id, to_id)
        total_time += distance
    
                                             
    for order_id in route[1:-1]:                                    
                                                       
                                                               
        service_time = 0                                                         
        total_time += service_time
    
    return total_time

def load_orders_data_lazy(file_path: str) -> pl.LazyFrame:
    """Загрузка данных о заказах в lazy режиме"""
    logger.info(f"Загрузка данных о заказах (lazy) из {file_path}")
    
                                                     
    orders_df = pl.read_json(file_path)
    orders_lf = (orders_df
        .explode('Orders')
        .unnest('Orders')
        .lazy()
    )
    
    logger.info("Данные загружены в lazy режиме")
    return orders_lf

def filter_polygons_by_size(polygon_stats: pl.DataFrame, min_size: int = 1, max_size: int = 15) -> pl.DataFrame:
    """Фильтрация полигонов по размеру с использованием lazy API"""
    logger.info(f"Фильтрация полигонов по размеру: {min_size}-{max_size} заказов")
    
    filtered_polygons = (polygon_stats
        .lazy()
        .filter((pl.col('order_count') >= min_size) & (pl.col('order_count') <= max_size))
        .collect()
    )
    
    logger.info(f"Отфильтровано {len(filtered_polygons)} полигонов из {len(polygon_stats)}")
    return filtered_polygons

def calculate_polygon_statistics(polygon_stats: pl.DataFrame) -> Dict:
    """Вычисление статистики по полигонам с использованием lazy API"""
    logger.info("Вычисление статистики по полигонам")
    
    stats = (polygon_stats
        .lazy()
        .agg([
            pl.count().alias('total_polygons'),
            pl.col('order_count').mean().alias('avg_orders_per_polygon'),
            pl.col('order_count').std().alias('std_orders_per_polygon'),
            pl.col('order_count').min().alias('min_orders'),
            pl.col('order_count').max().alias('max_orders'),
            pl.col('order_count').sum().alias('total_orders')
        ])
        .collect()
        .row(0, named=True)
    )
    
    logger.info(f"Статистика полигонов: {stats}")
    return stats

def optimize_polygon_processing_order(polygon_stats: pl.DataFrame) -> pl.DataFrame:
    """Оптимизация порядка обработки полигонов (сначала маленькие)"""
    logger.info("Оптимизация порядка обработки полигонов")
    
    optimized_order = (polygon_stats
        .lazy()
        .sort('order_count')                                           
        .collect()
    )
    
    logger.info("Порядок обработки оптимизирован")
    return optimized_order
