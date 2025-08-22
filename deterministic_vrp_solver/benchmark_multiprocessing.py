#!/usr/bin/env python3
"""
Бенчмарк для сравнения производительности threading vs multiprocessing
"""

import time
import psutil
import logging
from multiprocessing import cpu_count
import polars as pl
from polygon_optimizer import (
    optimize_all_polygons, 
    optimize_all_polygons_mp, 
    optimize_all_polygons_hybrid
)
from utils import load_orders_data_lazy, aggregate_orders_by_polygon_lazy

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_system_info():
    """Получить информацию о системе"""
    return {
        'cpu_count': cpu_count(),
        'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
        'memory_available': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
    }

def benchmark_approach(approach_name: str, func, *args, **kwargs):
    """Бенчмарк для конкретного подхода"""
    logger.info(f"Тестируем подход: {approach_name}")
    
    # Измеряем время и память
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    start_cpu = psutil.cpu_percent(interval=1)
    
    # Выполняем операцию
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    end_cpu = psutil.cpu_percent(interval=1)
    
    execution_time = end_time - start_time
    memory_used = end_memory - start_memory
    cpu_usage = (start_cpu + end_cpu) / 2
    
    logger.info(f"✅ {approach_name}:")
    logger.info(f"   Время: {execution_time:.2f} сек")
    logger.info(f"   Память: {memory_used:.1f} MB")
    logger.info(f"   CPU: {cpu_usage:.1f}%")
    logger.info(f"   Результатов: {len(result)}")
    
    return {
        'approach': approach_name,
        'time': execution_time,
        'memory': memory_used,
        'cpu': cpu_usage,
        'results_count': len(result)
    }

def main():
    """Основная функция бенчмарка"""
    logger.info("🚀 Запуск бенчмарка multiprocessing vs threading")
    
    # Информация о системе
    system_info = get_system_info()
    logger.info(f"💻 Система: {system_info['cpu_count']} CPU, {system_info['memory_total']:.1f} GB RAM")
    
    # Загружаем тестовые данные
    logger.info("📊 Загрузка тестовых данных...")
    orders_lf = load_orders_data_lazy("../ml_ozon_logistic/ml_ozon_logistic_dataSetOrders.json")
    polygon_stats = aggregate_orders_by_polygon_lazy(orders_lf)
    
    # Фильтруем для тестирования (берем первые 50 полигонов)
    test_polygons = polygon_stats.head(50)
    logger.info(f"Тестируем на {len(test_polygons)} полигонах")
    
    # Подготавливаем данные
    db_path = "../durations.sqlite"
    service_times = {1: {order_id: 300 for order_id in range(1, 1000)}}  # Заглушка
    
    results = []
    
    # Тест 1: Threading (старый подход)
    try:
        result1 = benchmark_approach(
            "Threading (ThreadPoolExecutor)",
            optimize_all_polygons,
            test_polygons, None, service_times, max_workers=4
        )
        results.append(result1)
    except Exception as e:
        logger.error(f"Ошибка в threading тесте: {e}")
    
    # Тест 2: Multiprocessing
    try:
        result2 = benchmark_approach(
            "Multiprocessing (Pool)",
            optimize_all_polygons_mp,
            test_polygons, db_path, service_times, max_workers=cpu_count()
        )
        results.append(result2)
    except Exception as e:
        logger.error(f"Ошибка в multiprocessing тесте: {e}")
    
    # Тест 3: Гибридный подход
    try:
        result3 = benchmark_approach(
            "Гибридный (Multiprocessing + Threading)",
            optimize_all_polygons_hybrid,
            test_polygons, db_path, service_times, max_workers=cpu_count()
        )
        results.append(result3)
    except Exception as e:
        logger.error(f"Ошибка в гибридном тесте: {e}")
    
    # Анализ результатов
    if len(results) > 1:
        logger.info("📈 Анализ результатов:")
        
        # Находим лучший результат по времени
        best_time = min(results, key=lambda x: x['time'])
        logger.info(f"🏆 Лучшее время: {best_time['approach']} ({best_time['time']:.2f} сек)")
        
        # Сравнение с threading
        threading_result = next((r for r in results if 'Threading' in r['approach']), None)
        if threading_result:
            for result in results:
                if 'Threading' not in result['approach']:
                    speedup = threading_result['time'] / result['time']
                    logger.info(f"⚡ {result['approach']}: {speedup:.2f}x быстрее threading")
        
        # Анализ использования ресурсов
        logger.info("💾 Использование ресурсов:")
        for result in results:
            efficiency = result['results_count'] / result['time']
            logger.info(f"   {result['approach']}: {efficiency:.1f} полигонов/сек")
    
    logger.info("✅ Бенчмарк завершен")

if __name__ == "__main__":
    main()
