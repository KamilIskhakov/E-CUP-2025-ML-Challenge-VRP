#!/usr/bin/env python3
"""
Тест логирования генетического алгоритма
"""

import logging
import sqlite3
from polygon_optimizer import PolygonTSPSolver

# Настройка подробного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('genetic_algorithm.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def test_genetic_algorithm():
    """Тест генетического алгоритма с подробным логированием"""
    
    # Создаем тестовое соединение с базой данных
    conn = sqlite3.connect("../durations.sqlite")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=1000000")
    
    # Создаем solver
    solver = PolygonTSPSolver(conn, max_exact_size=15, use_parallel=False)
    
    # Тестовые данные - полигон с 8 точками (достаточно для генетического алгоритма)
    test_order_ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
    
    logger.info("🚀 Тест генетического алгоритма")
    logger.info(f"📊 Тестовый полигон: {len(test_order_ids)} точек")
    logger.info(f"📍 Точки: {test_order_ids}")
    
    # Решаем TSP с использованием генетического алгоритма
    try:
        optimal_route, total_cost = solver.solve_tsp_dynamic(test_order_ids)
        
        logger.info("✅ Генетический алгоритм завершен успешно")
        logger.info(f"🎯 Оптимальный маршрут: {optimal_route}")
        logger.info(f"💰 Общая стоимость: {total_cost}")
        
        # Проверяем корректность маршрута
        if len(optimal_route) == len(test_order_ids):
            logger.info("✅ Все точки включены в маршрут")
        else:
            logger.warning(f"⚠️ Не все точки включены: {len(optimal_route)}/{len(test_order_ids)}")
        
        # Проверяем уникальность точек
        if len(set(optimal_route)) == len(optimal_route):
            logger.info("✅ Все точки уникальны")
        else:
            logger.warning("⚠️ Есть дублирующиеся точки")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в генетическом алгоритме: {e}")
    
    conn.close()
    logger.info("🏁 Тест завершен")

def test_multiple_polygons():
    """Тест нескольких полигонов разного размера"""
    
    conn = sqlite3.connect("../durations.sqlite")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=1000000")
    
    solver = PolygonTSPSolver(conn, max_exact_size=15, use_parallel=False)
    
    # Тестовые полигоны разного размера
    test_polygons = [
        [1001, 1002, 1003],  # 3 точки - будет использован NN
        [1001, 1002, 1003, 1004, 1005],  # 5 точек - генетический алгоритм
        [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],  # 8 точек - генетический алгоритм
    ]
    
    for i, polygon in enumerate(test_polygons):
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Тест полигона {i+1}: {len(polygon)} точек")
        logger.info(f"📍 Точки: {polygon}")
        
        try:
            optimal_route, total_cost = solver.solve_tsp_dynamic(polygon)
            logger.info(f"✅ Результат: стоимость={total_cost}, маршрут={optimal_route}")
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
    
    conn.close()

if __name__ == "__main__":
    logger.info("🧬 Демонстрация логирования генетического алгоритма")
    
    # Тест одного полигона
    test_genetic_algorithm()
    
    # Тест нескольких полигонов
    test_multiple_polygons()
    
    logger.info("\n📝 Логи сохранены в файл 'genetic_algorithm.log'")
