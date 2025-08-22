#!/usr/bin/env python3

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

def create_warehouse_ports_database(ports_database_path: str, 
                                   durations_db_path: str,
                                   output_path: str):
    """Создать БД расстояний от склада к портам"""
    
    logger.info(f"Создание БД расстояний от склада к портам: {output_path}")
    
    # Подключаемся к БД портов
    ports_conn = sqlite3.connect(ports_database_path)
    ports_cursor = ports_conn.cursor()
    
    # Подключаемся к основной БД
    durations_conn = sqlite3.connect(durations_db_path)
    durations_cursor = durations_conn.cursor()
    
    # Создаем новую БД
    warehouse_conn = sqlite3.connect(output_path)
    warehouse_cursor = warehouse_conn.cursor()
    
    # Создаем таблицу
    warehouse_cursor.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_port_distances (
            port_id INTEGER PRIMARY KEY,
            distance REAL
        )
    """)
    
    # Получаем все порты
    ports_cursor.execute("SELECT DISTINCT port_id FROM polygon_ports")
    all_ports = [row[0] for row in ports_cursor.fetchall()]
    
    logger.info(f"Найдено {len(all_ports)} портов")
    
    # Вычисляем расстояния от склада (ID=0) к каждому порту
    warehouse_id = 0
    processed = 0
    
    for port_id in all_ports:
        # Ищем расстояние от склада к порту в основной БД
        durations_cursor.execute(
            "SELECT d FROM dists WHERE f = ? AND t = ?",
            (warehouse_id, port_id)
        )
        result = durations_cursor.fetchone()
        
        if result and result[0] > 0:
            distance = result[0]
        else:
            # Если прямого расстояния нет, ищем через промежуточные точки
            distance = find_distance_via_intermediates(durations_cursor, warehouse_id, port_id)
        
        # Сохраняем расстояние
        warehouse_cursor.execute(
            "INSERT OR REPLACE INTO warehouse_port_distances (port_id, distance) VALUES (?, ?)",
            (port_id, distance)
        )
        
        processed += 1
        if processed % 100 == 0:
            logger.info(f"Обработано {processed}/{len(all_ports)} портов")
    
    warehouse_conn.commit()
    
    # Статистика
    warehouse_cursor.execute("SELECT COUNT(*) FROM warehouse_port_distances WHERE distance < 999999")
    valid_distances = warehouse_cursor.fetchone()[0]
    logger.info(f"Создано {valid_distances}/{len(all_ports)} валидных расстояний от склада")
    
    # Закрываем соединения
    ports_conn.close()
    durations_conn.close()
    warehouse_conn.close()
    
    logger.info("БД расстояний от склада к портам создана")

def find_distance_via_intermediates(cursor, from_id: int, to_id: int) -> float:
    """Найти расстояние через промежуточные точки"""
    
    # Получаем все доступные промежуточные точки
    cursor.execute("SELECT DISTINCT t FROM dists WHERE f = ? AND t != ? LIMIT 100", (from_id, to_id))
    intermediates = [row[0] for row in cursor.fetchall()]
    
    min_distance = float('inf')
    
    for intermediate in intermediates:
        # Расстояние от начальной точки до промежуточной
        cursor.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (from_id, intermediate))
        result1 = cursor.fetchone()
        if not result1 or result1[0] <= 0:
            continue
        dist1 = result1[0]
        
        # Расстояние от промежуточной до конечной точки
        cursor.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (intermediate, to_id))
        result2 = cursor.fetchone()
        if not result2 or result2[0] <= 0:
            continue
        dist2 = result2[0]
        
        total_distance = dist1 + dist2
        if total_distance < min_distance:
            min_distance = total_distance
    
    return min_distance if min_distance < float('inf') else 999999
