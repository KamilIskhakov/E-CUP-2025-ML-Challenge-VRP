#!/usr/bin/env python3

import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class DecomposedDistanceProvider:
    """Провайдер расстояний для декомпозированной системы"""
    
    def __init__(self, durations_db_path: str, ports_db_path: str, warehouse_ports_db_path: str):
        self.durations_db_path = durations_db_path
        self.ports_db_path = ports_db_path
        self.warehouse_ports_db_path = warehouse_ports_db_path
        
        # Соединения с БД
        self.durations_conn = None
        self.ports_conn = None
        self.warehouse_ports_conn = None
        
        # Агрессивное кэширование для максимального ускорения
        self.durations_cache = {}
        self.ports_cache = {}
        self.warehouse_ports_cache = {}
        self.access_cost_cache = {}  # Кэш для get_polygon_access_cost
        self.best_port_cache = {}    # Кэш для find_best_port_to_polygon
        
        # Информация о полигонах
        self.polygon_info = {}  # {polygon_id: {'ports': [], 'cost': float, 'portal_distances': {}}}
        
    def __enter__(self):
        self.durations_conn = sqlite3.connect(self.durations_db_path)
        self.ports_conn = sqlite3.connect(self.ports_db_path)
        self.warehouse_ports_conn = sqlite3.connect(self.warehouse_ports_db_path)
        
        # Оптимизация SQLite
        for conn in [self.durations_conn, self.ports_conn, self.warehouse_ports_conn]:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=1000000")
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.durations_conn:
            self.durations_conn.close()
        if self.ports_conn:
            self.ports_conn.close()
        if self.warehouse_ports_conn:
            self.warehouse_ports_conn.close()
    
    def set_polygon_info(self, polygon_info: Dict):
        """Установить информацию о полигонах"""
        self.polygon_info = polygon_info
    
    def get_distance_between_polygons(self, from_polygon_id: int, to_polygon_id: int) -> float:
        """Получить расстояние между полигонами (через durations.sqlite)"""
        key = (from_polygon_id, to_polygon_id)
        if key not in self.durations_cache:
            cursor = self.durations_conn.cursor()
            cursor.execute(
                "SELECT d FROM dists WHERE f = ? AND t = ?",
                (from_polygon_id, to_polygon_id)
            )
            result = cursor.fetchone()
            distance = result[0] if result and result[0] > 0 else float('inf')
            self.durations_cache[key] = distance
        return self.durations_cache[key]
    
    def get_distance_between_ports(self, from_port: int, to_port: int) -> float:
        """Получить расстояние между портами (через ports_database.sqlite)"""
        key = (from_port, to_port)
        if key not in self.ports_cache:
            cursor = self.ports_conn.cursor()
            cursor.execute(
                "SELECT distance FROM port_distances WHERE from_port = ? AND to_port = ?",
                (from_port, to_port)
            )
            result = cursor.fetchone()
            distance = result[0] if result else float('inf')
            self.ports_cache[key] = distance
        return self.ports_cache[key]
    
    def get_distance_from_warehouse_to_port(self, port_id: int) -> float:
        """Получить расстояние от склада к порту из warehouse_ports_database.sqlite"""
        if port_id not in self.warehouse_ports_cache:
            cursor = self.warehouse_ports_conn.cursor()
            cursor.execute(
                "SELECT distance FROM warehouse_port_distances WHERE port_id = ?",
                (port_id,)
            )
            result = cursor.fetchone()
            distance = result[0] if result and result[0] is not None and result[0] > 0 else float('inf')
            self.warehouse_ports_cache[port_id] = distance
        return self.warehouse_ports_cache[port_id]
    
    def get_polygon_ports(self, polygon_id: int) -> List[int]:
        """Получить порты полигона"""
        return self.polygon_info.get(polygon_id, {}).get('ports', [])
    
    def get_polygon_cost(self, polygon_id: int) -> float:
        """Получить базовую стоимость полигона (TSP)"""
        return self.polygon_info.get(polygon_id, {}).get('cost', float('inf'))
    
    def get_portal_distance(self, polygon_id: int, port_id: int) -> float:
        """Получить расстояние от порта до центрального пункта полигона (портала)"""
        portal_distances = self.polygon_info.get(polygon_id, {}).get('portal_distances', {})
        return portal_distances.get(port_id, float('inf'))
    
    def find_best_port_to_polygon(self, from_position: int, target_polygon_id: int) -> Tuple[int, float]:
        """Найти лучший порт для входа в полигон с кэшированием"""
        
        # Проверяем кэш
        cache_key = (from_position, target_polygon_id)
        if cache_key in self.best_port_cache:
            return self.best_port_cache[cache_key]
        
        target_ports = self.get_polygon_ports(target_polygon_id)
        if not target_ports:
            result = (None, float('inf'))
            self.best_port_cache[cache_key] = result
            return result
        
        best_port = None
        best_distance = float('inf')
        
        for port_id in target_ports:
            if from_position == 0:  # Склад имеет ID=0
                # Расстояние от склада до порта
                distance = self.get_distance_from_warehouse_to_port(port_id)
            else:
                # Сначала пробуем расстояние между портами
                distance = self.get_distance_between_ports(from_position, port_id)
                
                # Если расстояние слишком большое, пробуем прямое расстояние из основной БД
                if distance >= float('inf') or distance > 10000:  # Если больше 2.8 часов
                    direct_distance = self.get_original_distance(from_position, port_id)
                    if direct_distance < float('inf') and direct_distance < distance:
                        distance = direct_distance
            
            if distance < best_distance:
                best_distance = distance
                best_port = port_id
        
        result = (best_port, best_distance)
        self.best_port_cache[cache_key] = result
        return result
    
    def get_polygon_access_cost(self, from_position: int, polygon_id: int, 
                               service_time: float = 0) -> float:
        """Получить стоимость доступа к полигону с агрессивным кэшированием"""
        
        # Проверяем кэш
        cache_key = (from_position, polygon_id, service_time)
        if cache_key in self.access_cost_cache:
            return self.access_cost_cache[cache_key]
        
        # Находим лучший порт для входа
        best_port, port_distance = self.find_best_port_to_polygon(from_position, polygon_id)
        
        if port_distance >= float('inf'):
            result = float('inf')
        else:
            # Расстояние от порта до центрального пункта полигона (портала)
            portal_distance = self.get_portal_distance(polygon_id, best_port)
            
            # Базовая стоимость полигона (TSP)
            polygon_cost = self.get_polygon_cost(polygon_id)
            
            # Общая стоимость = путь к порту + путь к центру + стоимость полигона + сервисное время
            result = port_distance + portal_distance + polygon_cost + service_time
        
        # Сохраняем в кэш
        self.access_cost_cache[cache_key] = result
        return result
    
    def get_all_polygon_ports(self) -> Dict[int, List[int]]:
        """Получить все порты всех полигонов"""
        return {polygon_id: info.get('ports', []) 
                for polygon_id, info in self.polygon_info.items()}
    
    def get_original_distance(self, from_id: int, to_id: int) -> float:
        """Получить оригинальное расстояние из durations.sqlite"""
        key = (from_id, to_id)
        if key not in self.durations_cache:
            cursor = self.durations_conn.cursor()
            cursor.execute(
                "SELECT d FROM dists WHERE f = ? AND t = ?",
                (from_id, to_id)
            )
            result = cursor.fetchone()
            distance = result[0] if result and result[0] > 0 else float('inf')
            self.durations_cache[key] = distance
        return self.durations_cache[key]
    
    def close(self):
        """Закрыть все соединения"""
        if self.durations_conn:
            self.durations_conn.close()
        if self.ports_conn:
            self.ports_conn.close()
        if self.warehouse_ports_conn:
            self.warehouse_ports_conn.close()
    
    def get_statistics(self) -> Dict:
        """Получить статистику"""
        total_polygons = len(self.polygon_info)
        total_ports = sum(len(info.get('ports', [])) for info in self.polygon_info.values())
        
        return {
            'total_polygons': total_polygons,
            'total_ports': total_ports,
            'avg_ports_per_polygon': total_ports / total_polygons if total_polygons > 0 else 0
        }
