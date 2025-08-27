                      

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
        
                         
        self.durations_conn = None
        self.ports_conn = None
        self.warehouse_ports_conn = None
        
                                                             
        self.durations_cache = {}
        self.ports_cache = {}
        self.warehouse_ports_cache = {}
        self.access_cost_cache = {}                                   
        self.best_port_cache = {}                                       
        
                                
        self.polygon_info = {}                                                                      
                                                              
        self.warehouse_id = 0
        
    def __enter__(self):
        self.durations_conn = sqlite3.connect(self.durations_db_path)
        self.ports_conn = sqlite3.connect(self.ports_db_path)
        self.warehouse_ports_conn = sqlite3.connect(self.warehouse_ports_db_path)
        
                            
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
    
    def set_warehouse_id(self, warehouse_id: int) -> None:
        """Установить ID склада для правильной логики доступа/расстояний"""
        self.warehouse_id = int(warehouse_id)
    
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
            distance = result[0] if (result and result[0] and result[0] > 0) else float('inf')
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
            distance = result[0] if result and result[0] and result[0] > 0 else float('inf')
            self.warehouse_ports_cache[port_id] = distance
        return self.warehouse_ports_cache[port_id]

    def get_polygon_portal(self, polygon_id: int) -> Optional[int]:
        info = self.polygon_info.get(int(polygon_id), {})
        portal_id = info.get('portal_id', None)
        try:
            return int(portal_id) if portal_id is not None else None
        except Exception:
            return None
    
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
        """Возвращает единственную точку входа — портал полигона, и расстояние до него из durations."""
        cache_key = (from_position, target_polygon_id)
        if cache_key in self.best_port_cache:
            return self.best_port_cache[cache_key]
        portal_id = self.get_polygon_portal(int(target_polygon_id))
        if portal_id is None:
            result = (None, float('inf'))
        else:
            distance = self.get_original_distance(int(from_position), int(portal_id))
            result = (int(portal_id), float(distance))
        self.best_port_cache[cache_key] = result
        return result
    
    def get_polygon_access_cost(self, from_position: int, polygon_id: int, 
                               service_time: float = 0) -> float:
        """Стоимость доступа к полигону: от текущей позиции до портала полигона (durations) + TSP + сервис."""
        cache_key = (from_position, polygon_id, service_time)
        if cache_key in self.access_cost_cache:
            return self.access_cost_cache[cache_key]
        portal_id = self.get_polygon_portal(int(polygon_id))
        if portal_id is None:
            result = float('inf')
        else:
            to_portal = self.get_original_distance(int(from_position), int(portal_id))
            if to_portal >= float('inf'):
                result = float('inf')
            else:
                polygon_cost = self.get_polygon_cost(int(polygon_id))
                result = float(to_portal) + float(polygon_cost) + float(service_time)
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
