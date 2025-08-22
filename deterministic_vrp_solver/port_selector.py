#!/usr/bin/env python3

import sqlite3
import polars as pl
import logging
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PortCandidate:
    """Кандидат на порт полигона"""
    order_id: int
    polygon_id: int
    connectivity_score: float  # Количество доступных полигонов
    distance_score: float      # Среднее расстояние до других портов
    combined_score: float      # Комбинированный скор

class PortSelector:
    """Селектор портов для полигонов с комбинированными критериями"""
    
    def __init__(self, db_path: str, connectivity_weight: float = 0.7, distance_weight: float = 0.3):
        self.db_path = db_path
        self.connectivity_weight = connectivity_weight
        self.distance_weight = distance_weight
        self.conn = None
        self.port_distances = {}  # Кэш расстояний между портами
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=1000000")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def get_distance(self, from_id: int, to_id: int) -> float:
        """Получить расстояние между двумя точками"""
        key = (from_id, to_id)
        if key not in self.port_distances:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT d FROM dists WHERE f = ? AND t = ?",
                (from_id, to_id)
            )
            result = cursor.fetchone()
            distance = result[0] if result and result[0] > 0 else float('inf')
            self.port_distances[key] = distance
        return self.port_distances[key]
    
    def calculate_connectivity_score(self, candidate_id: int, all_polygon_orders: Dict[int, List[int]]) -> float:
        """Вычислить скор связности для кандидата"""
        accessible_polygons = 0
        total_polygons = len(all_polygon_orders)
        
        for polygon_id, order_ids in all_polygon_orders.items():
            # Проверяем, можем ли добраться до хотя бы одного заказа в полигоне
            for order_id in order_ids:
                distance = self.get_distance(candidate_id, order_id)
                if distance < float('inf'):
                    accessible_polygons += 1
                    break
        
        return accessible_polygons / total_polygons if total_polygons > 0 else 0.0
    
    def calculate_distance_score(self, candidate_id: int, all_candidates: List[int]) -> float:
        """Вычислить скор расстояния для кандидата"""
        distances = []
        
        for other_id in all_candidates:
            if other_id != candidate_id:
                distance = self.get_distance(candidate_id, other_id)
                if distance < float('inf'):
                    distances.append(distance)
        
        if not distances:
            return float('inf')
        
        # Нормализуем: меньшее расстояние = лучший скор
        avg_distance = np.mean(distances)
        return 1.0 / (1.0 + avg_distance / 1000.0)  # Нормализация
    
    def select_ports_for_polygon(self, polygon_id: int, order_ids: List[int], 
                                all_polygon_orders: Dict[int, List[int]], 
                                max_ports: int = 3) -> List[int]:
        """Выбрать порты для одного полигона"""
        
        # Собираем всех кандидатов
        all_candidates = []
        for orders in all_polygon_orders.values():
            all_candidates.extend(orders)
        
        candidates = []
        
        # Оцениваем каждого кандидата
        for order_id in order_ids:
            connectivity_score = self.calculate_connectivity_score(order_id, all_polygon_orders)
            distance_score = self.calculate_distance_score(order_id, all_candidates)
            
            # Комбинированный скор
            combined_score = (self.connectivity_weight * connectivity_score + 
                            self.distance_weight * distance_score)
            
            candidates.append(PortCandidate(
                order_id=order_id,
                polygon_id=polygon_id,
                connectivity_score=connectivity_score,
                distance_score=distance_score,
                combined_score=combined_score
            ))
        
        # Сортируем по комбинированному скору и выбираем лучших
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        
        selected_ports = [c.order_id for c in candidates[:max_ports]]
        
        logger.info(f"Полигон {polygon_id}: выбрано {len(selected_ports)} портов")
        for i, candidate in enumerate(candidates[:max_ports]):
            logger.debug(f"  Порт {i+1}: {candidate.order_id} "
                        f"(связность: {candidate.connectivity_score:.3f}, "
                        f"расстояние: {candidate.distance_score:.3f}, "
                        f"общий: {candidate.combined_score:.3f})")
        
        return selected_ports
    
    def select_all_ports(self, polygon_orders: Dict[int, List[int]], 
                        max_ports_per_polygon: int = 3) -> Dict[int, List[int]]:
        """Выбрать порты для всех полигонов"""
        
        logger.info(f"Выбор портов для {len(polygon_orders)} полигонов")
        
        polygon_ports = {}
        
        for polygon_id, order_ids in polygon_orders.items():
            logger.debug(f"Обработка полигона {polygon_id} ({len(order_ids)} заказов)")
            
            ports = self.select_ports_for_polygon(
                polygon_id, order_ids, polygon_orders, max_ports_per_polygon
            )
            polygon_ports[polygon_id] = ports
        
        # Статистика
        total_ports = sum(len(ports) for ports in polygon_ports.values())
        logger.info(f"Выбрано {total_ports} портов для {len(polygon_ports)} полигонов")
        
        return polygon_ports
    
    def create_ports_distance_matrix(self, polygon_ports: Dict[int, List[int]]) -> Dict[Tuple[int, int], float]:
        """Создать матрицу расстояний между всеми портами"""
        
        logger.info("Создание матрицы расстояний между портами")
        
        # Собираем все порты
        all_ports = []
        for ports in polygon_ports.values():
            all_ports.extend(ports)
        
        logger.info(f"Всего портов: {len(all_ports)}")
        
        # Вычисляем расстояния между всеми портами
        distance_matrix = {}
        total_pairs = len(all_ports) * len(all_ports)
        processed_pairs = 0
        
        for i, port1 in enumerate(all_ports):
            for j, port2 in enumerate(all_ports):
                if i != j:
                    distance = self.get_distance(port1, port2)
                    distance_matrix[(port1, port2)] = distance
                
                processed_pairs += 1
                if processed_pairs % 10000 == 0:
                    logger.debug(f"Обработано {processed_pairs}/{total_pairs} пар портов")
        
        # Статистика
        valid_distances = sum(1 for d in distance_matrix.values() if d < float('inf'))
        logger.info(f"Матрица создана: {valid_distances}/{len(distance_matrix)} валидных расстояний")
        
        return distance_matrix
    
    def save_ports_database(self, polygon_ports: Dict[int, List[int]], 
                           distance_matrix: Dict[Tuple[int, int], float], 
                           output_path: str):
        """Сохранить базу данных портов"""
        
        logger.info(f"Сохранение базы данных портов в {output_path}")
        
        conn = sqlite3.connect(output_path)
        
        # Создаем таблицы
        conn.execute("""
            CREATE TABLE IF NOT EXISTS polygon_ports (
                polygon_id INTEGER,
                port_id INTEGER,
                port_order INTEGER,
                PRIMARY KEY (polygon_id, port_id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS port_distances (
                from_port INTEGER,
                to_port INTEGER,
                distance REAL,
                PRIMARY KEY (from_port, to_port)
            )
        """)
        
        # Сохраняем порты полигонов
        for polygon_id, ports in polygon_ports.items():
            for i, port_id in enumerate(ports):
                conn.execute(
                    "INSERT OR REPLACE INTO polygon_ports (polygon_id, port_id, port_order) VALUES (?, ?, ?)",
                    (polygon_id, port_id, i)
                )
        
        # Сохраняем расстояния
        for (from_port, to_port), distance in distance_matrix.items():
            if distance < float('inf'):
                conn.execute(
                    "INSERT OR REPLACE INTO port_distances (from_port, to_port, distance) VALUES (?, ?, ?)",
                    (from_port, to_port, distance)
                )
        
        conn.commit()
        conn.close()
        
        logger.info("База данных портов сохранена")

def select_polygon_ports(polygon_orders: Dict[int, List[int]], 
                        db_path: str, 
                        output_path: str,
                        max_ports_per_polygon: int = 3) -> Dict[int, List[int]]:
    """Основная функция выбора портов"""
    
    with PortSelector(db_path) as selector:
        # Выбираем порты для всех полигонов
        polygon_ports = selector.select_all_ports(polygon_orders, max_ports_per_polygon)
        
        # Создаем матрицу расстояний
        distance_matrix = selector.create_ports_distance_matrix(polygon_ports)
        
        # Сохраняем в БД
        selector.save_ports_database(polygon_ports, distance_matrix, output_path)
        
        return polygon_ports
