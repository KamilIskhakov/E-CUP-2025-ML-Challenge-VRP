#!/usr/bin/env python3
"""
Воркер для многопроцессной оценки маршрутов курьеров
"""

import sqlite3
import json
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Настройка логирования для воркера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RouteEvaluation:
    """Результат оценки маршрута курьера"""
    courier_id: int
    route_time: int  # общее время маршрута в секундах
    travel_time: int  # время в пути
    service_time: int  # время обслуживания
    feasible: bool  # укладывается ли в 12 часов
    portal_pairs: List[Tuple[int, int]]  # пары порталов для переходов
    error_message: Optional[str] = None

class RouteWorker:
    """Воркер для оценки маршрутов курьеров"""
    
    def __init__(self, data_dir: str, durations_db_path: str):
        self.data_dir = Path(data_dir)
        self.durations_db_path = durations_db_path
        self.db_connection = None
        self.mp_data = {}
        self.service_times = {}
        self.warehouse_id = 1  # ID склада
        
        # Инициализация при создании воркера
        self._initialize_worker()
    
    def _initialize_worker(self):
        """Инициализация воркера - открытие соединений и загрузка данных"""
        logger.info("Инициализируем воркер...")
        
        # Открываем соединение с базой расстояний
        self.db_connection = sqlite3.connect(self.durations_db_path, check_same_thread=False)
        
        # Настройка SQLite для быстрого чтения
        cursor = self.db_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        
        # Загружаем метаданные микрополигонов
        mp_metadata_path = self.data_dir / "mp_metadata.json"
        with open(mp_metadata_path, 'r') as f:
            mp_metadata = json.load(f)
        
        for mp_info in mp_metadata:
            self.mp_data[mp_info['mp_id']] = {
                'order_ids': mp_info['order_ids'],
                'size': mp_info['size'],
                'portals': mp_info['portals'],
                'l_intra_best': mp_info['l_intra_best']
            }
        
        # Загружаем сервисные времена
        service_times_path = self.data_dir / "service_times_filtered.json"
        with open(service_times_path, 'r') as f:
            service_times_data = json.load(f)
        
        for record in service_times_data:
            courier_id = record['courier_id']
            mp_id = record['mp_id']
            service_time = record['service_time']
            
            if courier_id not in self.service_times:
                self.service_times[courier_id] = {}
            self.service_times[courier_id][mp_id] = service_time
        
        logger.info(f"Воркер инициализирован: {len(self.mp_data)} микрополигонов, {len(self.service_times)} курьеров")
    
    def evaluate_courier_route(self, courier_id: int, mp_sequence: List[int]) -> RouteEvaluation:
        """
        Оценка маршрута курьера
        
        Args:
            courier_id: ID курьера
            mp_sequence: Последовательность микрополигонов для посещения
            
        Returns:
            RouteEvaluation: Результат оценки
        """
        try:
            if not mp_sequence:
                return RouteEvaluation(
                    courier_id=courier_id,
                    route_time=0,
                    travel_time=0,
                    service_time=0,
                    feasible=True,
                    portal_pairs=[]
                )
            
            # Вычисляем время маршрута
            route_time, travel_time, service_time, portal_pairs = self._calculate_route_time(
                courier_id, mp_sequence
            )
            
            # Проверяем ограничение 12 часов
            feasible = route_time <= 43200  # 12 часов в секундах
            
            return RouteEvaluation(
                courier_id=courier_id,
                route_time=route_time,
                travel_time=travel_time,
                service_time=service_time,
                feasible=feasible,
                portal_pairs=portal_pairs
            )
            
        except Exception as e:
            logger.error(f"Ошибка при оценке маршрута курьера {courier_id}: {e}")
            return RouteEvaluation(
                courier_id=courier_id,
                route_time=0,
                travel_time=0,
                service_time=0,
                feasible=False,
                portal_pairs=[],
                error_message=str(e)
            )
    
    def _calculate_route_time(self, courier_id: int, mp_sequence: List[int]) -> Tuple[int, int, int, List[Tuple[int, int]]]:
        """
        Вычисляет время маршрута курьера
        
        Returns:
            (route_time, travel_time, service_time, portal_pairs)
        """
        travel_time = 0
        service_time = 0
        portal_pairs = []
        
        cursor = self.db_connection.cursor()
        
        # Время от склада до первого микрополигона
        if mp_sequence:
            first_mp = mp_sequence[0]
            first_portal = self.mp_data[first_mp]['portals'][0]  # берем первый портал
            
            # Расстояние от склада до первого портала
            warehouse_to_first = self._get_distance(cursor, self.warehouse_id, first_portal)
            travel_time += warehouse_to_first
            
            # Время внутреннего тура по первому микрополигону
            first_mp_intra = self.mp_data[first_mp]['l_intra_best']
            travel_time += first_mp_intra
            
            # Сервисное время для первого микрополигона
            first_mp_service = self._get_service_time(courier_id, first_mp)
            service_time += first_mp_service
        
        # Переходы между микрополигонами
        for i in range(len(mp_sequence) - 1):
            current_mp = mp_sequence[i]
            next_mp = mp_sequence[i + 1]
            
            # Выбираем оптимальную пару порталов
            best_portal_pair = self._find_best_portal_pair(
                cursor, current_mp, next_mp
            )
            
            if best_portal_pair:
                from_portal, to_portal = best_portal_pair
                portal_pairs.append((from_portal, to_portal))
                
                # Расстояние между порталами
                inter_portal_dist = self._get_distance(cursor, from_portal, to_portal)
                travel_time += inter_portal_dist
                
                # Время внутреннего тура по следующему микрополигону
                next_mp_intra = self.mp_data[next_mp]['l_intra_best']
                travel_time += next_mp_intra
                
                # Сервисное время для следующего микрополигона
                next_mp_service = self._get_service_time(courier_id, next_mp)
                service_time += next_mp_service
        
        # Время от последнего микрополигона до склада
        if mp_sequence:
            last_mp = mp_sequence[-1]
            last_portal = self.mp_data[last_mp]['portals'][0]  # берем первый портал
            
            # Расстояние от последнего портала до склада
            last_to_warehouse = self._get_distance(cursor, last_portal, self.warehouse_id)
            travel_time += last_to_warehouse
        
        route_time = travel_time + service_time
        
        return route_time, travel_time, service_time, portal_pairs
    
    def _find_best_portal_pair(self, cursor, from_mp: int, to_mp: int) -> Optional[Tuple[int, int]]:
        """Находит оптимальную пару порталов между микрополигонами"""
        from_portals = self.mp_data[from_mp]['portals']
        to_portals = self.mp_data[to_mp]['portals']
        
        best_pair = None
        min_distance = float('inf')
        
        for from_portal in from_portals:
            for to_portal in to_portals:
                distance = self._get_distance(cursor, from_portal, to_portal)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (from_portal, to_portal)
        
        return best_pair
    
    def _get_distance(self, cursor, from_id: int, to_id: int) -> int:
        """Получает расстояние между двумя точками"""
        cursor.execute(
            "SELECT d FROM dists WHERE f = ? AND t = ?",
            (from_id, to_id)
        )
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            # Если нет прямого расстояния, возвращаем большое значение
            return 10000
    
    def _get_service_time(self, courier_id: int, mp_id: int) -> int:
        """Получает сервисное время курьера для микрополигона"""
        if courier_id in self.service_times and mp_id in self.service_times[courier_id]:
            return self.service_times[courier_id][mp_id]
        else:
            # Значение по умолчанию
            return 300
    
    def close(self):
        """Закрывает соединения"""
        if self.db_connection:
            self.db_connection.close()

def init_worker(data_dir: str, durations_db_path: str) -> RouteWorker:
    """Инициализация воркера для ProcessPoolExecutor"""
    return RouteWorker(data_dir, durations_db_path)

def evaluate_route_batch(worker: RouteWorker, evaluations: List[Tuple[int, List[int]]]) -> List[RouteEvaluation]:
    """
    Пакетная оценка маршрутов
    
    Args:
        worker: Воркер
        evaluations: Список (courier_id, mp_sequence) для оценки
        
    Returns:
        Список результатов оценки
    """
    results = []
    for courier_id, mp_sequence in evaluations:
        result = worker.evaluate_courier_route(courier_id, mp_sequence)
        results.append(result)
    return results

if __name__ == "__main__":
    # Тест воркера
    worker = RouteWorker("data", "../durations.sqlite")
    
    # Тестовый маршрут
    test_route = [1, 2, 3]  # микрополигоны
    result = worker.evaluate_courier_route(1, test_route)
    
    print(f"Результат оценки: {result}")
    worker.close()
