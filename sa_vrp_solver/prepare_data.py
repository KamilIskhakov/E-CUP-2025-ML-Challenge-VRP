#!/usr/bin/env python3
"""
Предобработка данных для SA VRP решателя
Работает с существующим durations.sqlite
"""

import json
import sqlite3
import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import polars as pl
import numpy as np
from dataclasses import dataclass

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MicroPolygon:
    """Микрополигон с порталами и внутренними расстояниями"""
    mp_id: int
    order_ids: List[int]
    size: int
    diameter: float
    portals: List[int]  # 1-4 портала
    intra_costs: Dict[Tuple[int, int], int]  # (start_order, end_order) -> cost
    l_intra_best: int  # минимальное время внутреннего тура

class DataPreprocessor:
    """Предобработчик данных для SA VRP"""
    
    def __init__(self, orders_path: str, couriers_path: str, durations_db_path: str):
        self.orders_path = orders_path
        self.couriers_path = couriers_path
        self.durations_db_path = durations_db_path
        self.output_dir = Path("sa_vrp_solver/data")
        self.output_dir.mkdir(exist_ok=True)
        
    def prepare_data(self) -> Dict:
        """Основной метод предобработки"""
        logger.info("Начинаем предобработку данных...")
        start_time = time.time()
        
        # 1. Загрузка базовых данных
        orders_df, couriers_df, service_times_df = self._load_basic_data()
        
        # 2. Анализ микрополигонов
        mp_data = self._analyze_micro_polygons(orders_df)
        
        # 3. Вычисление порталов и внутренних расстояний
        mp_data = self._compute_portals_and_intra_costs(orders_df, mp_data)
        
        # 4. Сохранение метаданных
        self._save_metadata(orders_df, couriers_df, service_times_df, mp_data)
        
        elapsed = time.time() - start_time
        logger.info(f"Предобработка завершена за {elapsed:.1f} секунд")
        
        return {
            'mp_data': mp_data,
            'orders_df': orders_df,
            'couriers_df': couriers_df,
            'service_times_df': service_times_df
        }
    
    def _load_basic_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Загрузка базовых данных"""
        logger.info("Загружаем базовые данные...")
        
        # Заказы - правильный парсинг JSON
        with open(self.orders_path, 'r') as f:
            orders_data = json.load(f)
        orders_df = pl.DataFrame(orders_data['Orders'])
        logger.info(f"Загружено {len(orders_df)} заказов")
        
        # Курьеры
        with open(self.couriers_path, 'r') as f:
            couriers_data = json.load(f)
        
        couriers_df = pl.DataFrame(couriers_data['Couriers'])
        logger.info(f"Загружено {len(couriers_df)} курьеров")
        
        # Сервисные времена
        service_times_data = []
        for courier in couriers_data['Couriers']:
            courier_id = courier['ID']
            for service_time in courier['ServiceTimeInMps']:
                service_times_data.append({
                    'courier_id': courier_id,
                    'mp_id': service_time['MpID'],
                    'service_time': service_time['ServiceTime']
                })
        
        service_times_df = pl.DataFrame(service_times_data)
        logger.info(f"Создано {len(service_times_df)} записей сервисных времен")
        
        return orders_df, couriers_df, service_times_df
    
    def _analyze_micro_polygons(self, orders_df: pl.DataFrame) -> Dict[int, MicroPolygon]:
        """Анализ микрополигонов"""
        logger.info("Анализируем микрополигоны...")
        
        # Группировка по MpId
        mp_groups = orders_df.group_by('MpId').agg([
            pl.col('ID').alias('order_ids'),
            pl.col('Lat').alias('lats'),
            pl.col('Long').alias('longs')
        ])
        
        mp_data = {}
        for row in mp_groups.iter_rows(named=True):
            mp_id = row['MpId']
            order_ids = row['order_ids']
            lats = row['lats']
            longs = row['longs']
            
            # Вычисляем диаметр (максимальное расстояние между точками)
            if len(order_ids) > 1:
                coords = list(zip(lats, longs))
                max_dist = 0
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        dist = self._haversine_distance(coords[i], coords[j])
                        max_dist = max(max_dist, dist)
                diameter = max_dist
            else:
                diameter = 0
            
            mp_data[mp_id] = MicroPolygon(
                mp_id=mp_id,
                order_ids=order_ids,
                size=len(order_ids),
                diameter=diameter,
                portals=[],  # будет заполнено позже
                intra_costs={},
                l_intra_best=0
            )
        
        logger.info(f"Проанализировано {len(mp_data)} микрополигонов")
        return mp_data
    
    def _compute_portals_and_intra_costs(self, orders_df: pl.DataFrame, mp_data: Dict[int, MicroPolygon]) -> Dict[int, MicroPolygon]:
        """Вычисление порталов и внутренних расстояний"""
        logger.info("Вычисляем порталы и внутренние расстояния...")
        
        # Подключаемся к существующей базе расстояний
        conn = sqlite3.connect(self.durations_db_path)
        cursor = conn.cursor()
        
        for mp_id, mp in mp_data.items():
            if mp.size <= 1:
                # Для одиночных заказов - сам заказ является порталом
                mp.portals = mp.order_ids
                mp.l_intra_best = 0
                continue
            
            # Определяем количество порталов на основе диаметра
            num_portals = self._get_portal_count(mp.diameter)
            
            # Вычисляем внутренние расстояния
            intra_costs = {}
            for i, order1 in enumerate(mp.order_ids):
                for j, order2 in enumerate(mp.order_ids):
                    if i != j:
                        cursor.execute(
                            "SELECT d FROM dists WHERE f = ? AND t = ?",
                            (order1, order2)
                        )
                        result = cursor.fetchone()
                        if result:
                            intra_costs[(order1, order2)] = result[0]
                        else:
                            # Если нет прямого расстояния, используем большое значение
                            intra_costs[(order1, order2)] = 10000
            
            mp.intra_costs = intra_costs
            
            # Выбираем порталы (заказы с минимальной суммой расстояний до других)
            portal_scores = {}
            for order_id in mp.order_ids:
                total_dist = sum(intra_costs.get((order_id, other), 10000) 
                               for other in mp.order_ids if other != order_id)
                portal_scores[order_id] = total_dist
            
            # Выбираем лучшие порталы
            sorted_portals = sorted(portal_scores.items(), key=lambda x: x[1])
            mp.portals = [order_id for order_id, _ in sorted_portals[:num_portals]]
            
            # Вычисляем минимальное время внутреннего тура (упрощенно)
            if mp.size > 1:
                mp.l_intra_best = min(portal_scores.values()) // 2  # Упрощенная оценка
            else:
                mp.l_intra_best = 0
        
        conn.close()
        
        logger.info("Порталы и внутренние расстояния вычислены")
        return mp_data
    
    def _get_portal_count(self, diameter: float) -> int:
        """Определяет количество порталов на основе диаметра"""
        if diameter <= 300:
            return 1
        elif diameter <= 1055:
            return 2
        elif diameter <= 2300:
            return 3
        else:
            return 4
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Вычисляет расстояние между координатами (упрощенно)"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Упрощенная формула для небольших расстояний
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        return np.sqrt(dlat**2 + dlon**2) * 111000  # примерное расстояние в метрах
    
    def _save_metadata(self, orders_df: pl.DataFrame, couriers_df: pl.DataFrame, 
                      service_times_df: pl.DataFrame, mp_data: Dict[int, MicroPolygon]):
        """Сохранение метаданных"""
        logger.info("Сохраняем метаданные...")
        
        # Сохраняем микрополигоны
        mp_metadata = []
        for mp in mp_data.values():
            mp_metadata.append({
                'mp_id': mp.mp_id,
                'order_ids': mp.order_ids,
                'size': mp.size,
                'diameter': mp.diameter,
                'portals': mp.portals,
                'l_intra_best': mp.l_intra_best
            })
        
        with open(self.output_dir / "mp_metadata.json", 'w') as f:
            json.dump(mp_metadata, f, indent=2)
        
        # Сохраняем заказы
        orders_df.write_json(self.output_dir / "orders_filtered.json")
        
        # Сохраняем курьеров
        couriers_df.write_json(self.output_dir / "couriers_filtered.json")
        
        # Сохраняем сервисные времена
        service_times_df.write_json(self.output_dir / "service_times_filtered.json")
        
        logger.info("Метаданные сохранены")

if __name__ == "__main__":
    # Пример использования
    preprocessor = DataPreprocessor(
        orders_path="../ml_ozon_logistic/ml_ozon_logistic_dataSetOrders.json",
        couriers_path="../ml_ozon_logistic/ml_ozon_logistic_dataSetCouriers.json", 
        durations_db_path="../durations.sqlite"
    )
    
    data = preprocessor.prepare_data()
    print(f"Подготовлено {len(data['mp_data'])} микрополигонов")
