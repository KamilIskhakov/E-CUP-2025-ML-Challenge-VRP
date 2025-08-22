"""
Динамический планировщик назначения полигонов
Учитывает время освобождения курьеров и динамически перераспределяет задачи
"""

import logging
import heapq
from typing import List, Dict, Tuple, Optional
import polars as pl
from dataclasses import dataclass
from interfaces import IDistanceProvider

logger = logging.getLogger(__name__)

@dataclass
class Courier:
    """Курьер с информацией о времени освобождения"""
    id: int
    current_polygon: Optional[int] = None
    free_time: int = 0  # Время освобождения (секунды от начала)
    total_work_time: int = 0  # Общее время работы
    assigned_polygons: List[int] = None
    
    def __post_init__(self):
        if self.assigned_polygons is None:
            self.assigned_polygons = []

@dataclass
class Polygon:
    """Полигон с информацией о времени выполнения"""
    id: int
    cost: int  # Время выполнения полигона
    portal_id: Optional[int] = None
    order_ids: List[int] = None
    assigned: bool = False
    assigned_time: Optional[int] = None
    
    def __post_init__(self):
        if self.order_ids is None:
            self.order_ids = []

@dataclass
class AssignmentEvent:
    """Событие назначения/освобождения"""
    time: int
    event_type: str  # 'assign' или 'free'
    courier_id: int
    polygon_id: Optional[int] = None
    duration: Optional[int] = None

class DynamicScheduler:
    """Динамический планировщик назначения полигонов"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1):
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        self.current_time = 0
        self.events = []  # Очередь событий
        self.available_couriers = []  # Курьеры, доступные для назначения
        self.pending_polygons = []  # Полигоны, ожидающие назначения
        
    def schedule_polygons(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
                         max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """
        Динамическое планирование полигонов с учетом времени освобождения
        
        Args:
            polygons_df: DataFrame с полигонами
            couriers_df: DataFrame с курьерами
            max_time_per_courier: Максимальное время работы курьера
        
        Returns:
            Словарь {courier_id: [polygon_ids]}
        """
        logger.info("Начинаем динамическое планирование полигонов")
        
        # Инициализируем курьеров
        couriers = {}
        for i, row in enumerate(couriers_df.iter_rows(named=True)):
            couriers[i] = Courier(id=i, free_time=0)
            heapq.heappush(self.available_couriers, (0, i))  # (время освобождения, id курьера)
        
        # Инициализируем полигоны
        polygons = {}
        for row in polygons_df.iter_rows(named=True):
            polygon = Polygon(
                id=row['MpId'],
                cost=row['total_cost'],
                portal_id=row['portal_id'],
                order_ids=row['order_ids']
            )
            polygons[polygon.id] = polygon
            self.pending_polygons.append(polygon.id)
        
        # Сортируем полигоны по приоритету (сначала дешевые)
        self.pending_polygons.sort(key=lambda pid: polygons[pid].cost)
        
        logger.info(f"Инициализировано {len(couriers)} курьеров и {len(polygons)} полигонов")
        
        # Основной цикл планирования
        while self.pending_polygons and self.current_time < max_time_per_courier:
            # Обрабатываем события освобождения курьеров
            self._process_free_events(couriers, max_time_per_courier)
            
            # Назначаем полигоны доступным курьерам
            self._assign_pending_polygons(couriers, polygons, max_time_per_courier)
            
            # Переходим к следующему событию
            if self.events:
                next_event = heapq.heappop(self.events)
                self.current_time = next_event.time
                self._handle_event(next_event, couriers, polygons)
            else:
                # Если нет событий, но есть ожидающие полигоны, 
                # ждем освобождения курьера
                if self.available_couriers:
                    next_free_time, courier_id = heapq.heappop(self.available_couriers)
                    self.current_time = next_free_time
                    couriers[courier_id].free_time = next_free_time
                    heapq.heappush(self.available_couriers, (next_free_time, courier_id))
                else:
                    break
        
        # Формируем результат
        result = {}
        for courier_id, courier in couriers.items():
            if courier.assigned_polygons:
                result[courier_id] = courier.assigned_polygons
        
        logger.info(f"Динамическое планирование завершено. Назначено {sum(len(polygons) for polygons in result.values())} полигонов")
        return result
    
    def _process_free_events(self, couriers: Dict[int, Courier], max_time_per_courier: int):
        """Обрабатываем события освобождения курьеров"""
        while self.events and self.events[0].time <= self.current_time:
            event = heapq.heappop(self.events)
            if event.event_type == 'free':
                courier = couriers[event.courier_id]
                courier.free_time = event.time
                courier.current_polygon = None
                
                # Проверяем, не превысил ли курьер лимит времени
                if courier.total_work_time < max_time_per_courier:
                    heapq.heappush(self.available_couriers, (event.time, event.courier_id))
                else:
                    logger.debug(f"Курьер {event.courier_id} достиг лимита времени")
    
    def _assign_pending_polygons(self, couriers: Dict[int, Courier], 
                               polygons: Dict[int, Polygon], max_time_per_courier: int):
        """Назначаем ожидающие полигоны доступным курьерам"""
        while self.pending_polygons and self.available_couriers:
            # Берем следующий полигон
            polygon_id = self.pending_polygons[0]
            polygon = polygons[polygon_id]
            
            # Берем курьера с минимальным временем освобождения
            free_time, courier_id = heapq.heappop(self.available_couriers)
            courier = couriers[courier_id]
            
            # Вычисляем время до полигона
            travel_time = 0
            if polygon.portal_id:
                if courier.current_polygon is None:
                    # От склада до полигона
                    travel_time = self.distance_provider.get_distance(self.warehouse_id, polygon.portal_id)
                else:
                    # От текущего полигона до нового
                    current_polygon = polygons[courier.current_polygon]
                    if current_polygon.portal_id:
                        travel_time = self.distance_provider.get_distance(current_polygon.portal_id, polygon.portal_id)
            
            # Вычисляем время возврата на склад
            return_time = 0
            if polygon.portal_id:
                return_time = self.distance_provider.get_distance(polygon.portal_id, self.warehouse_id)
            
            # Общее время выполнения
            total_time = travel_time + polygon.cost + return_time
            
            # Проверяем ограничения
            if courier.total_work_time + total_time <= max_time_per_courier:
                # Назначаем полигон
                assignment_time = max(self.current_time, courier.free_time)
                completion_time = assignment_time + total_time
                
                # Обновляем курьера
                courier.current_polygon = polygon_id
                courier.total_work_time += total_time
                courier.assigned_polygons.append(polygon_id)
                
                # Обновляем полигон
                polygon.assigned = True
                polygon.assigned_time = assignment_time
                
                # Создаем событие освобождения
                free_event = AssignmentEvent(
                    time=completion_time,
                    event_type='free',
                    courier_id=courier_id,
                    polygon_id=polygon_id,
                    duration=total_time
                )
                heapq.heappush(self.events, (completion_time, free_event))
                
                # Убираем полигон из ожидающих
                self.pending_polygons.pop(0)
                
                logger.debug(f"Назначен полигон {polygon_id} курьеру {courier_id} "
                           f"(время: {assignment_time}-{completion_time})")
            else:
                # Курьер не может взять этот полигон, возвращаем его в очередь
                heapq.heappush(self.available_couriers, (free_time, courier_id))
                
                # Если ни один курьер не может взять полигон, пропускаем его
                if not self.available_couriers:
                    logger.warning(f"Полигон {polygon_id} не может быть назначен ни одному курьеру")
                    self.pending_polygons.pop(0)
                break
    
    def _handle_event(self, event: AssignmentEvent, couriers: Dict[int, Courier], 
                     polygons: Dict[int, Polygon]):
        """Обрабатываем событие"""
        if event.event_type == 'free':
            courier = couriers[event.courier_id]
            courier.free_time = event.time
            courier.current_polygon = None
            logger.debug(f"Курьер {event.courier_id} освободился в {event.time}")
    
    def get_schedule_statistics(self, couriers: Dict[int, Courier]) -> Dict:
        """Получить статистику расписания"""
        stats = {
            'total_couriers': len(couriers),
            'active_couriers': sum(1 for c in couriers.values() if c.assigned_polygons),
            'total_polygons_assigned': sum(len(c.assigned_polygons) for c in couriers.values()),
            'avg_work_time': sum(c.total_work_time for c in couriers.values()) / len(couriers),
            'max_work_time': max(c.total_work_time for c in couriers.values()),
            'min_work_time': min(c.total_work_time for c in couriers.values()),
            'total_time': self.current_time
        }
        return stats
