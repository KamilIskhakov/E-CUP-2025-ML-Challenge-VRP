"""
Умный динамический планировщик с учетом близости полигонов и приоритетов
"""

import logging
import heapq
from typing import List, Dict, Tuple, Optional
import polars as pl
from dataclasses import dataclass
from interfaces import IDistanceProvider
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SmartCourier:
    """Курьер с расширенной информацией"""
    id: int
    current_location: Optional[int] = None  # ID текущего полигона или склада
    free_time: int = 0
    total_work_time: int = 0
    assigned_polygons: List[int] = None
    efficiency_score: float = 1.0  # Эффективность курьера
    
    def __post_init__(self):
        if self.assigned_polygons is None:
            self.assigned_polygons = []

@dataclass
class SmartPolygon:
    """Полигон с расширенной информацией"""
    id: int
    cost: int
    portal_id: Optional[int] = None
    order_ids: List[int] = None
    assigned: bool = False
    assigned_time: Optional[int] = None
    priority: float = 1.0  # Приоритет полигона
    urgency: float = 0.0  # Срочность (увеличивается со временем)
    
    def __post_init__(self):
        if self.order_ids is None:
            self.order_ids = []

class SmartDynamicScheduler:
    """Умный динамический планировщик"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1):
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        self.current_time = 0
        self.events = []
        self.available_couriers = []
        self.pending_polygons = []
        self.polygon_locations = {}  # Кэш местоположений полигонов
        
    def schedule_polygons(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
                         max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Умное динамическое планирование"""
        logger.info("Начинаем умное динамическое планирование")
        
        # Инициализация
        couriers = self._initialize_couriers(couriers_df)
        polygons = self._initialize_polygons(polygons_df)
        
        # Основной цикл планирования
        while self.pending_polygons and self.current_time < max_time_per_courier:
            # Обновляем срочность полигонов
            self._update_polygon_urgency(polygons)
            
            # Обрабатываем события освобождения
            self._process_free_events(couriers, max_time_per_courier)
            
            # Умное назначение полигонов
            self._smart_assign_polygons(couriers, polygons, max_time_per_courier)
            
            # Переходим к следующему событию
            self._advance_time()
        
        return self._format_result(couriers)
    
    def _initialize_couriers(self, couriers_df: pl.DataFrame) -> Dict[int, SmartCourier]:
        """Инициализация курьеров"""
        couriers = {}
        for i, row in enumerate(couriers_df.iter_rows(named=True)):
            courier = SmartCourier(
                id=i,
                current_location=self.warehouse_id,
                efficiency_score=self._calculate_courier_efficiency(row)
            )
            couriers[i] = courier
            heapq.heappush(self.available_couriers, (0, i))
        return couriers
    
    def _initialize_polygons(self, polygons_df: pl.DataFrame) -> Dict[int, SmartPolygon]:
        """Инициализация полигонов"""
        polygons = {}
        for row in polygons_df.iter_rows(named=True):
            polygon = SmartPolygon(
                id=row['MpId'],
                cost=row['total_cost'],
                portal_id=row['portal_id'],
                order_ids=row['order_ids'],
                priority=self._calculate_polygon_priority(row)
            )
            polygons[polygon.id] = polygon
            self.pending_polygons.append(polygon.id)
            
            # Кэшируем местоположение
            if polygon.portal_id:
                self.polygon_locations[polygon.id] = polygon.portal_id
        
        # Сортируем по приоритету и стоимости
        self.pending_polygons.sort(key=lambda pid: (
            -polygons[pid].priority,  # Высокий приоритет сначала
            polygons[pid].cost  # Дешевые полигоны сначала
        ))
        return polygons
    
    def _calculate_courier_efficiency(self, courier_data) -> float:
        """Вычисление эффективности курьера"""
        # Можно учитывать исторические данные, рейтинг и т.д.
        return 1.0
    
    def _calculate_polygon_priority(self, polygon_data) -> float:
        """Вычисление приоритета полигона"""
        # Приоритет на основе количества заказов и стоимости
        order_count = len(polygon_data['order_ids'])
        cost = polygon_data['total_cost']
        
        # Больше заказов = выше приоритет
        # Меньше стоимость = выше приоритет
        priority = order_count / max(cost, 1)
        return priority
    
    def _update_polygon_urgency(self, polygons: Dict[int, SmartPolygon]):
        """Обновление срочности полигонов"""
        for polygon in polygons.values():
            if not polygon.assigned:
                # Срочность увеличивается со временем
                polygon.urgency = min(1.0, self.current_time / 3600)  # Максимум через час
    
    def _smart_assign_polygons(self, couriers: Dict[int, SmartCourier], 
                             polygons: Dict[int, SmartPolygon], max_time_per_courier: int):
        """Умное назначение полигонов с учетом близости и приоритетов"""
        while self.pending_polygons and self.available_couriers:
            # Находим лучшую пару курьер-полигон
            best_assignment = self._find_best_assignment(couriers, polygons, max_time_per_courier)
            
            if best_assignment:
                courier_id, polygon_id, score = best_assignment
                self._execute_assignment(courier_id, polygon_id, couriers, polygons, max_time_per_courier)
            else:
                # Если не можем назначить, пропускаем проблемный полигон
                problematic_polygon = self.pending_polygons.pop(0)
                logger.warning(f"Полигон {problematic_polygon} не может быть назначен")
    
    def _find_best_assignment(self, couriers: Dict[int, SmartCourier], 
                            polygons: Dict[int, SmartPolygon], 
                            max_time_per_courier: int) -> Optional[Tuple[int, int, float]]:
        """Поиск лучшего назначения курьер-полигон"""
        best_score = -1
        best_assignment = None
        
        for polygon_id in self.pending_polygons[:10]:  # Проверяем только первые 10 полигонов
            polygon = polygons[polygon_id]
            
            for free_time, courier_id in self.available_couriers[:5]:  # Проверяем только первых 5 курьеров
                courier = couriers[courier_id]
                
                # Проверяем возможность назначения
                if self._can_assign_polygon(courier, polygon, max_time_per_courier):
                    score = self._calculate_assignment_score(courier, polygon, free_time)
                    
                    if score > best_score:
                        best_score = score
                        best_assignment = (courier_id, polygon_id, score)
        
        return best_assignment
    
    def _can_assign_polygon(self, courier: SmartCourier, polygon: SmartPolygon, 
                          max_time_per_courier: int) -> bool:
        """Проверка возможности назначения полигона курьеру"""
        # Вычисляем время выполнения
        travel_time = self._calculate_travel_time(courier.current_location, polygon.portal_id)
        return_time = self._calculate_travel_time(polygon.portal_id, self.warehouse_id)
        total_time = travel_time + polygon.cost + return_time
        
        return courier.total_work_time + total_time <= max_time_per_courier
    
    def _calculate_assignment_score(self, courier: SmartCourier, polygon: SmartPolygon, 
                                  free_time: int) -> float:
        """Вычисление оценки назначения"""
        # Базовые факторы
        proximity_score = self._calculate_proximity_score(courier, polygon)
        efficiency_score = courier.efficiency_score
        priority_score = polygon.priority
        urgency_score = polygon.urgency
        
        # Время ожидания (чем меньше, тем лучше)
        wait_time = max(0, free_time - self.current_time)
        wait_score = 1.0 / (1.0 + wait_time / 3600)  # Нормализуем по часам
        
        # Общая оценка
        total_score = (
            proximity_score * 0.3 +
            efficiency_score * 0.2 +
            priority_score * 0.2 +
            urgency_score * 0.15 +
            wait_score * 0.15
        )
        
        return total_score
    
    def _calculate_proximity_score(self, courier: SmartCourier, polygon: SmartPolygon) -> float:
        """Вычисление оценки близости"""
        if not polygon.portal_id:
            return 0.5  # Нейтральная оценка для полигонов без портала
        
        # Расстояние от текущего местоположения курьера до полигона
        distance = self._calculate_travel_time(courier.current_location, polygon.portal_id)
        
        # Нормализуем расстояние (чем меньше, тем лучше)
        max_reasonable_distance = 3600  # 1 час
        proximity_score = max(0, 1.0 - distance / max_reasonable_distance)
        
        return proximity_score
    
    def _calculate_travel_time(self, from_location: Optional[int], to_location: Optional[int]) -> int:
        """Вычисление времени в пути"""
        if not from_location or not to_location:
            return 0
        
        return self.distance_provider.get_distance(from_location, to_location)
    
    def _execute_assignment(self, courier_id: int, polygon_id: int, 
                          couriers: Dict[int, SmartCourier], 
                          polygons: Dict[int, SmartPolygon], 
                          max_time_per_courier: int):
        """Выполнение назначения"""
        courier = couriers[courier_id]
        polygon = polygons[polygon_id]
        
        # Вычисляем время выполнения
        travel_time = self._calculate_travel_time(courier.current_location, polygon.portal_id)
        return_time = self._calculate_travel_time(polygon.portal_id, self.warehouse_id)
        total_time = travel_time + polygon.cost + return_time
        
        # Время назначения и завершения
        assignment_time = max(self.current_time, courier.free_time)
        completion_time = assignment_time + total_time
        
        # Обновляем курьера
        courier.current_location = polygon.portal_id
        courier.total_work_time += total_time
        courier.assigned_polygons.append(polygon_id)
        
        # Обновляем полигон
        polygon.assigned = True
        polygon.assigned_time = assignment_time
        
        # Создаем событие освобождения
        from dynamic_scheduler import AssignmentEvent
        free_event = AssignmentEvent(
            time=completion_time,
            event_type='free',
            courier_id=courier_id,
            polygon_id=polygon_id,
            duration=total_time
        )
        heapq.heappush(self.events, (completion_time, free_event))
        
        # Убираем полигон из ожидающих
        self.pending_polygons.remove(polygon_id)
        
        # Убираем курьера из доступных
        self.available_couriers = [(t, cid) for t, cid in self.available_couriers if cid != courier_id]
        
        logger.debug(f"Назначен полигон {polygon_id} курьеру {courier_id} "
                   f"(время: {assignment_time}-{completion_time}, оценка: {self._calculate_assignment_score(courier, polygon, courier.free_time):.3f})")
    
    def _process_free_events(self, couriers: Dict[int, SmartCourier], max_time_per_courier: int):
        """Обработка событий освобождения"""
        while self.events and self.events[0][0] <= self.current_time:
            event_time, event = heapq.heappop(self.events)
            if event.event_type == 'free':
                courier = couriers[event.courier_id]
                courier.free_time = event_time
                courier.current_location = self.warehouse_id  # Возвращается на склад
                
                if courier.total_work_time < max_time_per_courier:
                    heapq.heappush(self.available_couriers, (event_time, event.courier_id))
    
    def _advance_time(self):
        """Переход к следующему событию"""
        if self.events:
            next_event_time, _ = heapq.heappop(self.events)
            self.current_time = next_event_time
        elif self.available_couriers:
            next_free_time, _ = heapq.heappop(self.available_couriers)
            self.current_time = next_free_time
    
    def _format_result(self, couriers: Dict[int, SmartCourier]) -> Dict[int, List[int]]:
        """Форматирование результата"""
        result = {}
        for courier_id, courier in couriers.items():
            if courier.assigned_polygons:
                result[courier_id] = courier.assigned_polygons
        
        logger.info(f"Умное планирование завершено. Назначено {sum(len(polygons) for polygons in result.values())} полигонов")
        return result
