"""
Эвристический решатель назначения полигонов
"""

import logging
import random
from typing import List, Dict, Tuple
import polars as pl
from interfaces import IPolygonAssignmentSolver, IDistanceProvider

logger = logging.getLogger(__name__)

class HeuristicAssignmentSolver(IPolygonAssignmentSolver):
    """Эвристический решатель назначения полигонов"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1, 
                 courier_service_times: Dict[int, Dict[int, int]] = None):
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        self.courier_service_times = courier_service_times or {}
    
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить задачу назначения эвристическим методом"""
        logger.info("Начинаем эвристическое назначение полигонов")
        
        # Сортируем полигоны по стоимости (сначала дешевые)
        sorted_polygons = polygons_df.sort('total_cost').to_dicts()
        
        # Инициализируем назначения
        assignment = {i: [] for i in range(len(couriers_df))}
        courier_times = {i: 0 for i in range(len(couriers_df))}
        
        assigned_count = 0
        unassigned_polygons = []
        
        for polygon_data in sorted_polygons:
            polygon_id = polygon_data['MpId']
            base_distance_cost = polygon_data['total_distance']  # Только расстояние без сервисного времени
            portal_id = polygon_data['portal_id']
            
            # Вычисляем полную стоимость полигона для каждого курьера
            if portal_id:
                time_to_polygon = self.distance_provider.get_distance(self.warehouse_id, portal_id)
                time_from_polygon = self.distance_provider.get_distance(portal_id, self.warehouse_id)
                base_total_time = time_to_polygon + base_distance_cost + time_from_polygon
            else:
                base_total_time = base_distance_cost
            
            # Ищем курьера с минимальным временем работы
            best_courier = None
            min_time = float('inf')
            
            logger.debug(f"Назначаем полигон {polygon_id} (стоимость: {base_total_time} сек)")
            
            # Проверяем всех курьеров
            for courier_id in range(len(couriers_df)):
                # Получаем сервисное время для этого курьера и полигона
                service_time = 0  # УБИРАЕМ ЗАГЛУШКИ! Получаем из реальных данных курьера
                if courier_id in self.courier_service_times and polygon_id in self.courier_service_times[courier_id]:
                    service_time = self.courier_service_times[courier_id][polygon_id]
                
                logger.debug(f"  Курьер {courier_id}: сервисное время {service_time} сек")
                
                # Рассчитываем общее время для этого курьера
                total_polygon_time = base_distance_cost + service_time
                
                # Добавляем время до/от склада
                if portal_id is not None:
                    # Время от склада до первого заказа
                    warehouse_to_portal = self.distance_provider.get_distance(0, portal_id)
                    # Время от последнего заказа до склада (используем тот же портал)
                    portal_to_warehouse = self.distance_provider.get_distance(portal_id, 0)
                    total_polygon_time += warehouse_to_portal + portal_to_warehouse
                
                # Проверяем, не превышает ли это лимит времени курьера
                if courier_times[courier_id] + total_polygon_time <= max_time_per_courier:
                    if courier_times[courier_id] + total_polygon_time < min_time:
                        min_time = courier_times[courier_id] + total_polygon_time
                        best_courier = courier_id
                        best_total_time = total_polygon_time
                else:
                    logger.debug(f"    Курьер {courier_id}: превышает лимит ({courier_times[courier_id] + total_polygon_time} > {max_time_per_courier})")
            
            if best_courier is not None:
                # Назначаем полигон лучшему курьеру
                assignment[best_courier].append(polygon_id)
                courier_times[best_courier] += best_total_time
                assigned_count += 1
                logger.debug(f"  ✅ Назначен курьеру {best_courier}")
            else:
                logger.warning(f"  ❌ Не найден подходящий курьер для полигона {polygon_id}")
                unassigned_polygons.append(polygon_id)
        
        logger.info(f"Эвристика назначила {assigned_count} полигонов")
        logger.info(f"Максимальное время курьера: {max(courier_times.values())} сек")
        
        # Дополнительная отладочная информация
        if assigned_count == 0:
            logger.warning("НЕ НАЗНАЧЕНО НИ ОДНОГО ПОЛИГОНА!")
            logger.warning(f"Всего полигонов для назначения: {len(polygons_df)}")
            logger.warning(f"Всего курьеров: {len(couriers_df)}")
            logger.warning(f"Лимит времени на курьера: {max_time_per_courier} сек")
            
            # Проверяем первые несколько полигонов
            for i, row in enumerate(polygons_df.iter_rows(named=True)):
                if i >= 3:  # Проверяем только первые 3
                    break
                polygon_id = row['MpId']
                total_distance = row['total_distance']
                portal_id = row.get('portal_id')
                logger.warning(f"Полигон {polygon_id}: расстояние={total_distance} сек, портал={portal_id}")
        
        return assignment
    
    def solve_with_improvement(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
                              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить с последующим улучшением"""
        # Базовое назначение
        assignment = self.solve(polygons_df, couriers_df, max_time_per_courier)
        
        # Улучшение через локальный поиск
        improved_assignment = self._local_search_improvement(
            assignment, polygons_df, couriers_df, max_time_per_courier
        )
        
        return improved_assignment
    
    def _local_search_improvement(self, assignment: Dict[int, List[int]], 
                                polygons_df: pl.DataFrame, couriers_df: pl.DataFrame,
                                max_time_per_courier: int) -> Dict[int, List[int]]:
        """Локальный поиск для улучшения назначения"""
        logger.info("Начинаем локальный поиск для улучшения назначения...")
        
        improved = True
        iterations = 0
        max_iterations = 10
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Вычисляем текущие времена курьеров
            courier_times = self._calculate_courier_times(assignment, polygons_df)
            
            # Пробуем переставить полигоны между курьерами
            for courier1 in assignment:
                for courier2 in assignment:
                    if courier1 >= courier2:
                        continue
                    
                    for polygon1 in assignment[courier1]:
                        for polygon2 in assignment[courier2]:
                            # Пробуем поменять полигоны местами
                            if self._can_swap_polygons(assignment, courier1, courier2, 
                                                      polygon1, polygon2, polygons_df, 
                                                      max_time_per_courier):
                                # Выполняем обмен
                                assignment[courier1].remove(polygon1)
                                assignment[courier1].append(polygon2)
                                assignment[courier2].remove(polygon2)
                                assignment[courier2].append(polygon1)
                                
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        logger.info(f"Локальный поиск завершен за {iterations} итераций")
        
        # Вычисляем финальные времена
        final_times = self._calculate_courier_times(assignment, polygons_df)
        logger.info(f"После локального поиска: {sum(len(polygons) for polygons in assignment.values())} полигонов")
        logger.info(f"Максимальное время курьера: {max(final_times.values())} сек")
        
        return assignment
    
    def _calculate_courier_times(self, assignment: Dict[int, List[int]], 
                               polygons_df: pl.DataFrame) -> Dict[int, int]:
        """Вычислить времена работы курьеров"""
        courier_times = {}
        
        for courier_id, polygon_ids in assignment.items():
            total_time = 0
            for polygon_id in polygon_ids:
                polygon_row = polygons_df.filter(pl.col('MpId') == polygon_id).row(0, named=True)
                polygon_cost = polygon_row['total_cost']
                portal_id = polygon_row['portal_id']
                
                if portal_id:
                    time_to_polygon = self.distance_provider.get_distance(self.warehouse_id, portal_id)
                    time_from_polygon = self.distance_provider.get_distance(portal_id, self.warehouse_id)
                    total_polygon_time = time_to_polygon + polygon_cost + time_from_polygon
                else:
                    total_polygon_time = polygon_cost
                
                total_time += total_polygon_time
            
            courier_times[courier_id] = total_time
        
        return courier_times
    
    def _can_swap_polygons(self, assignment: Dict[int, List[int]], 
                          courier1: int, courier2: int,
                          polygon1: int, polygon2: int,
                          polygons_df: pl.DataFrame,
                          max_time_per_courier: int) -> bool:
        """Проверить, можно ли поменять полигоны местами"""
        # Временное назначение для проверки
        temp_assignment = {k: v.copy() for k, v in assignment.items()}
        
        temp_assignment[courier1].remove(polygon1)
        temp_assignment[courier1].append(polygon2)
        temp_assignment[courier2].remove(polygon2)
        temp_assignment[courier2].append(polygon1)
        
        # Проверяем ограничения
        courier_times = self._calculate_courier_times(temp_assignment, polygons_df)
        
        return all(time <= max_time_per_courier for time in courier_times.values())
