"""
OR-Tools решатель назначения полигонов
"""

import logging
from typing import List, Dict, Optional
import polars as pl
from interfaces import IPolygonAssignmentSolver, IDistanceProvider

logger = logging.getLogger(__name__)

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    logger.warning("OR-Tools не установлен")

class ORToolsAssignmentSolver(IPolygonAssignmentSolver):
    """Решатель назначения с использованием OR-Tools CP-SAT"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1):
        if not ORTOOLS_AVAILABLE:
            raise ImportError("OR-Tools не установлен")
        
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
    
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить задачу назначения с помощью OR-Tools CP-SAT"""
        logger.info("Начинаем решение задачи назначения с OR-Tools CP-SAT")
        
        model = cp_model.CpModel()
        
        # Параметры задачи
        num_couriers = len(couriers_df)
        num_polygons = len(polygons_df)
        
        logger.info(f"Задача: {num_couriers} курьеров, {num_polygons} полигонов")
        
        # Создаем переменные назначения
        assignments = {}
        for c in range(num_couriers):
            for p in range(num_polygons):
                assignments[(c, p)] = model.NewBoolVar(f'assign_c{c}_p{p}')
        
        # Ограничение 1: каждый полигон назначается ровно одному курьеру
        for p in range(num_polygons):
            model.Add(sum(assignments[(c, p)] for c in range(num_couriers)) == 1)
        
        # Ограничение 2: время работы каждого курьера не превышает лимит
        for c in range(num_couriers):
            courier_time = 0
            for p in range(num_polygons):
                polygon_row = polygons_df.row(p, named=True)
                polygon_cost = polygon_row['total_cost']
                portal_id = polygon_row['portal_id']
                
                # Добавляем время до полигона и обратно на склад
                if portal_id:
                    time_to_polygon = self.distance_provider.get_distance(self.warehouse_id, portal_id)
                    time_from_polygon = self.distance_provider.get_distance(portal_id, self.warehouse_id)
                    total_polygon_time = time_to_polygon + polygon_cost + time_from_polygon
                else:
                    total_polygon_time = polygon_cost
                
                courier_time += assignments[(c, p)] * total_polygon_time
            
            model.Add(courier_time <= max_time_per_courier)
        
        # Целевая функция: минимизация суммарного времени
        objective = 0
        for c in range(num_couriers):
            for p in range(num_polygons):
                polygon_row = polygons_df.row(p, named=True)
                polygon_cost = polygon_row['total_cost']
                portal_id = polygon_row['portal_id']
                
                if portal_id:
                    time_to_polygon = self.distance_provider.get_distance(self.warehouse_id, portal_id)
                    time_from_polygon = self.distance_provider.get_distance(portal_id, self.warehouse_id)
                    total_polygon_time = time_to_polygon + polygon_cost + time_from_polygon
                else:
                    total_polygon_time = polygon_cost
                
                objective += assignments[(c, p)] * total_polygon_time
        
        model.Minimize(objective)
        
        # Решаем задачу
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300.0  # 5 минут максимум
        
        logger.info("Запускаем OR-Tools CP-SAT решатель...")
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            logger.info(f"Решение найдено: {solver.StatusName(status)}")
            
            # Извлекаем результат
            assignment = {}
            for c in range(num_couriers):
                assigned_polygons = []
                for p in range(num_polygons):
                    if solver.Value(assignments[(c, p)]) == 1:
                        polygon_id = polygons_df.row(p, named=True)['MpId']
                        assigned_polygons.append(polygon_id)
                
                if assigned_polygons:
                    assignment[c] = assigned_polygons
            
            logger.info(f"OR-Tools назначил {sum(len(polygons) for polygons in assignment.values())} полигонов")
            return assignment
        else:
            logger.warning(f"OR-Tools не нашел решение: {solver.StatusName(status)}")
            return {}
