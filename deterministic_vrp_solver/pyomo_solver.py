"""
Pyomo решатель назначения полигонов
"""

import logging
from typing import List, Dict, Optional
import polars as pl
from interfaces import IPolygonAssignmentSolver, IDistanceProvider

logger = logging.getLogger(__name__)

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    logger.warning("Pyomo не установлен")

class PyomoAssignmentSolver(IPolygonAssignmentSolver):
    """Решатель назначения с использованием Pyomo"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1, 
                 solver_name: str = 'cbc'):
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo не установлен")
        
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        self.solver_name = solver_name
    
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить задачу назначения с помощью Pyomo"""
        logger.info(f"Начинаем решение задачи назначения с Pyomo ({self.solver_name})")
        
        # Создаем модель Pyomo
        model = pyo.ConcreteModel()
        
        # Параметры задачи
        num_couriers = len(couriers_df)
        num_polygons = len(polygons_df)
        
        logger.info(f"Задача: {num_couriers} курьеров, {num_polygons} полигонов")
        
        # Множества
        model.couriers = pyo.RangeSet(0, num_couriers - 1)
        model.polygons = pyo.RangeSet(0, num_polygons - 1)
        
        # Переменные назначения
        model.assign = pyo.Var(model.couriers, model.polygons, domain=pyo.Binary)
        
        # Ограничение 1: каждый полигон назначается ровно одному курьеру
        def polygon_assignment_rule(model, p):
            return sum(model.assign[c, p] for c in model.couriers) == 1
        
        model.polygon_assignment = pyo.Constraint(model.polygons, rule=polygon_assignment_rule)
        
        # Ограничение 2: время работы каждого курьера не превышает лимит
        def courier_time_rule(model, c):
            total_time = 0
            for p in model.polygons:
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
                
                total_time += model.assign[c, p] * total_polygon_time
            
            return total_time <= max_time_per_courier
        
        model.courier_time = pyo.Constraint(model.couriers, rule=courier_time_rule)
        
        # Целевая функция: минимизация суммарного времени
        def objective_rule(model):
            total_cost = 0
            for c in model.couriers:
                for p in model.polygons:
                    polygon_row = polygons_df.row(p, named=True)
                    polygon_cost = polygon_row['total_cost']
                    portal_id = polygon_row['portal_id']
                    
                    if portal_id:
                        time_to_polygon = self.distance_provider.get_distance(self.warehouse_id, portal_id)
                        time_from_polygon = self.distance_provider.get_distance(portal_id, self.warehouse_id)
                        total_polygon_time = time_to_polygon + polygon_cost + time_from_polygon
                    else:
                        total_polygon_time = polygon_cost
                    
                    total_cost += model.assign[c, p] * total_polygon_time
            
            return total_cost
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Решаем задачу
        solver = SolverFactory(self.solver_name)
        
        # Настройки решателя
        if self.solver_name == 'cbc':
            solver.options['seconds'] = 300  # 5 минут максимум
        elif self.solver_name == 'glpk':
            solver.options['tmlim'] = 300
        
        logger.info(f"Запускаем Pyomo решатель ({self.solver_name})...")
        results = solver.solve(model, tee=True)
        
        # Проверяем результат
        if results.solver.termination_condition == pyo.TerminationCondition.optimal or \
           results.solver.termination_condition == pyo.TerminationCondition.feasible:
            
            logger.info(f"Решение найдено: {results.solver.termination_condition}")
            
            # Извлекаем результат
            assignment = {}
            for c in model.couriers:
                assigned_polygons = []
                for p in model.polygons:
                    if pyo.value(model.assign[c, p]) > 0.5:  # Бинарная переменная
                        polygon_id = polygons_df.row(p, named=True)['MpId']
                        assigned_polygons.append(polygon_id)
                
                if assigned_polygons:
                    assignment[c] = assigned_polygons
            
            logger.info(f"Pyomo назначил {sum(len(polygons) for polygons in assignment.values())} полигонов")
            return assignment
        else:
            logger.warning(f"Pyomo не нашел решение: {results.solver.termination_condition}")
            return {}
    
    def solve_with_relaxation(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
                             max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить с релаксацией для больших задач"""
        logger.info("Пробуем решение с релаксацией...")
        
        # Создаем модель с релаксацией
        model = pyo.ConcreteModel()
        
        num_couriers = len(couriers_df)
        num_polygons = len(polygons_df)
        
        model.couriers = pyo.RangeSet(0, num_couriers - 1)
        model.polygons = pyo.RangeSet(0, num_polygons - 1)
        
        # Переменные с релаксацией (0 <= x <= 1)
        model.assign = pyo.Var(model.couriers, model.polygons, domain=pyo.NonNegativeReals, bounds=(0, 1))
        
        # Ограничения
        def polygon_assignment_rule(model, p):
            return sum(model.assign[c, p] for c in model.couriers) == 1
        
        model.polygon_assignment = pyo.Constraint(model.polygons, rule=polygon_assignment_rule)
        
        def courier_time_rule(model, c):
            total_time = 0
            for p in model.polygons:
                polygon_row = polygons_df.row(p, named=True)
                polygon_cost = polygon_row['total_cost']
                portal_id = polygon_row['portal_id']
                
                if portal_id:
                    time_to_polygon = self.distance_provider.get_distance(self.warehouse_id, portal_id)
                    time_from_polygon = self.distance_provider.get_distance(portal_id, self.warehouse_id)
                    total_polygon_time = time_to_polygon + polygon_cost + time_from_polygon
                else:
                    total_polygon_time = polygon_cost
                
                total_time += model.assign[c, p] * total_polygon_time
            
            return total_time <= max_time_per_courier
        
        model.courier_time = pyo.Constraint(model.couriers, rule=courier_time_rule)
        
        # Целевая функция
        def objective_rule(model):
            total_cost = 0
            for c in model.couriers:
                for p in model.polygons:
                    polygon_row = polygons_df.row(p, named=True)
                    polygon_cost = polygon_row['total_cost']
                    portal_id = polygon_row['portal_id']
                    
                    if portal_id:
                        time_to_polygon = self.distance_provider.get_distance(self.warehouse_id, portal_id)
                        time_from_polygon = self.distance_provider.get_distance(portal_id, self.warehouse_id)
                        total_polygon_time = time_to_polygon + polygon_cost + time_from_polygon
                    else:
                        total_polygon_time = polygon_cost
                    
                    total_cost += model.assign[c, p] * total_polygon_time
            
            return total_cost
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Решаем
        solver = SolverFactory(self.solver_name)
        results = solver.solve(model, tee=True)
        
        if results.solver.termination_condition in [pyo.TerminationCondition.optimal, 
                                                   pyo.TerminationCondition.feasible]:
            
            # Округляем результат
            assignment = {}
            for c in model.couriers:
                assigned_polygons = []
                for p in model.polygons:
                    value = pyo.value(model.assign[c, p])
                    if value > 0.5:  # Округляем до ближайшего целого
                        polygon_id = polygons_df.row(p, named=True)['MpId']
                        assigned_polygons.append(polygon_id)
                
                if assigned_polygons:
                    assignment[c] = assigned_polygons
            
            logger.info(f"Pyomo (релаксация) назначил {sum(len(polygons) for polygons in assignment.values())} полигонов")
            return assignment
        else:
            logger.warning("Pyomo с релаксацией не нашел решение")
            return {}
