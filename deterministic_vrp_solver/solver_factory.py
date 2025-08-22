"""
Фабрика для создания решателей назначения полигонов
"""

import logging
from typing import Dict, List
import polars as pl
from interfaces import IPolygonAssignmentSolver, IDistanceProvider

logger = logging.getLogger(__name__)

class AssignmentSolverFactory:
    """Фабрика для создания решателей назначения"""
    
    @staticmethod
    def create_solver(solver_type: str, distance_provider: IDistanceProvider, 
                     warehouse_id: int = 1, courier_service_times: Dict[int, Dict[int, int]] = None, **kwargs) -> IPolygonAssignmentSolver:
        """
        Создать решатель указанного типа
        
        Args:
            solver_type: Тип решателя ('ortools', 'pyomo', 'heuristic', 'hybrid', 'dynamic', 'smart_dynamic')
            distance_provider: Провайдер расстояний
            warehouse_id: ID склада
            **kwargs: Дополнительные параметры (например, solver_name для Pyomo)
        
        Returns:
            Решатель назначения
        """
        if solver_type == 'ortools':
            try:
                from ortools_solver import ORToolsAssignmentSolver
                return ORToolsAssignmentSolver(distance_provider, warehouse_id)
            except ImportError:
                logger.warning("OR-Tools недоступен, используем эвристику")
                from heuristic_solver import HeuristicAssignmentSolver
                return HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
        
        elif solver_type == 'pyomo':
            try:
                from pyomo_solver import PyomoAssignmentSolver
                solver_name = kwargs.get('solver_name', 'cbc')
                return PyomoAssignmentSolver(distance_provider, warehouse_id, solver_name)
            except ImportError:
                logger.warning("Pyomo недоступен, используем эвристику")
                from heuristic_solver import HeuristicAssignmentSolver
                return HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
        
        elif solver_type == 'heuristic':
            from heuristic_solver import HeuristicAssignmentSolver
            return HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
        
        elif solver_type == 'hybrid':
            try:
                from hybrid_solver import HybridAssignmentSolver
                return HybridAssignmentSolver(distance_provider, warehouse_id)
            except ImportError:
                logger.warning("HybridSolver недоступен, используем эвристику")
                from heuristic_solver import HeuristicAssignmentSolver
                return HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
        
        elif solver_type == 'pyomo_hybrid':
            try:
                from pyomo_hybrid_solver import PyomoHybridAssignmentSolver
                return PyomoHybridAssignmentSolver(distance_provider, warehouse_id, **kwargs)
            except ImportError:
                logger.warning("PyomoHybridSolver недоступен, используем эвристику")
                from heuristic_solver import HeuristicAssignmentSolver
                return HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
        
        elif solver_type == 'dynamic':
            try:
                from dynamic_scheduler import DynamicScheduler
                return DynamicAssignmentSolver(distance_provider, warehouse_id)
            except ImportError:
                logger.warning("DynamicScheduler недоступен, используем эвристику")
                from heuristic_solver import HeuristicAssignmentSolver
                return HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
        
        elif solver_type == 'smart_dynamic':
            try:
                from smart_dynamic_scheduler import SmartDynamicScheduler
                return SmartDynamicAssignmentSolver(distance_provider, warehouse_id)
            except ImportError:
                logger.warning("SmartDynamicScheduler недоступен, используем эвристику")
                from heuristic_solver import HeuristicAssignmentSolver
                return HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
        
        elif solver_type == 'reinforcement':
            try:
                from reinforcement_scheduler import ReinforcementScheduler
                return ReinforcementAssignmentSolver(
                    distance_provider, warehouse_id, courier_service_times
                )
            except ImportError as e:
                logger.error(f"ReinforcementScheduler недоступен: {e}")
                raise
        
        else:
            raise ValueError(f"Неизвестный тип решателя: {solver_type}")

class DynamicAssignmentSolver(IPolygonAssignmentSolver):
    """Обертка для динамического планировщика"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1):
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        from dynamic_scheduler import DynamicScheduler
        self.scheduler = DynamicScheduler(distance_provider, warehouse_id)
    
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить с помощью динамического планировщика"""
        logger.info("Используем динамический планировщик")
        return self.scheduler.schedule_polygons(polygons_df, couriers_df, max_time_per_courier)

class SmartDynamicAssignmentSolver(IPolygonAssignmentSolver):
    """Обертка для умного динамического планировщика"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1):
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        from smart_dynamic_scheduler import SmartDynamicScheduler
        self.scheduler = SmartDynamicScheduler(distance_provider, warehouse_id)
    
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить с помощью умного динамического планировщика"""
        logger.info("Используем умный динамический планировщик")
        return self.scheduler.schedule_polygons(polygons_df, couriers_df, max_time_per_courier)

class ReinforcementAssignmentSolver(IPolygonAssignmentSolver):
    """Обертка для планировщика с подкреплением"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1,
                 courier_service_times: Dict[int, Dict[int, int]] = None):
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        self.courier_service_times = courier_service_times or {}
    
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить с помощью планировщика с подкреплением"""
        logger.info("🎯 Используем планировщик с подкреплением")
        
        from reinforcement_scheduler import ReinforcementScheduler
        
        scheduler = ReinforcementScheduler(
            polygons_df, couriers_df, max_time_per_courier,
            self.distance_provider, self.courier_service_times
        )
        
        return scheduler.solve(polygons_df, couriers_df, max_time_per_courier)

class HybridAssignmentSolver(IPolygonAssignmentSolver):
    """Гибридный решатель: сначала OR-Tools, потом эвристика"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1, 
                 courier_service_times: Dict[int, Dict[int, int]] = None):
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        self.courier_service_times = courier_service_times or {}
        
        # Создаем решатели
        try:
            self.ortools_solver = ORToolsAssignmentSolver(distance_provider, warehouse_id)
            self.heuristic_solver = HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
            self.use_ortools = True
        except ImportError:
            self.heuristic_solver = HeuristicAssignmentSolver(distance_provider, warehouse_id, courier_service_times)
            self.use_ortools = False
            logger.warning("OR-Tools недоступен, используем только эвристику")
    
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Гибридное решение"""
        logger.info("Начинаем гибридное решение")
        
        # Пробуем OR-Tools
        if self.use_ortools:
            logger.info("Пробуем OR-Tools...")
            ortools_result = self.ortools_solver.solve(polygons_df, couriers_df, max_time_per_courier)
            
            if ortools_result:
                assigned_count = sum(len(polygons) for polygons in ortools_result.values())
                logger.info(f"OR-Tools назначил {assigned_count} полигонов")
                
                # Если назначил достаточно полигонов, используем результат
                if assigned_count >= len(polygons_df) * 0.8:  # 80% полигонов
                    return ortools_result
                else:
                    logger.info("OR-Tools назначил мало полигонов, переходим к эвристике")
        
        # Используем эвристику
        logger.info("Используем эвристический решатель...")
        return self.heuristic_solver.solve_with_improvement(polygons_df, couriers_df, max_time_per_courier)

class PyomoHybridAssignmentSolver(IPolygonAssignmentSolver):
    """Гибридный решатель: сначала Pyomo, потом эвристика"""
    
    def __init__(self, distance_provider: IDistanceProvider, warehouse_id: int = 1, **kwargs):
        self.distance_provider = distance_provider
        self.warehouse_id = warehouse_id
        
        # Создаем решатели
        try:
            solver_name = kwargs.get('solver_name', 'cbc')
            self.pyomo_solver = PyomoAssignmentSolver(distance_provider, warehouse_id, solver_name)
            self.heuristic_solver = HeuristicAssignmentSolver(distance_provider, warehouse_id)
            self.use_pyomo = True
        except ImportError:
            self.heuristic_solver = HeuristicAssignmentSolver(distance_provider, warehouse_id)
            self.use_pyomo = False
            logger.warning("Pyomo недоступен, используем только эвристику")
    
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Гибридное решение с Pyomo"""
        logger.info("Начинаем гибридное решение с Pyomo")
        
        # Пробуем Pyomo
        if self.use_pyomo:
            logger.info("Пробуем Pyomo...")
            pyomo_result = self.pyomo_solver.solve(polygons_df, couriers_df, max_time_per_courier)
            
            if pyomo_result:
                assigned_count = sum(len(polygons) for polygons in pyomo_result.values())
                logger.info(f"Pyomo назначил {assigned_count} полигонов")
                
                # Если назначил достаточно полигонов, используем результат
                if assigned_count >= len(polygons_df) * 0.8:  # 80% полигонов
                    return pyomo_result
                else:
                    logger.info("Pyomo назначил мало полигонов, пробуем релаксацию...")
                    
                    # Пробуем с релаксацией
                    relaxed_result = self.pyomo_solver.solve_with_relaxation(
                        polygons_df, couriers_df, max_time_per_courier
                    )
                    
                    if relaxed_result:
                        relaxed_count = sum(len(polygons) for polygons in relaxed_result.values())
                        logger.info(f"Pyomo (релаксация) назначил {relaxed_count} полигонов")
                        
                        if relaxed_count >= len(polygons_df) * 0.8:
                            return relaxed_result
                        else:
                            logger.info("Pyomo с релаксацией назначил мало полигонов, переходим к эвристике")
        
        # Используем эвристику
        logger.info("Используем эвристический решатель...")
        return self.heuristic_solver.solve_with_improvement(polygons_df, couriers_df, max_time_per_courier)
