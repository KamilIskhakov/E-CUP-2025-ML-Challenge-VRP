"""
Интерфейсы для соблюдения принципов SOLID
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import polars as pl

class IDistanceProvider(ABC):
    """Интерфейс для получения расстояний"""
    
    @abstractmethod
    def get_distance(self, from_id: int, to_id: int) -> int:
        """Получить расстояние между двумя точками"""
        pass
    
    @abstractmethod
    def get_distances_batch(self, from_ids: List[int], to_ids: List[int]) -> List[int]:
        """Получить расстояния для батча точек"""
        pass

class IPolygonAssignmentSolver(ABC):
    """Интерфейс для решателя назначения полигонов"""
    
    @abstractmethod
    def solve(self, polygons_df: pl.DataFrame, couriers_df: pl.DataFrame, 
              max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Решить задачу назначения полигонов курьерам"""
        pass

class ILocalSearchOptimizer(ABC):
    """Интерфейс для локального поиска"""
    
    @abstractmethod
    def optimize(self, assignment: Dict[int, List[int]], 
                polygons_df: pl.DataFrame, 
                couriers_df: pl.DataFrame,
                max_time_per_courier: int = 43200) -> Dict[int, List[int]]:
        """Оптимизировать назначение с помощью локального поиска"""
        pass

class IPolygonValidator(ABC):
    """Интерфейс для валидации полигонов"""
    
    @abstractmethod
    def validate_polygon(self, polygon_id: int, polygon_cost: int, 
                        max_time_per_courier: int) -> bool:
        """Проверить, можно ли назначить полигон курьеру"""
        pass
    
    @abstractmethod
    def get_unassignable_polygons(self, polygons_df: pl.DataFrame, 
                                 max_time_per_courier: int) -> List[int]:
        """Получить список полигонов, которые нельзя назначить"""
        pass

class IAssignmentValidator(ABC):
    """Интерфейс для валидации назначений"""
    
    @abstractmethod
    def validate_assignment(self, assignment: Dict[int, List[int]], 
                          polygons_df: pl.DataFrame,
                          max_time_per_courier: int) -> Dict:
        """Валидировать назначение"""
        pass
