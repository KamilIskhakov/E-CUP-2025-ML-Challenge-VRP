"""
Валидатор полигонов - проверяет возможность назначения полигонов курьерам
"""

import logging
from typing import List, Dict
import polars as pl
from interfaces import IPolygonValidator

logger = logging.getLogger(__name__)

class PolygonValidator(IPolygonValidator):
    """Валидатор полигонов"""
    
    def __init__(self, max_time_per_courier: int = 43200):
        self.max_time_per_courier = max_time_per_courier
    
    def validate_polygon(self, polygon_id: int, polygon_cost: int, 
                        max_time_per_courier: int = None) -> bool:
        """Проверить, можно ли назначить полигон курьеру"""
        if max_time_per_courier is None:
            max_time_per_courier = self.max_time_per_courier
        
        is_valid = polygon_cost <= max_time_per_courier
        
        if not is_valid:
            logger.warning(f"Полигон {polygon_id} слишком дорогой: {polygon_cost} сек > {max_time_per_courier} сек")
        
        return is_valid
    
    def get_unassignable_polygons(self, polygons_df: pl.DataFrame, 
                                 max_time_per_courier: int = None) -> List[int]:
        """Получить список полигонов, которые нельзя назначить"""
        if max_time_per_courier is None:
            max_time_per_courier = self.max_time_per_courier
        
        # Используем lazy API для фильтрации
        unassignable_polygons = (polygons_df
            .lazy()
            .filter(pl.col('total_cost') > max_time_per_courier)
            .select('MpId')
            .collect()
            .get_column('MpId')
            .to_list()
        )
        
        logger.info(f"Найдено {len(unassignable_polygons)} неназначаемых полигонов")
        return unassignable_polygons
    
    def get_polygon_statistics(self, polygons_df: pl.DataFrame) -> Dict:
        """Получить статистику по полигонам с использованием lazy API"""
        logger.info("Вычисление статистики по полигонам (lazy)")
        
        # Используем lazy API для вычисления статистики
        stats_lazy = (polygons_df
            .lazy()
            .select([
                pl.count().alias('total_polygons'),
                pl.col('total_cost').mean().alias('avg_cost'),
                pl.col('total_cost').max().alias('max_cost'),
                pl.col('total_cost').min().alias('min_cost'),
                pl.col('total_cost').filter(pl.col('total_cost') > self.max_time_per_courier).count().alias('unassignable_polygons')
            ])
            .collect()
            .row(0, named=True)
        )
        
        stats = {
            'total_polygons': stats_lazy['total_polygons'],
            'assignable_polygons': stats_lazy['total_polygons'] - stats_lazy['unassignable_polygons'],
            'unassignable_polygons': stats_lazy['unassignable_polygons'],
            'avg_cost': stats_lazy['avg_cost'],
            'max_cost': stats_lazy['max_cost'],
            'min_cost': stats_lazy['min_cost']
        }
        
        return stats
    
    def filter_assignable_polygons(self, polygons_df: pl.DataFrame, 
                                 max_time_per_courier: int = None) -> pl.DataFrame:
        """Отфильтровать только назначаемые полигоны с использованием lazy API"""
        if max_time_per_courier is None:
            max_time_per_courier = self.max_time_per_courier
        
        logger.info(f"Фильтрация назначаемых полигонов (максимум {max_time_per_courier} сек)")
        
        assignable_polygons = (polygons_df
            .lazy()
            .filter(pl.col('total_cost') <= max_time_per_courier)
            .collect()
        )
        
        logger.info(f"Отфильтровано {len(assignable_polygons)} назначаемых полигонов из {len(polygons_df)}")
        return assignable_polygons
    
    def get_polygon_cost_distribution(self, polygons_df: pl.DataFrame) -> Dict:
        """Получить распределение стоимости полигонов с использованием lazy API"""
        logger.info("Анализ распределения стоимости полигонов")
        
        # Используем lazy API для вычисления квантилей
        distribution = (polygons_df
            .lazy()
            .select([
                pl.col('total_cost').quantile(0.25).alias('q25'),
                pl.col('total_cost').quantile(0.5).alias('q50'),
                pl.col('total_cost').quantile(0.75).alias('q75'),
                pl.col('total_cost').quantile(0.9).alias('q90'),
                pl.col('total_cost').quantile(0.95).alias('q95'),
                pl.col('total_cost').quantile(0.99).alias('q99')
            ])
            .collect()
            .row(0, named=True)
        )
        
        logger.info(f"Распределение стоимости: Q25={distribution['q25']:.0f}, "
                   f"Q50={distribution['q50']:.0f}, Q75={distribution['q75']:.0f}, "
                   f"Q90={distribution['q90']:.0f}, Q95={distribution['q95']:.0f}, "
                   f"Q99={distribution['q99']:.0f}")
        
        return distribution
