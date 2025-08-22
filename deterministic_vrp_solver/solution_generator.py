import json
import logging
from typing import List, Dict, Tuple
import polars as pl
import sqlite3

logger = logging.getLogger(__name__)

class SolutionGenerator:
    """Генератор финального решения в требуемом формате"""
    
    def __init__(self, conn: sqlite3.Connection, warehouse_id: int = 0):
        self.conn = conn
        self.warehouse_id = warehouse_id
        self.optimized_routes: Dict[int, Dict] = {}
    
    def generate_final_routes(self, optimized_routes: Dict[int, Dict], 
                            polygons_df: pl.DataFrame,
                            orders_df: pl.DataFrame) -> List[Dict]:
        """
        Генерация финальных маршрутов в формате solution.json
        
        Args:
            optimized_routes: Оптимизированные маршруты курьеров
            polygons_df: DataFrame с информацией о полигонах
            orders_df: DataFrame с заказами
        
        Returns:
            Список маршрутов в формате {"courier_id": X, "route": [0, order1, order2, ..., 0]}
        """
        logger.info("Генерация финальных маршрутов")
        
        final_routes = []
        
        for courier_id, route_info in optimized_routes.items():
            logger.info(f"Генерация маршрута для курьера {courier_id}")
            
            # Строим полный маршрут
            full_route = self._build_full_route(courier_id, route_info, polygons_df, orders_df)
            
            final_routes.append({
                "courier_id": courier_id,
                "route": full_route
            })
        
        logger.info(f"Сгенерировано {len(final_routes)} маршрутов")
        return final_routes
    
    def _build_full_route(self, courier_id: int, route_info: Dict, 
                         polygons_df: pl.DataFrame, orders_df: pl.DataFrame) -> List[int]:
        """
        Построение полного маршрута курьера
        
        Args:
            courier_id: ID курьера
            route_info: Информация о маршруте
            polygons_df: DataFrame с полигонами
            orders_df: DataFrame с заказами
        
        Returns:
            Полный маршрут [0, order1, order2, ..., 0]
        """
        full_route = [self.warehouse_id]

        def rotate_to_start(route: List[int], start_node: int) -> List[int]:
            if not route:
                return []
            try:
                idx = route.index(start_node)
                return route[idx:] + route[:idx]
            except ValueError:
                return route

        for polygon_id in route_info.get('polygon_order', []):
            polygon_row = polygons_df.filter(pl.col('MpId') == polygon_id).row(0, named=True)
            optimal_route = polygon_row['optimal_route']
            portal_id = polygon_row['portal_id']

            # Внутренний маршрут начинаем с портала, чтобы межполигонные переходы шли портал->портал
            rotated = rotate_to_start(optimal_route, portal_id)
            full_route.extend(rotated)

        full_route.append(self.warehouse_id)
        return full_route
    
    def validate_solution(self, routes: List[Dict], orders_df: pl.DataFrame) -> Dict:
        """
        Валидация финального решения
        
        Args:
            routes: Список маршрутов
            orders_df: DataFrame с заказами
        
        Returns:
            Словарь с результатами валидации
        """
        logger.info("Валидация финального решения")
        
        # Преобразуем LazyFrame в DataFrame для подсчета
        if hasattr(orders_df, 'collect'):
            orders_df_collected = orders_df.collect()
        else:
            orders_df_collected = orders_df
            
        validation_result = {
            'is_valid': True,
            'total_orders': len(orders_df_collected),
            'assigned_orders': 0,
            'unassigned_orders': [],
            'duplicate_orders': [],
            'max_route_time': 0,
            'total_route_time': 0,
            'violations': []
        }
        
        # Собираем все назначенные заказы
        assigned_orders = set()
        all_orders = set(orders_df_collected['ID'].to_list())
        
        for route in routes:
            courier_id = route['courier_id']
            route_orders = route['route'][1:-1]  # Исключаем склад
            
            # Проверяем дубли
            for order_id in route_orders:
                if order_id in assigned_orders:
                    validation_result['duplicate_orders'].append(order_id)
                    validation_result['is_valid'] = False
                else:
                    assigned_orders.add(order_id)
            
            # Вычисляем время маршрута: используем оптимизированные времена, если доступны
            if self.optimized_routes and courier_id in self.optimized_routes:
                route_time = int(self.optimized_routes[courier_id].get('total_time', 0))
            else:
                route_time = self._calculate_route_time(route['route'])
            validation_result['total_route_time'] += route_time
            validation_result['max_route_time'] = max(validation_result['max_route_time'], route_time)
            
            # Проверяем ограничение по времени
            if route_time > 43200:  # 12 часов
                validation_result['violations'].append(f"Курьер {courier_id}: превышение времени {route_time} сек")
                validation_result['is_valid'] = False
        
        validation_result['assigned_orders'] = len(assigned_orders)
        validation_result['unassigned_orders'] = list(all_orders - assigned_orders)
        
        # Не считаем неназначенные заказы критерием невалидности
        # (оставляем is_valid как True/False только по явным нарушениям времени/дубли)
        
        # Логируем результаты
        logger.info(f"Всего заказов: {validation_result['total_orders']}")
        logger.info(f"Назначено заказов: {validation_result['assigned_orders']}")
        logger.info(f"Неназначено заказов: {len(validation_result['unassigned_orders'])}")
        logger.info(f"Дубли заказов: {len(validation_result['duplicate_orders'])}")
        logger.info(f"Максимальное время маршрута: {validation_result['max_route_time']} сек")
        logger.info(f"Общее время маршрутов: {validation_result['total_route_time']} сек")
        
        if validation_result['violations']:
            logger.warning(f"Нарушения: {validation_result['violations']}")
        
        return validation_result
    
    def _calculate_route_time(self, route: List[int]) -> int:
        """Вычисление времени маршрута"""
        if len(route) < 3:  # Только склад
            return 0
        
        total_time = 0
        
        # Время перемещения между точками
        for i in range(len(route) - 1):
            from_id = route[i]
            to_id = route[i + 1]
            
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT d FROM dists WHERE f = ? AND t = ?",
                (from_id, to_id)
            )
            result = cursor.fetchone()
            distance = result[0] if result else 0
            # Если расстояние 0, значит нет пути - возвращаем большое значение
            distance = distance if distance > 0 else 999999
            
            total_time += distance
        
        # Добавляем сервисное время (упрощенно)
        service_time = len(route[1:-1]) * 300  # 5 минут на заказ
        total_time += service_time
        
        return total_time
    
    def generate_statistics(self, routes: List[Dict], orders_df: pl.DataFrame) -> Dict:
        """
        Генерация статистики решения
        
        Args:
            routes: Список маршрутов
            orders_df: DataFrame с заказами
        
        Returns:
            Словарь со статистикой
        """
        logger.info("Генерация статистики решения")
        
        # Убеждаемся, что orders_df - это обычный DataFrame, а не LazyFrame
        if hasattr(orders_df, 'collect'):
            orders_df = orders_df.collect()
        
        stats = {
            'total_couriers': len(routes),
            'total_orders': len(orders_df),
            'assigned_orders': 0,
            'route_times': [],
            'route_lengths': [],
            'polygons_per_courier': [],
            'orders_per_courier': []
        }
        
        assigned_orders = set()
        
        for route in routes:
            route_orders = route['route'][1:-1]  # Исключаем склад
            courier_id = route['courier_id']
            if self.optimized_routes and courier_id in self.optimized_routes:
                route_time = int(self.optimized_routes[courier_id].get('total_time', 0))
            else:
                route_time = self._calculate_route_time(route['route'])
            
            stats['route_times'].append(route_time)
            stats['route_lengths'].append(len(route['route']))
            stats['orders_per_courier'].append(len(route_orders))
            stats['assigned_orders'] += len(route_orders)
            
            assigned_orders.update(route_orders)
        
        # Вычисляем дополнительные метрики
        if stats['route_times']:
            stats['avg_route_time'] = sum(stats['route_times']) / len(stats['route_times'])
            stats['min_route_time'] = min(stats['route_times'])
            stats['max_route_time'] = max(stats['route_times'])
            stats['total_route_time'] = sum(stats['route_times'])
        
        if stats['orders_per_courier']:
            stats['avg_orders_per_courier'] = sum(stats['orders_per_courier']) / len(stats['orders_per_courier'])
            stats['min_orders_per_courier'] = min(stats['orders_per_courier'])
            stats['max_orders_per_courier'] = max(stats['orders_per_courier'])
        
        stats['unassigned_orders'] = len(orders_df) - stats['assigned_orders']
        stats['assignment_rate'] = stats['assigned_orders'] / len(orders_df) if len(orders_df) > 0 else 0
        
        # Логируем статистику
        logger.info(f"Статистика решения:")
        logger.info(f"  Курьеров: {stats['total_couriers']}")
        logger.info(f"  Заказов: {stats['assigned_orders']}/{stats['total_orders']} ({stats['assignment_rate']:.2%})")
        logger.info(f"  Среднее время маршрута: {int(stats.get('avg_route_time', 0))} сек")
        logger.info(f"  Максимальное время маршрута: {int(stats.get('max_route_time', 0))} сек")
        logger.info(f"  Общее время: {int(stats.get('total_route_time', 0))} сек")
        
        return stats

def generate_solution(optimized_routes: Dict[int, Dict], 
                    polygons_df: pl.DataFrame,
                    orders_df: pl.DataFrame,
                    conn: sqlite3.Connection,
                    output_path: str = "solution.json") -> Dict:
    """
    Основная функция генерации решения
    
    Args:
        optimized_routes: Оптимизированные маршруты курьеров
        polygons_df: DataFrame с информацией о полигонах
        orders_df: DataFrame с заказами
        conn: Подключение к базе данных
        output_path: Путь для сохранения решения
    
    Returns:
        Словарь с результатами генерации
    """
    generator = SolutionGenerator(conn, warehouse_id=0)
    generator.optimized_routes = optimized_routes
    
    # Генерируем финальные маршруты
    final_routes = generator.generate_final_routes(optimized_routes, polygons_df, orders_df)
    
    # Валидируем решение
    validation_result = generator.validate_solution(final_routes, orders_df)
    
    # Генерируем статистику
    statistics = generator.generate_statistics(final_routes, orders_df)
    
    # Сохраняем решение всегда, независимо от валидации
    solution = {"routes": final_routes}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(solution, f, indent=2)
    logger.info(f"Решение сохранено в {output_path}")
    
    return {
        'routes': final_routes,
        'validation': validation_result,
        'statistics': statistics,
        'saved': True
    }
