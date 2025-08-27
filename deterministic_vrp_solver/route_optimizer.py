import sqlite3
import numpy as np
from typing import List, Dict, Tuple
import logging
import polars as pl
from .polygon import PolygonTSPSolver

logger = logging.getLogger(__name__)

class RouteOptimizer:
    """Оптимизатор маршрутов курьеров на уровне полигонов"""
    
    def __init__(self, conn: sqlite3.Connection, warehouse_id: int = 0, courier_service_times: Dict[int, Dict[int, int]] = None):
        self.conn = conn
        self.warehouse_id = warehouse_id
        self.distance_cache = {}
        self.courier_service_times = courier_service_times or {}
    
    def get_distance(self, from_id: int, to_id: int) -> int:
        """Получение расстояния с кэшированием"""
        key = (from_id, to_id)
        if key not in self.distance_cache:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT d FROM dists WHERE f = ? AND t = ?",
                (from_id, to_id)
            )
            result = cursor.fetchone()
            distance = result[0] if result else 0
                                                                    
            self.distance_cache[key] = int(distance) if (result and distance and distance > 0) else float('inf')
        return self.distance_cache[key]
    
    def get_polygon_cost_for_courier(self, mp_id: int, courier_id: int, polygons_df: pl.DataFrame) -> int:
        """
        Получает стоимость полигона для конкретного курьера с учетом его сервисного времени
        
        Args:
            mp_id: ID полигона
            courier_id: ID курьера
            polygons_df: DataFrame с информацией о полигонах
        
        Returns:
            Общая стоимость полигона для курьера (TSP + сервисное время курьера)
        """
        polygon_row = polygons_df.filter(pl.col('MpId') == mp_id).row(0, named=True)
        
                                                               
        base_cost = int(polygon_row['total_distance'])
                                       
        order_ids = polygon_row.get('order_ids') or []
        num_orders = len(order_ids) if isinstance(order_ids, list) else int(polygon_row.get('order_count', 0) or 0)
                                                              
        per_order_service_time = int(self.courier_service_times.get(courier_id, {}).get(mp_id, 0))
        service_time_total = per_order_service_time * max(0, num_orders)
                                                              
        total_cost = int(base_cost) + int(service_time_total)
        
        logger.debug(f"Полигон {mp_id} для курьера {courier_id}: TSP={base_cost}, сервис={service_time_total}, общая={total_cost}")
        
        return total_cost
    
    def optimize_courier_route(self, polygon_ids: List[int], polygons_df: pl.DataFrame, courier_id: int) -> Dict:
        """
        Оптимизация маршрута курьера на уровне полигонов
        
        Args:
            polygon_ids: Список ID полигонов, назначенных курьеру
            polygons_df: DataFrame с информацией о полигонах
        
        Returns:
            Словарь с оптимизированным маршрутом
        """
        logger.debug(f"optimize_courier_route: {len(polygon_ids)} полигонов")
        
        if not polygon_ids:
            logger.debug("Пустой список полигонов")
            return {
                'polygon_order': [],
                'total_time': 0,
                'route_details': []
            }
        
        if len(polygon_ids) == 1:
            logger.debug(f"Один полигон: {polygon_ids[0]} для курьера {courier_id}")
                                            
            mp_id = polygon_ids[0]
            polygon_row = polygons_df.filter(pl.col('MpId') == mp_id).row(0, named=True)
            portal_id = polygon_row['portal_id']
            
                                                                             
            polygon_cost = self.get_polygon_cost_for_courier(mp_id, courier_id, polygons_df)
            
            if portal_id:
                time_to_polygon = self.get_distance(self.warehouse_id, portal_id)
                time_from_polygon = self.get_distance(portal_id, self.warehouse_id)
                if time_to_polygon == float('inf') or time_from_polygon == float('inf'):
                                                                        
                    return {
                        'polygon_order': [],
                        'total_time': 0,
                        'route_details': []
                    }
                total_time = time_to_polygon + polygon_cost + time_from_polygon
            else:
                                                    
                return {
                    'polygon_order': [],
                    'total_time': 0,
                    'route_details': []
                }
            
            return {
                'polygon_order': [mp_id],
                'total_time': total_time,
                'route_details': [{
                    'polygon_id': mp_id,
                    'portal_id': portal_id,
                    'polygon_cost': polygon_cost,
                    'time_to_polygon': time_to_polygon if portal_id else 0,
                    'time_from_polygon': time_from_polygon if portal_id else 0
                }]
            }
        
        logger.debug(f"Несколько полигонов: {polygon_ids} для курьера {courier_id}, вызываем _solve_polygon_tsp")
                                                             
        return self._solve_polygon_tsp(polygon_ids, polygons_df, courier_id)
    
    def _solve_polygon_tsp(self, polygon_ids: List[int], polygons_df: pl.DataFrame, courier_id: int) -> Dict:
        """Решение TSP на уровне полигонов через их порталы"""
        logger.debug(f"_solve_polygon_tsp: {len(polygon_ids)} полигонов")
        
                                                  
        portal_info = []
        logger.debug(f"Собираем информацию о порталах для {len(polygon_ids)} полигонов")
        
        for i, mp_id in enumerate(polygon_ids):
            logger.debug(f"  Обрабатываем полигон {i+1}/{len(polygon_ids)}: {mp_id}")
            try:
                polygon_row = polygons_df.filter(pl.col('MpId') == mp_id).row(0, named=True)
                portal_id = polygon_row['portal_id']
                if portal_id:
                                                                                     
                    polygon_cost = self.get_polygon_cost_for_courier(mp_id, courier_id, polygons_df)
                    
                    portal_info.append({
                        'polygon_id': mp_id,
                        'portal_id': portal_id,
                        'polygon_cost': polygon_cost
                    })
                    logger.debug(f"    Полигон {mp_id}: портал {portal_id}")
                else:
                    logger.warning(f"Полигон {mp_id} не имеет портала")
            except Exception as e:
                logger.error(f"Ошибка при получении информации о полигоне {mp_id}: {e}")
                raise
        
        logger.debug(f"Найдено {len(portal_info)} полигонов с порталами")
        
        if not portal_info:
            logger.warning("Нет полигонов с порталами")
            return {
                'polygon_order': [],
                'total_time': 0,
                'route_details': []
            }
        
                                         
        portal_ids = [info['portal_id'] for info in portal_info]
        logger.debug(f"Порталы для TSP: {portal_ids}")
        
                                                                   
        portal_indices = portal_ids
        
                                 
        try:
            tsp_solver = PolygonTSPSolver(self.conn)
            logger.debug(f"Запуск TSP для {len(portal_indices)} порталов")
            optimal_portal_route, portal_distance = tsp_solver.solve(portal_indices, start_id=portal_indices[0])
            logger.debug(f"TSP решен: маршрут {optimal_portal_route}, расстояние {portal_distance}")
        except Exception as e:
            logger.error(f"Ошибка при решении TSP для порталов: {e}")
            raise
        
                                        
        warehouse_to_first = self.get_distance(self.warehouse_id, optimal_portal_route[0])
        last_to_warehouse = self.get_distance(optimal_portal_route[-1], self.warehouse_id)
        if warehouse_to_first == float('inf') or last_to_warehouse == float('inf'):
            return {
                'polygon_order': [],
                'total_time': 0,
                'route_details': []
            }
        
        total_distance = warehouse_to_first + portal_distance + last_to_warehouse
        
                                   
                                                                              
                                                                                                            
        total_polygon_cost = sum(info['polygon_cost'] for info in portal_info)
        total_time = total_distance + total_polygon_cost
        
                                                         
        logger.debug(f"Расчет времени маршрута:")
        logger.debug(f"  Расстояние между полигонами: {total_distance} сек")
        logger.debug(f"  Время прохождения полигонов: {total_polygon_cost} сек")
        logger.debug(f"  Общее время: {total_time} сек")
        
                                                     
        if total_time > 100000:                  
            logger.warning(f"Подозрительно большое время маршрута: {total_time} сек ({total_time/3600:.1f} ч)")
            logger.warning(f"  Расстояние между полигонами: {total_distance} сек")
            logger.warning(f"  Время полигонов: {total_polygon_cost} сек")
        
                                 
        route_details = []
        portal_to_polygon = {info['portal_id']: info['polygon_id'] for info in portal_info}
        for i, portal_id in enumerate(optimal_portal_route):
            info = next(info for info in portal_info if info['portal_id'] == portal_id)
            
            if i == 0:
                time_to_polygon = warehouse_to_first
            else:
                prev_portal = optimal_portal_route[i-1]
                time_to_polygon = self.get_distance(prev_portal, portal_id)
                if time_to_polygon == float('inf'):
                    return {
                        'polygon_order': [],
                        'total_time': 0,
                        'route_details': []
                    }
            
            if i == len(optimal_portal_route) - 1:
                time_from_polygon = last_to_warehouse
            else:
                next_portal = optimal_portal_route[i+1]
                time_from_polygon = self.get_distance(portal_id, next_portal)
                if time_from_polygon == float('inf'):
                    return {
                        'polygon_order': [],
                        'total_time': 0,
                        'route_details': []
                    }
            
            route_details.append({
                'polygon_id': info['polygon_id'],
                'portal_id': portal_id,
                'polygon_cost': info['polygon_cost'],
                'time_to_polygon': time_to_polygon,
                'time_from_polygon': time_from_polygon
            })
        
                                                                 
        polygon_order = [portal_to_polygon[p] for p in optimal_portal_route]
        
        return {
            'polygon_order': polygon_order,
            'total_time': total_time,
            'route_details': route_details
        }
    
    def optimize_all_courier_routes(self, assignment: Dict[int, List[int]], 
                                  polygons_df: pl.DataFrame) -> Dict[int, Dict]:
        """
        Оптимизация маршрутов всех курьеров
        
        Args:
            assignment: Назначение полигонов курьерам
            polygons_df: DataFrame с информацией о полигонах
        
        Returns:
            Словарь {courier_id: optimized_route}
        """
        logger.info("Начинаем оптимизацию маршрутов курьеров")
        logger.info(f"Всего курьеров для оптимизации: {len(assignment)}")
        logger.info(f"Размер assignment: {len(assignment)}")
        logger.info(f"Размер polygons_df: {len(polygons_df)}")
        
        optimized_routes = {}
        
        for i, (courier_id, polygon_ids) in enumerate(assignment.items()):
            logger.info(f"[{i+1}/{len(assignment)}] Оптимизация маршрута курьера {courier_id} ({len(polygon_ids)} полигонов)")
            
                                                   
            logger.info(f"  Полигоны курьера {courier_id}: {polygon_ids[:5]}{'...' if len(polygon_ids) > 5 else ''}")
            
            try:
                logger.info(f"  Вызываем optimize_courier_route для курьера {courier_id}")
                optimized_route = self.optimize_courier_route(polygon_ids, polygons_df, courier_id)
                logger.info(f"  optimize_courier_route завершился успешно для курьера {courier_id}")
                
                optimized_routes[courier_id] = optimized_route
                
                logger.info(f"Курьер {courier_id}: время = {optimized_route['total_time']} сек")
            except Exception as e:
                logger.error(f"Ошибка при оптимизации курьера {courier_id}: {e}")
                logger.error(f"  Полигоны: {polygon_ids}")
                raise
        
                                          
        if optimized_routes:
            max_time = max(route['total_time'] for route in optimized_routes.values())
            logger.info(f"Максимальное время курьера: {max_time} сек")
        else:
            max_time = 0
            logger.warning("Нет назначенных маршрутов")
        
        if max_time > 43200:            
            logger.warning(f"Превышение лимита времени: {max_time} сек > 43200 сек")
        
        return optimized_routes
    
    def improve_routes_local_search(self, optimized_routes: Dict[int, Dict],
                                  assignment: Dict[int, List[int]],
                                  polygons_df: pl.DataFrame) -> Dict[int, Dict]:
        """
        Улучшение маршрутов с помощью локального поиска
        
        Args:
            optimized_routes: Текущие оптимизированные маршруты
            assignment: Назначение полигонов курьерам
            polygons_df: DataFrame с информацией о полигонах
        
        Returns:
            Улучшенные маршруты
        """
        logger.info("Начинаем локальный поиск для улучшения маршрутов")
        
        improved = True
        iteration = 0
        max_iterations = 5
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
                                                        
            courier_ids = list(optimized_routes.keys())
            
            for i, courier1 in enumerate(courier_ids):
                for courier2 in courier_ids[i+1:]:
                    route1 = optimized_routes[courier1]
                    route2 = optimized_routes[courier2]
                    
                                                    
                    for poly1 in route1['polygon_order']:
                        for poly2 in route2['polygon_order']:
                                                     
                            new_polygons1 = [p for p in route1['polygon_order'] if p != poly1] + [poly2]
                            new_polygons2 = [p for p in route2['polygon_order'] if p != poly2] + [poly1]
                            
                            new_route1 = self.optimize_courier_route(new_polygons1, polygons_df, courier1)
                            new_route2 = self.optimize_courier_route(new_polygons2, polygons_df, courier2)
                            
                                                 
                            old_total = route1['total_time'] + route2['total_time']
                            new_total = new_route1['total_time'] + new_route2['total_time']
                            
                            if (new_route1['total_time'] <= 43200 and 
                                new_route2['total_time'] <= 43200 and
                                new_total < old_total):
                                
                                                 
                                optimized_routes[courier1] = new_route1
                                optimized_routes[courier2] = new_route2
                                assignment[courier1] = new_polygons1
                                assignment[courier2] = new_polygons2
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        logger.info(f"Локальный поиск маршрутов завершен за {iteration} итераций")
        return optimized_routes

    def improve_routes_global_relocate(self,
                                       optimized_routes: Dict[int, Dict],
                                       assignment: Dict[int, List[int]],
                                       polygons_df: pl.DataFrame,
                                       time_cap: int = 43200,
                                       max_iters: int = 2,
                                       tail_sample: int = 3) -> Dict[int, Dict]:
        """Глобальный релокейт: переносим хвостовые полигоны из любых курьеров к тем,
        у кого вставка даёт наибольшее снижение суммарного времени, при ограничении по 12 часам.
        """
        if not optimized_routes:
            return optimized_routes

        for _ in range(max_iters):
            items = sorted(
                [(cid, r.get('total_time', 0)) for cid, r in optimized_routes.items() if r],
                key=lambda x: x[1], reverse=True,
            )
            if not items:
                break
            improved = False
                                                   
            for cid_o, _ in items:
                route_o = optimized_routes.get(cid_o) or {}
                polys_o = list(route_o.get('polygon_order') or [])
                if not polys_o:
                    continue
                                                       
                tail = polys_o[-tail_sample:] if len(polys_o) > tail_sample else polys_o[:]
                best = None
                best_delta = 0.0
                for poly in reversed(tail):
                                                                     
                    for cid_t, _ in items:
                        if cid_t == cid_o:
                            continue
                        route_t = optimized_routes.get(cid_t) or {}
                        polys_t = list(route_t.get('polygon_order') or [])
                        new_polys_o = [p for p in polys_o if p != poly]
                        new_polys_t = polys_t + [poly]
                        new_route_o = self.optimize_courier_route(new_polys_o, polygons_df, cid_o)
                        new_route_t = self.optimize_courier_route(new_polys_t, polygons_df, cid_t)
                        if (new_route_o['total_time'] == 0 or new_route_t['total_time'] == 0):
                            continue
                        if new_route_o['total_time'] > time_cap or new_route_t['total_time'] > time_cap:
                            continue
                        old_sum = float(route_o.get('total_time', 0)) + float(route_t.get('total_time', 0))
                        new_sum = float(new_route_o.get('total_time', 0)) + float(new_route_t.get('total_time', 0))
                        delta = old_sum - new_sum
                        if delta > best_delta:
                            best_delta = delta
                            best = (poly, cid_t, new_route_o, new_route_t, new_polys_o, new_polys_t)
                if best is not None and best_delta > 0.0:
                    poly, cid_t, new_route_o, new_route_t, new_polys_o, new_polys_t = best
                    optimized_routes[cid_o] = new_route_o
                    optimized_routes[cid_t] = new_route_t
                    assignment[cid_o] = new_polys_o
                    assignment[cid_t] = new_polys_t
                    improved = True
                    break
            if not improved:
                break
        return optimized_routes

    def improve_routes_cross_exchange(self,
                                      optimized_routes: Dict[int, Dict],
                                      assignment: Dict[int, List[int]],
                                      polygons_df: pl.DataFrame,
                                      time_cap: int = 43200,
                                      max_iters: int = 2) -> Dict[int, Dict]:
        """Межкурьерский cross-exchange/relocate для снижения перегрузок и суммы времен.

        Стратегия:
        - Ищем перегруженных курьеров (по времени) и недогруженных.
        - Для верхних N перегруженных: пытаемся
          1) relocate: перенести один хвостовой полигон к недогруженному,
          2) swap: обменять по одному полигону.
        Принимаем шаг, если обе траектории <= time_cap и суммарное время снижается.
        """
        if not optimized_routes:
            return optimized_routes

        for _ in range(max_iters):
                                                      
            items = [(cid, r.get('total_time', 0)) for cid, r in optimized_routes.items() if r]
            if not items:
                break
            items.sort(key=lambda x: x[1], reverse=True)
            overloaded = [cid for cid, _ in items[:10]]                        
            underloaded = [cid for cid, _ in items[-20:]]                           

            improved = False
                                                       
            for cid_o in overloaded:
                route_o = optimized_routes.get(cid_o) or {}
                polys_o = list(route_o.get('polygon_order') or [])
                if not polys_o:
                    continue
                                                                                 
                for poly in reversed(polys_o):
                    best_delta = 0.0
                    best_target = None
                    best_new_o = None
                    best_new_t = None
                    best_new_polys_o = None
                    best_new_polys_t = None
                    for cid_t in underloaded:
                        if cid_t == cid_o:
                            continue
                        route_t = optimized_routes.get(cid_t) or {}
                        polys_t = list(route_t.get('polygon_order') or [])
                                               
                        new_polys_o = [p for p in polys_o if p != poly]
                        new_polys_t = polys_t + [poly]
                        new_route_o = self.optimize_courier_route(new_polys_o, polygons_df, cid_o)
                        new_route_t = self.optimize_courier_route(new_polys_t, polygons_df, cid_t)
                        if (new_route_o['total_time'] == 0 or new_route_t['total_time'] == 0):
                            continue
                        if new_route_o['total_time'] > time_cap or new_route_t['total_time'] > time_cap:
                            continue
                        old_sum = float(route_o.get('total_time', 0)) + float(route_t.get('total_time', 0))
                        new_sum = float(new_route_o.get('total_time', 0)) + float(new_route_t.get('total_time', 0))
                        delta = old_sum - new_sum
                        if delta > best_delta:
                            best_delta = delta
                            best_target = cid_t
                            best_new_o = new_route_o
                            best_new_t = new_route_t
                            best_new_polys_o = new_polys_o
                            best_new_polys_t = new_polys_t
                    if best_target is not None and best_delta > 0.0:
                                            
                        optimized_routes[cid_o] = best_new_o
                        optimized_routes[best_target] = best_new_t
                        assignment[cid_o] = best_new_polys_o
                        assignment[best_target] = best_new_polys_t
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

                                                      
            for cid_o in overloaded:
                route_o = optimized_routes.get(cid_o) or {}
                polys_o = list(route_o.get('polygon_order') or [])
                if not polys_o:
                    continue
                for cid_t in underloaded:
                    if cid_t == cid_o:
                        continue
                    route_t = optimized_routes.get(cid_t) or {}
                    polys_t = list(route_t.get('polygon_order') or [])
                    if not polys_t:
                        continue
                    best_delta = 0.0
                    best_pair = None
                    best_new_o = None
                    best_new_t = None
                    best_new_polys_o = None
                    best_new_polys_t = None
                    for p_o in polys_o:
                        for p_t in polys_t:
                            new_polys_o = [p for p in polys_o if p != p_o] + [p_t]
                            new_polys_t = [p for p in polys_t if p != p_t] + [p_o]
                            new_route_o = self.optimize_courier_route(new_polys_o, polygons_df, cid_o)
                            new_route_t = self.optimize_courier_route(new_polys_t, polygons_df, cid_t)
                            if (new_route_o['total_time'] == 0 or new_route_t['total_time'] == 0):
                                continue
                            if new_route_o['total_time'] > time_cap or new_route_t['total_time'] > time_cap:
                                continue
                            old_sum = float(route_o.get('total_time', 0)) + float(route_t.get('total_time', 0))
                            new_sum = float(new_route_o.get('total_time', 0)) + float(new_route_t.get('total_time', 0))
                            delta = old_sum - new_sum
                            if delta > best_delta:
                                best_delta = delta
                                best_pair = (p_o, p_t)
                                best_new_o = new_route_o
                                best_new_t = new_route_t
                                best_new_polys_o = new_polys_o
                                best_new_polys_t = new_polys_t
                    if best_pair is not None and best_delta > 0.0:
                        optimized_routes[cid_o] = best_new_o
                        optimized_routes[cid_t] = best_new_t
                        assignment[cid_o] = best_new_polys_o
                        assignment[cid_t] = best_new_polys_t
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return optimized_routes

def optimize_courier_routes(assignment: Dict[int, List[int]], 
                          polygons_df: pl.DataFrame,
                          conn: sqlite3.Connection,
                          courier_service_times: Dict[int, Dict[int, int]] = None) -> Dict[int, Dict]:
    """
    Основная функция оптимизации маршрутов курьеров
    
    Args:
        assignment: Назначение полигонов курьерам
        polygons_df: DataFrame с информацией о полигонах
        conn: Подключение к базе данных
        courier_service_times: Сервисные времена курьеров {courier_id: {mp_id: service_time}}
    
    Returns:
        Словарь {courier_id: optimized_route}
    """
    logger.info("=== optimize_courier_routes: начало ===")
    logger.info(f"Создаем RouteOptimizer с сервисными временами")
    optimizer = RouteOptimizer(conn, courier_service_times=courier_service_times)
    logger.info(f"RouteOptimizer создан успешно")
    
                           
    logger.info(f"Вызываем optimize_all_courier_routes")
    optimized_routes = optimizer.optimize_all_courier_routes(assignment, polygons_df)
    logger.info(f"optimize_all_courier_routes завершился успешно")
    
                                                    
                                                              
                                                   
       
    
    logger.info("=== optimize_courier_routes: завершение ===")
    return optimized_routes
