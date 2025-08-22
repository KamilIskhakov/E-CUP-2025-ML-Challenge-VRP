import sqlite3
import numpy as np
from typing import List, Tuple, Dict
import logging
from functools import lru_cache
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time
import os
import pickle

logger = logging.getLogger(__name__)

class PolygonTSPSolver:
    """Улучшенный решатель TSP внутри микрополигонов с поддержкой больших полигонов"""
    
    def __init__(self, conn: sqlite3.Connection, max_exact_size: int = 15, use_parallel: bool = True):
        self.conn = conn
        self.distance_cache = {}
        self.max_exact_size = max_exact_size  # Увеличиваем лимит для точного решения
        self.use_parallel = use_parallel
    
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
            # Если расстояние 0, значит нет прямого пути - ищем через промежуточные точки
            if distance == 0:
                distance = self._find_path_through_intermediates(from_id, to_id)
            self.distance_cache[key] = distance
        return self.distance_cache[key]
    
    def _find_path_through_intermediates(self, from_id: int, to_id: int) -> int:
        """Поиск пути через промежуточные точки"""
        cursor = self.conn.cursor()
        
        # Получаем все доступные промежуточные точки
        cursor.execute("SELECT DISTINCT f FROM dists WHERE f != ? AND f != ?", (from_id, to_id))
        intermediates = [row[0] for row in cursor.fetchall()]
        
        if not intermediates:
            return 999999  # Нет промежуточных точек
        
        # Ищем минимальный путь через промежуточные точки
        min_path = 999999
        
        for intermediate in intermediates[:10]:  # Ограничиваем поиск первыми 10 точками
            # Путь: from_id -> intermediate -> to_id
            cursor.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (from_id, intermediate))
            result1 = cursor.fetchone()
            if not result1 or result1[0] == 0:
                continue
                
            cursor.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (intermediate, to_id))
            result2 = cursor.fetchone()
            if not result2 or result2[0] == 0:
                continue
            
            path_length = result1[0] + result2[0]
            if path_length < min_path:
                min_path = path_length
        
        return min_path
    
    def solve_tsp_dynamic(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], int]:
        """
        Улучшенное решение TSP с использованием динамического программирования
        
        Args:
            order_ids: Список ID заказов в полигоне
            start_id: ID начальной точки (если None, выбирается оптимально)
        
        Returns:
            Tuple[оптимальный маршрут, общая стоимость]
        """
        n = len(order_ids)
        
        if n == 0:
            return [], 0
        if n == 1:
            return order_ids, 0
        if n == 2:
            dist = self.get_distance(order_ids[0], order_ids[1])
            return order_ids, dist
        
        # Увеличиваем лимит для точного решения
        if n > self.max_exact_size:
            logger.info(f"Полигон с {n} точками использует улучшенную эвристику")
            return self.solve_tsp_advanced_heuristic(order_ids, start_id)
        
        # Создаем матрицу расстояний
        dist_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i][j] = self.get_distance(order_ids[i], order_ids[j])
        
        # Динамическое программирование
        best_cost, best_path = self._held_karp_tsp(dist_matrix, start_id)
        
        # Преобразуем индексы обратно в ID заказов
        result_path = [order_ids[i] for i in best_path]
        
        return result_path, best_cost
    
    def _held_karp_tsp(self, dist_matrix: np.ndarray, start_id: int = None) -> Tuple[int, List[int]]:
        """Реализация алгоритма Хелда-Карпа для TSP"""
        n = len(dist_matrix)
        
        # Если не указана начальная точка, выбираем оптимальную
        if start_id is None:
            start_id = 0
        
        # Проверяем, что start_id в допустимом диапазоне
        if start_id >= n:
            logger.warning(f"start_id {start_id} выходит за пределы матрицы размером {n}, используем 0")
            start_id = 0
        
        # Мемоизация для подзадач
        memo = {}
        
        def solve_dp(mask: int, pos: int) -> Tuple[int, List[int]]:
            """Рекурсивное решение с мемоизацией"""
            if mask == (1 << n) - 1:  # Все города посещены
                return dist_matrix[pos][start_id], [pos]
            
            state = (mask, pos)
            if state in memo:
                return memo[state]
            
            min_cost = float('inf')
            best_path = []
            
            for next_pos in range(n):
                if mask & (1 << next_pos) == 0:  # Город не посещен
                    new_mask = mask | (1 << next_pos)
                    cost, path = solve_dp(new_mask, next_pos)
                    total_cost = dist_matrix[pos][next_pos] + cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_path = [pos] + path
            
            memo[state] = (min_cost, best_path)
            return memo[state]
        
        # Запускаем с начальной точки
        initial_mask = 1 << start_id
        total_cost, path = solve_dp(initial_mask, start_id)
        
        return total_cost, path
    
    def solve_tsp_heuristic(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], int]:
        """Эвристическое решение TSP (ближайший сосед + 2-opt)"""
        n = len(order_ids)
        
        if n == 0:
            return [], 0
        if n == 1:
            return order_ids, 0
        
        # Начинаем с ближайшего соседа
        if start_id is None:
            start_id = order_ids[0]
        
        unvisited = set(order_ids)
        current = start_id
        path = [current]
        unvisited.remove(current)
        total_cost = 0
        
        # Жадный алгоритм ближайшего соседа
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.get_distance(current, x))
            cost = self.get_distance(current, nearest)
            total_cost += cost
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Добавляем возврат к началу
        if len(path) > 1:
            total_cost += self.get_distance(path[-1], path[0])
        
        # Применяем 2-opt улучшение
        improved_path, improved_cost = self._two_opt_improvement(path, total_cost)
        
        return improved_path, improved_cost
    
    def _two_opt_improvement(self, path: List[int], initial_cost: int) -> Tuple[List[int], int]:
        """Улучшение маршрута с помощью 2-opt"""
        n = len(path)
        if n <= 3:
            return path, initial_cost
        
        best_path = path.copy()
        best_cost = initial_cost
        improved = True
        
        while improved:
            improved = False
            
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    
                    # Создаем новый маршрут с разворотом сегмента
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    
                    # Вычисляем новую стоимость
                    new_cost = 0
                    for k in range(len(new_path) - 1):
                        new_cost += self.get_distance(new_path[k], new_path[k + 1])
                    
                    if new_cost < best_cost:
                        best_path = new_path
                        best_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_path, best_cost
    
    def solve_tsp_advanced_heuristic(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], int]:
        """Улучшенная эвристика для больших полигонов"""
        n = len(order_ids)
        
        if n == 0:
            return [], 0
        if n == 1:
            return order_ids, 0
        
        # Пробуем несколько эвристик и выбираем лучшую
        heuristics = [
            self._nearest_neighbor_heuristic,
            self._christofides_heuristic,
            self._genetic_heuristic
        ]
        
        best_path = None
        best_cost = float('inf')
        best_heuristic = None
        
        logger.info(f"🔍 Тестирование {len(heuristics)} эвристик для полигона с {n} точками")
        
        for i, heuristic in enumerate(heuristics):
            try:
                logger.info(f"  {i+1}. Тестируем {heuristic.__name__}...")
                path, cost = heuristic(order_ids, start_id)
                logger.info(f"     Результат: стоимость={cost}")
                
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
                    best_heuristic = heuristic.__name__
                    logger.info(f"     ✨ Новый лучший результат!")
            except Exception as e:
                logger.warning(f"     ❌ Эвристика {heuristic.__name__} не удалась: {e}")
                continue
        
        logger.info(f"🏆 Выбрана эвристика: {best_heuristic} (стоимость: {best_cost})")
        
        if best_path is None:
            # Fallback к простой эвристике
            return self.solve_tsp_heuristic(order_ids, start_id)
        
        return best_path, best_cost
    
    def _nearest_neighbor_heuristic(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], int]:
        """Улучшенный алгоритм ближайшего соседа с множественными стартовыми точками"""
        if start_id is None:
            # Пробуем несколько стартовых точек
            candidates = order_ids[:min(5, len(order_ids))]
            best_path = None
            best_cost = float('inf')
            
            for candidate in candidates:
                path, cost = self._nearest_neighbor_single_start(order_ids, candidate)
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
            
            return best_path, best_cost
        else:
            return self._nearest_neighbor_single_start(order_ids, start_id)
    
    def _nearest_neighbor_single_start(self, order_ids: List[int], start_id: int) -> Tuple[List[int], int]:
        """Алгоритм ближайшего соседа с одной стартовой точкой"""
        unvisited = set(order_ids)
        current = start_id
        path = [current]
        unvisited.remove(current)
        total_cost = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.get_distance(current, x))
            cost = self.get_distance(current, nearest)
            total_cost += cost
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Добавляем возврат к началу
        if len(path) > 1:
            total_cost += self.get_distance(path[-1], path[0])
        
        return path, total_cost
    
    def _christofides_heuristic(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], int]:
        """Алгоритм Кристофида (упрощенная версия)"""
        # Для упрощения используем ближайшего соседа с улучшением
        path, cost = self._nearest_neighbor_heuristic(order_ids, start_id)
        
        # Применяем 3-opt улучшение
        improved_path, improved_cost = self._three_opt_improvement(path, cost)
        
        return improved_path, improved_cost
    
    def _genetic_heuristic(self, order_ids: List[int], start_id: int = None) -> Tuple[List[int], int]:
        """Простая генетическая эвристика с подробным логированием"""
        if len(order_ids) < 3:
            return self._nearest_neighbor_heuristic(order_ids, start_id)
        
        logger.info(f"🧬 Запуск генетического алгоритма для полигона с {len(order_ids)} точками")
        
        # Параметры генетического алгоритма
        population_size = 20
        generations = 50
        mutation_rate = 0.1
        tournament_size = 3
        
        logger.info(f"📊 Параметры GA: популяция={population_size}, поколения={generations}, мутация={mutation_rate}")
        
        # Создаем начальную популяцию
        population = []
        initial_costs = []
        
        for i in range(population_size):
            path = order_ids.copy()
            np.random.shuffle(path)
            population.append(path)
            cost = self._calculate_path_cost(path)
            initial_costs.append(cost)
        
        best_initial = min(initial_costs)
        worst_initial = max(initial_costs)
        avg_initial = sum(initial_costs) / len(initial_costs)
        
        logger.info(f"🎯 Начальная популяция: лучший={best_initial}, худший={worst_initial}, средний={avg_initial:.0f}")
        
        # Отслеживание лучшего решения
        global_best_path = min(population, key=lambda p: self._calculate_path_cost(p))
        global_best_cost = self._calculate_path_cost(global_best_path)
        
        # Эволюция
        improvement_history = []
        diversity_history = []
        
        print(f"🧬 Начинаю эволюцию: {generations} поколений")
        
        for generation in range(generations):
            # Оценка приспособленности
            fitness = []
            costs = []
            for path in population:
                cost = self._calculate_path_cost(path)
                costs.append(cost)
                fitness.append(1.0 / (cost + 1))  # Обратная приспособленность
            
            # Статистика поколения
            best_cost = min(costs)
            worst_cost = max(costs)
            avg_cost = sum(costs) / len(costs)
            
            # Отслеживаем улучшения
            if best_cost < global_best_cost:
                global_best_cost = best_cost
                global_best_path = population[costs.index(best_cost)]
                improvement = True
            else:
                improvement = False
            
            # Вычисляем разнообразие популяции
            diversity = len(set(tuple(p) for p in population)) / len(population)
            
            improvement_history.append(improvement)
            diversity_history.append(diversity)
            
            # Логируем каждые 10 поколений или при улучшении
            if generation % 10 == 0 or improvement:
                print(f"🔄 Поколение {generation:2d}: лучший={best_cost:6d}, худший={worst_cost:6d}, "
                      f"средний={avg_cost:6.0f}, разнообразие={diversity:.2f} "
                      f"{'✨ УЛУЧШЕНИЕ' if improvement else ''}")
                logger.info(f"🔄 Поколение {generation:2d}: лучший={best_cost:6d}, худший={worst_cost:6d}, "
                          f"средний={avg_cost:6.0f}, разнообразие={diversity:.2f} "
                          f"{'✨ УЛУЧШЕНИЕ' if improvement else ''}")
            
            # Селекция и скрещивание
            new_population = []
            for _ in range(population_size // 2):
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.extend([child1, child2])
            
            # Мутация
            mutations_count = 0
            for i in range(len(new_population)):
                if np.random.random() < mutation_rate:
                    new_population[i] = self._mutate(new_population[i])
                    mutations_count += 1
            
            if generation % 10 == 0:
                print(f"   Мутации: {mutations_count}/{len(new_population)} ({mutations_count/len(new_population)*100:.1f}%)")
                logger.debug(f"   Мутации: {mutations_count}/{len(new_population)} ({mutations_count/len(new_population)*100:.1f}%)")
            
            population = new_population
        
        # Финальная статистика
        final_costs = [self._calculate_path_cost(p) for p in population]
        final_best = min(final_costs)
        final_avg = sum(final_costs) / len(final_costs)
        
        total_improvements = sum(improvement_history)
        avg_diversity = sum(diversity_history) / len(diversity_history)
        
        print(f"🏆 Генетический алгоритм завершен:")
        print(f"   Начальная стоимость: {best_initial}")
        print(f"   Финальная стоимость: {final_best}")
        print(f"   Улучшение: {best_initial - final_best} ({((best_initial - final_best) / best_initial * 100):.1f}%)")
        print(f"   Улучшений за эволюцию: {total_improvements}/{generations}")
        print(f"   Среднее разнообразие: {avg_diversity:.2f}")
        
        logger.info(f"🏆 Генетический алгоритм завершен:")
        logger.info(f"   Начальная стоимость: {best_initial}")
        logger.info(f"   Финальная стоимость: {final_best}")
        logger.info(f"   Улучшение: {best_initial - final_best} ({((best_initial - final_best) / best_initial * 100):.1f}%)")
        logger.info(f"   Улучшений за эволюцию: {total_improvements}/{generations}")
        logger.info(f"   Среднее разнообразие: {avg_diversity:.2f}")
        
        return global_best_path, global_best_cost
    
    def _calculate_path_cost(self, path: List[int]) -> int:
        """Вычисление стоимости маршрута"""
        if len(path) < 2:
            return 0
        
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += self.get_distance(path[i], path[i + 1])
        
        # Добавляем возврат к началу
        total_cost += self.get_distance(path[-1], path[0])
        
        return total_cost
    
    def _tournament_selection(self, population: List[List[int]], fitness: List[float]) -> List[int]:
        """Турнирная селекция"""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Скрещивание (Order Crossover)"""
        n = len(parent1)
        start, end = sorted(np.random.choice(n, 2, replace=False))
        
        # Создаем ребенка 1
        child1 = [-1] * n
        child1[start:end] = parent1[start:end]
        
        remaining = [x for x in parent2 if x not in child1[start:end]]
        j = 0
        for i in range(n):
            if child1[i] == -1:
                child1[i] = remaining[j]
                j += 1
        
        # Создаем ребенка 2
        child2 = [-1] * n
        child2[start:end] = parent2[start:end]
        
        remaining = [x for x in parent1 if x not in child2[start:end]]
        j = 0
        for i in range(n):
            if child2[i] == -1:
                child2[i] = remaining[j]
                j += 1
        
        return child1, child2
    
    def _mutate(self, path: List[int]) -> List[int]:
        """Мутация (swap mutation)"""
        mutated = path.copy()
        if len(mutated) >= 2:
            i, j = np.random.choice(len(mutated), 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def _three_opt_improvement(self, path: List[int], initial_cost: int) -> Tuple[List[int], int]:
        """3-opt улучшение маршрута"""
        if len(path) < 4:
            return path, initial_cost
        
        best_path = path.copy()
        best_cost = initial_cost
        improved = True
        
        while improved:
            improved = False
            
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    for k in range(j + 1, len(path)):
                        # Пробуем 3-opt перестановку
                        new_path = best_path[:i] + best_path[j:k] + best_path[i:j] + best_path[k:]
                        new_cost = self._calculate_path_cost(new_path)
                        
                        if new_cost < best_cost:
                            best_path = new_path
                            best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return best_path, best_cost
    
    def calculate_polygon_cost(self, order_ids: List[int], service_times: Dict[int, int]) -> Dict:
        """
        Вычисление стоимости обслуживания полигона
        
        Args:
            order_ids: Список ID заказов в полигоне
            service_times: Словарь {order_id: service_time}
        
        Returns:
            Словарь с информацией о полигоне
        """
        if not order_ids:
            return {
                'optimal_route': [],
                'total_distance': 0,
                'total_service_time': 0,
                'total_cost': 0,
                'portal_id': None
            }
        
        # Решаем TSP
        optimal_route, total_distance = self.solve_tsp_dynamic(order_ids)
        

        
        # Вычисляем сервисное время
        total_service_time = sum(service_times.get(order_id, 0) for order_id in order_ids)
        
        # Общая стоимость
        total_cost = total_distance + total_service_time
        
        # Логируем для отладки
        if total_distance > 50000:  # Если расстояние подозрительно большое
            logger.warning(f"Большое расстояние для полигона: {total_distance} сек ({total_distance/3600:.1f} ч)")
            logger.warning(f"   Количество заказов: {len(order_ids)}")
            logger.warning(f"   Сервисное время: {total_service_time} сек")
            logger.warning(f"   Общая стоимость: {total_cost} сек ({total_cost/3600:.1f} ч)")
        
        # Определяем портал (первая точка маршрута)
        portal_id = optimal_route[0] if optimal_route else None
        
        return {
            'optimal_route': optimal_route,
            'total_distance': total_distance,
            'total_service_time': total_service_time,
            'total_cost': total_cost,
            'portal_id': portal_id
        }

def optimize_all_polygons(polygon_stats: pl.DataFrame, conn: sqlite3.Connection, 
                         service_times: Dict[int, Dict[int, int]], max_workers: int = 4) -> pl.DataFrame:
    """
    Оптимизация всех полигонов с параллельной обработкой
    
    Args:
        polygon_stats: DataFrame с информацией о полигонах
        conn: Подключение к базе данных
        service_times: Словарь {courier_id: {order_id: service_time}}
        max_workers: Максимальное количество потоков
    
    Returns:
        DataFrame с оптимизированными полигонами
    """
    logger.info(f"Начинаем оптимизацию полигонов (параллельно, {max_workers} потоков)")
    
    # Создаем копию соединения для каждого потока
    def create_connection():
        return sqlite3.connect(conn.execute("PRAGMA database_list").fetchone()[2])
    
    def optimize_single_polygon(polygon_data):
        """Оптимизация одного полигона"""
        mp_id = polygon_data['MpId']
        order_ids = polygon_data['order_ids']
        
        # Создаем отдельное соединение для потока
        thread_conn = create_connection()
        solver = PolygonTSPSolver(thread_conn, max_exact_size=15, use_parallel=False)
        
        try:
            # Используем сервисные времена для конкретного полигона
            # service_times имеет структуру {courier_id: {mp_id: service_time}}
            mp_id = polygon_data['MpId']
            polygon_service_times = {}
            
            # УБИРАЕМ ЗАГЛУШКИ! Оптимизируем только по расстояниям TSP
            # Сервисные времена = 0 для этапа оптимизации полигонов
            mp_service_time = 0
            
            # Присваиваем одинаковое сервисное время всем заказам в полигоне
            for order_id in order_ids:
                polygon_service_times[order_id] = mp_service_time
            
            # Оптимизируем полигон
            polygon_info = solver.calculate_polygon_cost(order_ids, polygon_service_times)
            
            result = {
                'MpId': mp_id,
                'order_count': polygon_data['order_count'],
                'order_ids': order_ids,
                'optimal_route': polygon_info['optimal_route'],
                'total_distance': polygon_info['total_distance'],
                'total_service_time': polygon_info['total_service_time'],
                'total_cost': polygon_info['total_cost'],
                'portal_id': polygon_info['portal_id']
            }
            
            logger.debug(f"Оптимизирован полигон {mp_id} ({len(order_ids)} заказов)")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации полигона {mp_id}: {e}")
            # Возвращаем базовую информацию в случае ошибки
            # УБИРАЕМ ЗАГЛУШКИ! Сервисные времена = 0 для этапа оптимизации полигонов
            mp_service_time = 0
            
            return {
                'MpId': mp_id,
                'order_count': polygon_data['order_count'],
                'order_ids': order_ids,
                'optimal_route': order_ids,
                'total_distance': 0,
                'total_service_time': len(order_ids) * mp_service_time,
                'total_cost': len(order_ids) * mp_service_time,
                'portal_id': order_ids[0] if order_ids else None
            }
        finally:
            thread_conn.close()
    
    # Подготавливаем данные для параллельной обработки
    polygon_data_list = [row for row in polygon_stats.iter_rows(named=True)]
    
    optimized_polygons = []
    
    # Параллельная обработка
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем задачи
        future_to_polygon = {
            executor.submit(optimize_single_polygon, polygon_data): polygon_data['MpId'] 
            for polygon_data in polygon_data_list
        }
        
        # Собираем результаты
        for future in as_completed(future_to_polygon):
            polygon_id = future_to_polygon[future]
            try:
                result = future.result()
                optimized_polygons.append(result)
            except Exception as e:
                logger.error(f"Ошибка получения результата для полигона {polygon_id}: {e}")
    
    result_df = pl.DataFrame(optimized_polygons)
    logger.info(f"Оптимизировано {len(result_df)} полигонов")
    
    return result_df

def solve_tsp_worker(args):
    """
    Worker функция для multiprocessing - решает TSP для одного полигона
    Обходит GIL для CPU-интенсивных операций
    """
    polygon_data, db_path, max_exact_size, avg_service_time = args
    
    try:
        # Создаем отдельное соединение для процесса
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=1000000")
        
        mp_id = polygon_data['MpId']
        order_ids = polygon_data['order_ids']
        
        logger.info(f"Процесс: Начинаю оптимизацию полигона {mp_id} ({len(order_ids)} заказов)")
        
        # Упрощенная эвристика вместо сложного TSP
        if len(order_ids) <= 2:
            optimal_route = order_ids
            total_distance = 0
        else:
            # Простая эвристика: ближайший сосед
            optimal_route = [order_ids[0]]
            remaining = order_ids[1:]
            total_distance = 0
            
            cursor = conn.cursor()
            current = order_ids[0]
            
            while remaining:
                # Находим ближайшего соседа
                min_dist = float('inf')
                nearest = None
                
                for next_point in remaining:
                    cursor.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (current, next_point))
                    result = cursor.fetchone()
                    if result and result[0] is not None and result[0] > 0:
                        dist = result[0]
                    else:
                        dist = float('inf')  # Если расстояние не найдено, пропускаем
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest = next_point
                
                if nearest:
                    optimal_route.append(nearest)
                    total_distance += min_dist
                    remaining.remove(nearest)
                    current = nearest
                else:
                    break
        
        # Вычисляем общую стоимость
        total_service_time = len(order_ids) * avg_service_time
        total_cost = total_distance + total_service_time
        
        polygon_info = {
            'optimal_route': optimal_route,
            'total_distance': total_distance,
            'total_service_time': total_service_time,
            'total_cost': total_cost,
            'portal_id': order_ids[0] if order_ids else None
        }
        
        logger.info(f"Процесс: Полигон {mp_id} оптимизирован (стоимость: {polygon_info['total_cost']})")
        
        result = {
            'MpId': mp_id,
            'order_count': polygon_data['order_count'],
            'order_ids': order_ids,
            'optimal_route': polygon_info['optimal_route'],
            'total_distance': polygon_info['total_distance'],
            'total_service_time': polygon_info['total_service_time'],
            'total_cost': polygon_info['total_cost'],
            'portal_id': polygon_info['portal_id']
        }
        
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Ошибка в worker процессе для полигона {polygon_data.get('MpId', 'unknown')}: {e}")
        # Возвращаем базовую информацию в случае ошибки
        return {
            'MpId': polygon_data.get('MpId', 0),
            'order_count': polygon_data.get('order_count', 0),
            'order_ids': polygon_data.get('order_ids', []),
            'optimal_route': polygon_data.get('order_ids', []),
            'total_distance': 0,
            'total_service_time': len(polygon_data.get('order_ids', [])) * avg_service_time,
            'total_cost': len(polygon_data.get('order_ids', [])) * avg_service_time,
            'portal_id': polygon_data.get('order_ids', [None])[0] if polygon_data.get('order_ids') else None
        }

def optimize_all_polygons_mp(polygon_stats: pl.DataFrame, db_path: str, 
                           service_times: Dict[int, Dict[int, int]], 
                           max_workers: int = None, 
                           max_exact_size: int = 15) -> pl.DataFrame:
    """
    Оптимизация всех полигонов с использованием multiprocessing
    Обходит GIL для CPU-интенсивных TSP операций
    
    Args:
        polygon_stats: DataFrame с информацией о полигонах
        db_path: Путь к базе данных SQLite
        service_times: Словарь {courier_id: {order_id: service_time}}
        max_workers: Максимальное количество процессов (по умолчанию cpu_count())
        max_exact_size: Максимальный размер для точного решения TSP
    
    Returns:
        DataFrame с оптимизированными полигонами
    """
    if max_workers is None:
        max_workers = min(cpu_count(), len(polygon_stats))
    
    logger.info(f"Начинаем оптимизацию полигонов (multiprocessing, {max_workers} процессов)")
    start_time = time.time()
    
    # Подготавливаем данные для multiprocessing
    polygon_data_list = [row for row in polygon_stats.iter_rows(named=True)]
    
    # УБИРАЕМ ЗАГЛУШКИ! Оптимизируем только по расстояниям TSP
    # Сервисные времена = 0 для этапа оптимизации полигонов
    avg_service_time = 0
    
    # Создаем аргументы для worker процессов
    worker_args = [(polygon_data, db_path, max_exact_size, avg_service_time) for polygon_data in polygon_data_list]
    
    optimized_polygons = []
    
    # Используем multiprocessing для CPU-интенсивных операций
    with Pool(processes=max_workers) as pool:
        logger.info(f"Запускаю {max_workers} процессов для обработки {len(worker_args)} полигонов")
        
        # Запускаем задачи параллельно
        results = pool.map(solve_tsp_worker, worker_args)
        
        logger.info(f"Собираю результаты от {len(results)} процессов")
        
        # Собираем результаты
        processed_count = 0
        for result in results:
            if result is not None:
                optimized_polygons.append(result)
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Обработано {processed_count}/{len(results)} полигонов")
        
        logger.info(f"Успешно обработано {processed_count} полигонов из {len(results)}")
    
    end_time = time.time()
    logger.info(f"Оптимизировано {len(optimized_polygons)} полигонов за {end_time - start_time:.2f} секунд")
    
    result_df = pl.DataFrame(optimized_polygons)
    return result_df

def optimize_all_polygons_hybrid(polygon_stats: pl.DataFrame, db_path: str,
                                service_times: Dict[int, Dict[int, int]], 
                                max_workers: int = None,
                                max_exact_size: int = 15) -> pl.DataFrame:
    """
    Гибридная оптимизация: multiprocessing для больших полигонов, threading для маленьких
    """
    if max_workers is None:
        max_workers = min(cpu_count(), len(polygon_stats))
    
    logger.info(f"Гибридная оптимизация полигонов (max_workers={max_workers})")
    
    # Разделяем полигоны по размеру
    small_polygons = []
    large_polygons = []
    
    for row in polygon_stats.iter_rows(named=True):
        if row['order_count'] <= 8:  # Маленькие полигоны - threading
            small_polygons.append(row)
        else:  # Большие полигоны - multiprocessing
            large_polygons.append(row)
    
    logger.info(f"Маленьких полигонов: {len(small_polygons)}, больших: {len(large_polygons)}")
    
    optimized_polygons = []
    
    # Обрабатываем большие полигоны через multiprocessing
    if large_polygons:
        logger.info(f"Начинаю multiprocessing для {len(large_polygons)} больших полигонов")
        large_df = pl.DataFrame(large_polygons)
        large_results = optimize_all_polygons_mp(large_df, db_path, service_times, 
                                           max_workers, max_exact_size)
        optimized_polygons.extend(large_results.iter_rows(named=True))
        logger.info(f"Multiprocessing завершен: {len(large_results)} полигонов обработано")
    
    # Обрабатываем маленькие полигоны через multiprocessing (отключаем threading из-за проблем с SQLite)
    if small_polygons:
        logger.info(f"Начинаю multiprocessing для {len(small_polygons)} маленьких полигонов")
        small_df = pl.DataFrame(small_polygons)
        small_results = optimize_all_polygons_mp(small_df, db_path, service_times, 
                                           max_workers, max_exact_size)
        optimized_polygons.extend(small_results.iter_rows(named=True))
        logger.info(f"Multiprocessing для маленьких полигонов завершен: {len(small_results)} полигонов обработано")
    
    result_df = pl.DataFrame(optimized_polygons)
    logger.info(f"Гибридная оптимизация завершена: {len(result_df)} полигонов")
    
    return result_df
