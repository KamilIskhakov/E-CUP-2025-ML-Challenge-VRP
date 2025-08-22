#!/usr/bin/env python3
"""
Simulated Annealing решатель для VRP задачи
"""

import json
import time
import logging
import random
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import polars as pl
import numpy as np
import psutil

# Импорты из других модулей
from worker import RouteWorker, RouteEvaluation

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Solution:
    """Решение VRP задачи"""
    assignment: Dict[int, int]  # mp_id -> courier_id
    routes: Dict[int, List[int]]  # courier_id -> [mp_ids]
    total_cost: int = 0
    penalty_count: int = 0
    feasible: bool = True
    evaluations: Dict[int, RouteEvaluation] = field(default_factory=dict)

@dataclass
class Move:
    """Операция перемещения в SA"""
    move_type: str  # 'relocate', 'swap', 'block_relocate'
    mp_id: int
    from_courier: int
    to_courier: int
    additional_data: Dict = field(default_factory=dict)

class SASolver:
    """Simulated Annealing решатель для VRP"""
    
    def __init__(self, data_dir: str, durations_db_path: str, max_workers: int = 4, time_budget: int = 2400):
        self.data_dir = Path(data_dir)
        self.durations_db_path = durations_db_path
        self.max_workers = max_workers
        self.time_budget = time_budget  # в секундах
        
        # Параметры SA
        self.T0 = 1.0  # начальная температура
        self.alpha = 0.9996  # коэффициент охлаждения
        self.penalty_unassigned = 3000  # штраф за неназначенный заказ
        self.penalty_exceed_12h = 1e8  # штраф за превышение 12 часов
        
        # Данные
        self.mp_data = {}
        self.couriers = []
        self.service_times = {}
        self.total_orders = 0
        self.orders_df = None
        
        # Статистика
        self.iterations = 0
        self.accepted_moves = 0
        self.infeasible_count = 0
        
        # Загружаем данные
        self._load_data()
    
    def _load_data(self):
        """Загрузка данных"""
        logger.info("Загружаем данные...")
        
        # Загружаем микрополигоны
        mp_metadata_path = self.data_dir / "mp_metadata.json"
        with open(mp_metadata_path, 'r') as f:
            mp_metadata = json.load(f)
        
        for mp_info in mp_metadata:
            self.mp_data[mp_info['mp_id']] = mp_info
            self.total_orders += mp_info['size']
        
        # Загружаем курьеров
        couriers_path = self.data_dir / "couriers_filtered.json"
        couriers_df = pl.read_json(couriers_path)
        self.couriers = couriers_df['ID'].to_list()
        
        # Загружаем заказы
        orders_path = self.data_dir / "orders_filtered.json"
        self.orders_df = pl.read_json(orders_path)
        
        # Загружаем сервисные времена
        service_times_path = self.data_dir / "service_times_filtered.json"
        with open(service_times_path, 'r') as f:
            service_times_data = json.load(f)
        
        for record in service_times_data:
            courier_id = record['courier_id']
            mp_id = record['mp_id']
            service_time = record['service_time']
            
            if courier_id not in self.service_times:
                self.service_times[courier_id] = {}
            self.service_times[courier_id][mp_id] = service_time
        
        logger.info(f"Загружено: {len(self.mp_data)} микрополигонов, {len(self.couriers)} курьеров, {self.total_orders} заказов")
    
    def solve(self) -> Solution:
        """Основной метод решения"""
        logger.info("Начинаем SA решение...")
        start_time = time.time()
        
        # Создаем начальное решение
        current_solution = self._create_initial_solution()
        best_solution = self._deep_copy_solution(current_solution)
        
        # Создаем один воркер для всех оценок
        worker = RouteWorker(str(self.data_dir), self.durations_db_path)
        
        try:
            # Оцениваем начальное решение
            current_solution = self._evaluate_solution_with_worker(current_solution, worker)
            best_solution = self._deep_copy_solution(current_solution)
            
            logger.info(f"Начальное решение: стоимость={current_solution.total_cost}, штрафы={current_solution.penalty_count}")
            
            # Основной цикл SA
            T = self.T0
            iteration = 0
            
            while time.time() - start_time < self.time_budget:
                # Генерируем ход
                move = self._generate_move(current_solution)
                
                # Применяем ход
                candidate = self._apply_move(current_solution, move)
                
                # Оцениваем кандидата
                candidate = self._evaluate_solution_with_worker(candidate, worker)
                
                # Вычисляем разность стоимостей
                delta = candidate.total_cost - current_solution.total_cost
                
                # Принимаем или отклоняем
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_solution = candidate
                    self.accepted_moves += 1
                    
                    if candidate.total_cost < best_solution.total_cost:
                        best_solution = self._deep_copy_solution(candidate)
                        logger.info(f"Новое лучшее решение: стоимость={best_solution.total_cost}, итерация={iteration}")
                
                # Охлаждение
                T *= self.alpha
                iteration += 1
                
                # Логирование прогресса
                if iteration % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Итерация {iteration}, T={T:.6f}, текущая стоимость={current_solution.total_cost}, "
                              f"лучшая={best_solution.total_cost}, принято={self.accepted_moves}, "
                              f"время={elapsed:.1f}s")
                
                # Проверяем ограничение по времени
                if time.time() - start_time >= self.time_budget:
                    break
        
        finally:
            worker.close()
    
        self.iterations = iteration
        elapsed = time.time() - start_time
        
        logger.info(f"SA завершен: {iteration} итераций за {elapsed:.1f} секунд")
        logger.info(f"Финальное решение: стоимость={best_solution.total_cost}, штрафы={best_solution.penalty_count}")
        
        return best_solution
    
    def _create_initial_solution(self) -> Solution:
        """Создание начального решения жадным алгоритмом"""
        logger.info("Создаем начальное решение...")
        
        assignment = {}
        routes = {courier_id: [] for courier_id in self.couriers}
        
        # Сортируем микрополигоны по размеру (от больших к маленьким)
        sorted_mps = sorted(self.mp_data.items(), key=lambda x: x[1]['size'], reverse=True)
        
        for mp_id, mp_info in sorted_mps:
            # Находим лучшего курьера для микрополигона
            best_courier = self._find_best_courier_for_mp(mp_id, routes)
            
            if best_courier is not None:
                assignment[mp_id] = best_courier
                routes[best_courier].append(mp_id)
            else:
                # Не можем назначить - оставляем неназначенным (штраф)
                pass
        
        return Solution(assignment=assignment, routes=routes)
    
    def _find_best_courier_for_mp(self, mp_id: int, routes: Dict[int, List[int]]) -> Optional[int]:
        """Находит лучшего курьера для микрополигона"""
        mp_info = self.mp_data[mp_id]
        mp_size = mp_info['size']
        
        best_courier = None
        best_score = float('inf')
        
        for courier_id in self.couriers:
            # Оцениваем, насколько хорошо курьер подходит для микрополигона
            current_route = routes[courier_id]
            
            # Простая оценка: учитываем размер маршрута и сервисное время
            route_size = len(current_route)
            service_time = self._get_service_time(courier_id, mp_id)
            
            # Предпочитаем курьеров с меньшими маршрутами
            score = route_size * 100 + service_time
            
            if score < best_score:
                best_score = score
                best_courier = courier_id
        
        return best_courier
    
    def _generate_move(self, solution: Solution) -> Move:
        """Генерирует случайный ход"""
        move_types = ['relocate', 'swap']
        move_type = random.choice(move_types)
        
        if move_type == 'relocate':
            # Перемещение одного микрополигона
            assigned_mps = list(solution.assignment.keys())
            if assigned_mps:
                mp_id = random.choice(assigned_mps)
                from_courier = solution.assignment[mp_id]
                
                # Выбираем случайного курьера (может быть тот же)
                to_courier = random.choice(self.couriers)
                
                return Move(
                    move_type='relocate',
                    mp_id=mp_id,
                    from_courier=from_courier,
                    to_courier=to_courier
                )
        
        elif move_type == 'swap':
            # Обмен микрополигонами между курьерами
            assigned_mps = list(solution.assignment.keys())
            if len(assigned_mps) >= 2:
                mp1, mp2 = random.sample(assigned_mps, 2)
                courier1 = solution.assignment[mp1]
                courier2 = solution.assignment[mp2]
                
                return Move(
                    move_type='swap',
                    mp_id=mp1,
                    from_courier=courier1,
                    to_courier=courier2,
                    additional_data={'mp2': mp2, 'courier2': courier2}
                )
        
        # Fallback
        return Move(move_type='relocate', mp_id=1, from_courier=1, to_courier=1)
    
    def _apply_move(self, solution: Solution, move: Move) -> Solution:
        """Применяет ход к решению"""
        new_assignment = solution.assignment.copy()
        new_routes = {courier_id: route.copy() for courier_id, route in solution.routes.items()}
        
        if move.move_type == 'relocate':
            # Удаляем из старого курьера
            if move.mp_id in new_assignment:
                old_courier = new_assignment[move.mp_id]
                if move.mp_id in new_routes[old_courier]:
                    new_routes[old_courier].remove(move.mp_id)
            
            # Добавляем к новому курьеру
            new_assignment[move.mp_id] = move.to_courier
            new_routes[move.to_courier].append(move.mp_id)
        
        elif move.move_type == 'swap':
            mp2 = move.additional_data['mp2']
            courier2 = move.additional_data['courier2']
            
            # Обмениваем микрополигоны
            new_assignment[move.mp_id] = courier2
            new_assignment[mp2] = move.from_courier
            
            # Обновляем маршруты
            if move.mp_id in new_routes[move.from_courier]:
                new_routes[move.from_courier].remove(move.mp_id)
            if mp2 in new_routes[courier2]:
                new_routes[courier2].remove(mp2)
            
            new_routes[courier2].append(move.mp_id)
            new_routes[move.from_courier].append(mp2)
        
        return Solution(assignment=new_assignment, routes=new_routes)
    
    def _evaluate_solution(self, solution: Solution) -> Solution:
        """Оценка решения (последовательная версия)"""
        total_cost = 0
        penalty_count = 0
        evaluations = {}
        
        # Оцениваем каждого курьера
        for courier_id, mp_sequence in solution.routes.items():
            if mp_sequence:
                # Создаем временного воркера для оценки
                worker = RouteWorker(str(self.data_dir), self.durations_db_path)
                evaluation = worker.evaluate_courier_route(courier_id, mp_sequence)
                worker.close()
                
                evaluations[courier_id] = evaluation
                total_cost += evaluation.route_time
                
                if not evaluation.feasible:
                    penalty_count += 1
                    total_cost += self.penalty_exceed_12h
        
        # Штраф за неназначенные заказы
        assigned_orders = sum(len(self.mp_data[mp_id]['order_ids']) 
                            for mp_id in solution.assignment.keys())
        unassigned_orders = self.total_orders - assigned_orders
        penalty_count += unassigned_orders
        total_cost += unassigned_orders * self.penalty_unassigned
        
        solution.total_cost = total_cost
        solution.penalty_count = penalty_count
        solution.evaluations = evaluations
        solution.feasible = penalty_count == 0
        
        return solution
    
    def _evaluate_solution_with_worker(self, solution: Solution, worker: RouteWorker) -> Solution:
        """Оценка решения с использованием одного воркера"""
        total_cost = 0
        penalty_count = 0
        evaluations = {}
        
        # Подготавливаем задачи для оценки
        evaluation_tasks = []
        for courier_id, mp_sequence in solution.routes.items():
            if mp_sequence:
                evaluation_tasks.append((courier_id, mp_sequence))
        
        # Выполняем оценку с одним воркером
        for courier_id, mp_sequence in evaluation_tasks:
            try:
                evaluation = worker.evaluate_courier_route(courier_id, mp_sequence)
                evaluations[courier_id] = evaluation
                total_cost += evaluation.route_time
                
                if not evaluation.feasible:
                    penalty_count += 1
                    total_cost += self.penalty_exceed_12h
            except Exception as e:
                logger.error(f"Ошибка при оценке курьера {courier_id}: {e}")
                # Создаем фиктивную оценку с большим штрафом
                evaluations[courier_id] = RouteEvaluation(
                    courier_id=courier_id,
                    route_time=0,
                    travel_time=0,
                    service_time=0,
                    feasible=False,
                    portal_pairs=[],
                    error_message=str(e)
                )
                penalty_count += 1
                total_cost += self.penalty_exceed_12h
        
        # Штраф за неназначенные заказы
        assigned_orders = sum(len(self.mp_data[mp_id]['order_ids']) 
                            for mp_id in solution.assignment.keys())
        unassigned_orders = self.total_orders - assigned_orders
        penalty_count += unassigned_orders
        total_cost += unassigned_orders * self.penalty_unassigned
        
        solution.total_cost = total_cost
        solution.penalty_count = penalty_count
        solution.evaluations = evaluations
        solution.feasible = penalty_count == 0
        
        return solution
    
    def _evaluate_route_worker(self, worker: RouteWorker, courier_id: int, mp_sequence: List[int]) -> RouteEvaluation:
        """Оценка маршрута через воркер"""
        return worker.evaluate_courier_route(courier_id, mp_sequence)
    
    def _init_worker(self, data_dir: str, durations_db_path: str) -> RouteWorker:
        """Инициализация воркера"""
        return RouteWorker(data_dir, durations_db_path)
    
    def _get_service_time(self, courier_id: int, mp_id: int) -> int:
        """Получает сервисное время курьера для микрополигона"""
        if courier_id in self.service_times and mp_id in self.service_times[courier_id]:
            return self.service_times[courier_id][mp_id]
        return 300  # значение по умолчанию
    
    def _deep_copy_solution(self, solution: Solution) -> Solution:
        """Глубокое копирование решения"""
        return Solution(
            assignment=solution.assignment.copy(),
            routes={courier_id: route.copy() for courier_id, route in solution.routes.items()},
            total_cost=solution.total_cost,
            penalty_count=solution.penalty_count,
            feasible=solution.feasible,
            evaluations=solution.evaluations.copy()
        )
    
    def save_solution(self, solution: Solution, output_path: str):
        """Сохранение решения в формате solution.json"""
        logger.info(f"Сохраняем решение в {output_path}")
        
        # Подготавливаем маршруты в нужном формате
        routes_data = []
        
        for courier_id, mp_sequence in solution.routes.items():
            if mp_sequence:
                # Создаем маршрут в формате [0, order1, order2, ..., 0]
                route = [0]  # начинаем со склада
                
                # Добавляем заказы из микрополигонов
                for mp_id in mp_sequence:
                    mp_orders = self.mp_data[mp_id]['order_ids']
                    route.extend(mp_orders)
                
                route.append(0)  # заканчиваем на складе
                
                route_data = {
                    "courier_id": courier_id,
                    "route": route
                }
                routes_data.append(route_data)
        
        # Создаем итоговый JSON в нужном формате
        output = {
            "routes": routes_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Решение сохранено: {len(routes_data)} маршрутов")

def log_resource_usage():
    """Логирование использования ресурсов"""
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    
    logger.info(f"Использование ресурсов: CPU={cpu_percent:.1f}%, "
               f"RAM={memory_info.rss / 1024 / 1024:.1f}MB")

if __name__ == "__main__":
    # Пример использования
    solver = SASolver("data", "../durations.sqlite", max_workers=4, time_budget=60)  # 1 минута для теста
    
    log_resource_usage()
    solution = solver.solve()
    log_resource_usage()
    
    solver.save_solution(solution, "sa_solution.json")
    print(f"Решение сохранено: стоимость={solution.total_cost}, штрафы={solution.penalty_count}")
