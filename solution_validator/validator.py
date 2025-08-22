#!/usr/bin/env python3
"""
Валидатор решения VRP задачи
Проверяет корректность выходного файла solution.json и вычисляет целевую функцию
"""

import json
import sqlite3
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import polars as pl
from dataclasses import dataclass

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Результат валидации"""
    is_valid: bool
    total_cost: float
    total_penalty: int
    unassigned_orders: int
    violations: List[str]
    courier_stats: Dict[int, Dict]
    execution_time: float

class SolutionValidator:
    """Валидатор решения VRP задачи"""
    
    def __init__(self, orders_path: str, couriers_path: str, durations_db_path: str):
        self.orders_path = Path(orders_path)
        self.couriers_path = Path(couriers_path)
        self.durations_db_path = Path(durations_db_path)
        
        # Данные
        self.orders_data = None
        self.couriers_data = None
        self.order_to_mp = {}  # order_id -> MpId
        self.mp_to_orders = {}  # MpId -> [order_ids]
        self.courier_service_times = {}  # courier_id -> {MpId -> service_time}
        self.warehouse_id = 1
        
        # Ограничения
        self.max_work_time = 12 * 3600  # 12 часов в секундах
        self.penalty_unassigned = 3000  # штраф за неназначенный заказ
        self.penalty_exceed_12h = 1000000  # большой штраф за превышение 12 часов
        
    def load_data(self):
        """Загружает входные данные"""
        logger.info("Загружаем данные...")
        
        # Загружаем заказы
        with open(self.orders_path, 'r') as f:
            orders_data = json.load(f)
        self.orders_data = orders_data['Orders']
        
        # Создаем маппинги
        for order in self.orders_data:
            order_id = order['ID']
            mp_id = order['MpId']
            self.order_to_mp[order_id] = mp_id
            
            if mp_id not in self.mp_to_orders:
                self.mp_to_orders[mp_id] = []
            self.mp_to_orders[mp_id].append(order_id)
        
        # Загружаем курьеров
        with open(self.couriers_path, 'r') as f:
            couriers_data = json.load(f)
        self.couriers_data = couriers_data
        
        # Создаем маппинг сервис-таймов
        for courier in couriers_data['Couriers']:
            courier_id = courier['ID']
            self.courier_service_times[courier_id] = {}
            for service_time_info in courier['ServiceTimeInMps']:
                mp_id = service_time_info['MpID']
                service_time = service_time_info['ServiceTime']
                self.courier_service_times[courier_id][mp_id] = service_time
        
        logger.info(f"Загружено: {len(self.orders_data)} заказов, {len(couriers_data['Couriers'])} курьеров")
    
    def get_distance(self, from_id: int, to_id: int) -> int:
        """Получает расстояние между двумя точками из БД"""
        conn = sqlite3.connect(self.durations_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT d FROM dists WHERE f = ? AND t = ?", (from_id, to_id))
        result = cursor.fetchone()
        
        conn.close()
        
        if result is None:
            logger.warning(f"Не найдено расстояние от {from_id} до {to_id}")
            return 0
        
        return result[0]
    
    def validate_route(self, courier_id: int, route: List[int]) -> Tuple[float, List[str], Dict]:
        """Валидирует маршрут курьера и возвращает стоимость и нарушения"""
        violations = []
        total_travel_time = 0
        total_service_time = 0
        visited_orders = set()
        visited_mps = set()
        
        # Проверяем, что маршрут начинается и заканчивается на складе
        if route[0] != 0 or route[-1] != 0:
            violations.append(f"Маршрут должен начинаться и заканчиваться на складе (0), получено: {route[0]} -> {route[-1]}")
        
        # Вычисляем время маршрута
        for i in range(len(route) - 1):
            from_id = route[i]
            to_id = route[i + 1]
            
            # Добавляем время перемещения
            travel_time = self.get_distance(from_id, to_id)
            total_travel_time += travel_time
            
            # Если это не склад, добавляем сервис-тайм
            if to_id != 0:
                if to_id in self.order_to_mp:
                    mp_id = self.order_to_mp[to_id]
                    visited_orders.add(to_id)
                    visited_mps.add(mp_id)
                    
                    # Получаем сервис-тайм для данного курьера и MpId
                    if courier_id in self.courier_service_times and mp_id in self.courier_service_times[courier_id]:
                        service_time = self.courier_service_times[courier_id][mp_id]
                        total_service_time += service_time
                    else:
                        violations.append(f"Не найден сервис-тайм для курьера {courier_id} и MpId {mp_id}")
                else:
                    violations.append(f"Заказ {to_id} не найден в данных")
        
        total_route_time = total_travel_time + total_service_time
        
        # Проверяем ограничение 12 часов
        if total_route_time > self.max_work_time:
            violations.append(f"Превышение лимита 12 часов: {total_route_time} > {self.max_work_time}")
        
        # Проверяем дубли заказов
        if len(visited_orders) != len(route) - 2:  # -2 для склада в начале и конце
            violations.append(f"Дубли заказов в маршруте: {len(visited_orders)} уникальных из {len(route) - 2}")
        
        stats = {
            'travel_time': total_travel_time,
            'service_time': total_service_time,
            'total_time': total_route_time,
            'orders_count': len(visited_orders),
            'mps_count': len(visited_mps),
            'is_feasible': total_route_time <= self.max_work_time
        }
        
        return total_route_time, violations, stats
    
    def validate_solution(self, solution_path: str) -> ValidationResult:
        """Валидирует полное решение"""
        import time
        start_time = time.time()
        
        logger.info(f"Валидируем решение: {solution_path}")
        
        # Загружаем решение
        with open(solution_path, 'r') as f:
            solution = json.load(f)
        
        if 'routes' not in solution:
            raise ValueError("Файл решения должен содержать ключ 'routes'")
        
        routes = solution['routes']
        violations = []
        total_cost = 0
        total_penalty = 0
        all_visited_orders = set()
        courier_stats = {}
        
        # Проверяем каждый маршрут
        for route_info in routes:
            if 'courier_id' not in route_info or 'route' not in route_info:
                violations.append("Каждый маршрут должен содержать 'courier_id' и 'route'")
                continue
            
            courier_id = route_info['courier_id']
            route = route_info['route']
            
            # Валидируем маршрут
            route_cost, route_violations, stats = self.validate_route(courier_id, route)
            
            # Добавляем нарушения
            violations.extend([f"Курьер {courier_id}: {v}" for v in route_violations])
            
            # Собираем посещенные заказы
            for order_id in route:
                if order_id != 0:  # не склад
                    all_visited_orders.add(order_id)
            
            # Добавляем стоимость
            if stats['is_feasible']:
                total_cost += route_cost
            else:
                total_cost += route_cost + self.penalty_exceed_12h
                total_penalty += 1
            
            courier_stats[courier_id] = stats
        
        # Проверяем неназначенные заказы
        all_orders = set(self.order_to_mp.keys())
        unassigned_orders = all_orders - all_visited_orders
        unassigned_penalty = len(unassigned_orders) * self.penalty_unassigned
        total_cost += unassigned_penalty
        total_penalty += len(unassigned_orders)
        
        if unassigned_orders:
            violations.append(f"Неназначенные заказы: {len(unassigned_orders)}")
        
        # Проверяем атомарность микрополигонов
        mp_assignments = {}  # MpId -> courier_id
        for route_info in routes:
            courier_id = route_info['courier_id']
            route = route_info['route']
            
            for order_id in route:
                if order_id != 0 and order_id in self.order_to_mp:
                    mp_id = self.order_to_mp[order_id]
                    if mp_id in mp_assignments and mp_assignments[mp_id] != courier_id:
                        violations.append(f"Нарушение атомарности: MpId {mp_id} назначен курьерам {mp_assignments[mp_id]} и {courier_id}")
                    mp_assignments[mp_id] = courier_id
        
        execution_time = time.time() - start_time
        
        result = ValidationResult(
            is_valid=len(violations) == 0,
            total_cost=total_cost,
            total_penalty=total_penalty,
            unassigned_orders=len(unassigned_orders),
            violations=violations,
            courier_stats=courier_stats,
            execution_time=execution_time
        )
        
        return result
    
    def print_report(self, result: ValidationResult):
        """Выводит отчет о валидации"""
        print("\n" + "="*60)
        print("ОТЧЕТ О ВАЛИДАЦИИ РЕШЕНИЯ")
        print("="*60)
        
        print(f"Время валидации: {result.execution_time:.2f} секунд")
        print(f"Допустимость: {'ДА' if result.is_valid else 'НЕТ'}")
        print(f"Общая стоимость: {result.total_cost:,.0f}")
        print(f"Общий штраф: {result.total_penalty}")
        print(f"Неназначенные заказы: {result.unassigned_orders}")
        
        if result.violations:
            print(f"\nНарушения ({len(result.violations)}):")
            for i, violation in enumerate(result.violations[:10], 1):  # Показываем первые 10
                print(f"  {i}. {violation}")
            if len(result.violations) > 10:
                print(f"  ... и еще {len(result.violations) - 10} нарушений")
        
        # Статистика по курьерам
        if result.courier_stats:
            feasible_count = sum(1 for stats in result.courier_stats.values() if stats['is_feasible'])
            total_couriers = len(result.courier_stats)
            avg_time = sum(stats['total_time'] for stats in result.courier_stats.values()) / total_couriers
            
            print(f"\nСтатистика курьеров:")
            print(f"  Всего курьеров: {total_couriers}")
            print(f"  Допустимых маршрутов: {feasible_count}")
            print(f"  Среднее время маршрута: {avg_time:.0f} секунд")
            
            # Топ-5 самых длинных маршрутов
            sorted_couriers = sorted(result.courier_stats.items(), 
                                   key=lambda x: x[1]['total_time'], reverse=True)
            print(f"\nТоп-5 самых длинных маршрутов:")
            for i, (courier_id, stats) in enumerate(sorted_couriers[:5], 1):
                status = "✓" if stats['is_feasible'] else "✗"
                print(f"  {i}. Курьер {courier_id}: {stats['total_time']:.0f}s "
                      f"({stats['travel_time']:.0f}+{stats['service_time']:.0f}) {status}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Валидатор решения VRP задачи')
    parser.add_argument('--solution', required=True, help='Путь к файлу решения solution.json')
    parser.add_argument('--orders', required=True, help='Путь к файлу заказов')
    parser.add_argument('--couriers', required=True, help='Путь к файлу курьеров')
    parser.add_argument('--durations_db', required=True, help='Путь к SQLite базе расстояний')
    
    args = parser.parse_args()
    
    # Проверяем существование файлов
    for file_path in [args.solution, args.orders, args.couriers, args.durations_db]:
        if not Path(file_path).exists():
            logger.error(f"Файл не найден: {file_path}")
            return 1
    
    try:
        # Создаем валидатор
        validator = SolutionValidator(args.orders, args.couriers, args.durations_db)
        validator.load_data()
        
        # Валидируем решение
        result = validator.validate_solution(args.solution)
        
        # Выводим отчет
        validator.print_report(result)
        
        return 0 if result.is_valid else 1
        
    except Exception as e:
        logger.error(f"Ошибка валидации: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
