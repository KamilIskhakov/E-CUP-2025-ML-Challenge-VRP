#!/usr/bin/env python3
"""
Главный модуль для запуска SA VRP решателя
"""

import argparse
import time
import logging
import psutil
import os
from pathlib import Path
from typing import Dict, Any

from prepare_data import DataPreprocessor
from sa_solver import SASolver, log_resource_usage

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sa_vrp_solver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Мониторинг использования ресурсов"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.peak_memory = self.start_memory
        self.cpu_samples = []
    
    def log_current_usage(self):
        """Логирует текущее использование ресурсов"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Память
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # CPU
        cpu_percent = self.process.cpu_percent()
        self.cpu_samples.append(cpu_percent)
        
        # Системные ресурсы
        system_memory = psutil.virtual_memory()
        system_cpu = psutil.cpu_percent(interval=1)
        
        logger.info(f"Ресурсы [{elapsed:.1f}s]: "
                   f"CPU={cpu_percent:.1f}% (система: {system_cpu:.1f}%), "
                   f"RAM={current_memory/1024/1024:.1f}MB (пик: {self.peak_memory/1024/1024:.1f}MB), "
                   f"Система RAM={system_memory.percent:.1f}%")
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку использования ресурсов"""
        elapsed = time.time() - self.start_time
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        
        return {
            'elapsed_time': elapsed,
            'peak_memory_mb': self.peak_memory / 1024 / 1024,
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max(self.cpu_samples) if self.cpu_samples else 0,
            'memory_growth_mb': (self.peak_memory - self.start_memory) / 1024 / 1024
        }

def check_data_files(data_dir: Path) -> bool:
    """Проверяет наличие необходимых файлов данных"""
    required_files = [
        "mp_metadata.json",
        "orders_filtered.json", 
        "couriers_filtered.json",
        "service_times_filtered.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (data_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"Отсутствуют файлы данных: {missing_files}")
        return False
    
    logger.info("Все необходимые файлы данных найдены")
    return True

def prepare_data_if_needed(orders_path: str, couriers_path: str, durations_db_path: str, 
                          data_dir: Path, force_prepare: bool = False) -> bool:
    """Подготавливает данные если нужно"""
    if force_prepare or not check_data_files(data_dir):
        logger.info("Начинаем подготовку данных...")
        
        try:
            preprocessor = DataPreprocessor(orders_path, couriers_path, durations_db_path)
            data = preprocessor.prepare_data()
            logger.info("Подготовка данных завершена успешно")
            return True
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            return False
    
    return True

def run_sa_solver(data_dir: str, durations_db_path: str, max_workers: int, time_budget: int, 
                 output_path: str, seed: int = None) -> bool:
    """Запускает SA решатель"""
    logger.info(f"Запускаем SA решатель: workers={max_workers}, time_budget={time_budget}s")
    
    # Устанавливаем seed для воспроизводимости
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Установлен seed: {seed}")
    
    # Инициализируем мониторинг ресурсов
    monitor = ResourceMonitor()
    monitor.log_current_usage()
    
    try:
        # Создаем решатель
        solver = SASolver(data_dir, durations_db_path, max_workers=max_workers, time_budget=time_budget)
        
        # Запускаем решение
        start_time = time.time()
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        # Логируем финальное использование ресурсов
        monitor.log_current_usage()
        resource_summary = monitor.get_summary()
        
        # Сохраняем решение
        solver.save_solution(solution, output_path)
        
        # Логируем результаты
        logger.info(f"Решение завершено за {elapsed:.1f} секунд")
        logger.info(f"Финальная стоимость: {solution.total_cost}")
        logger.info(f"Количество штрафов: {solution.penalty_count}")
        logger.info(f"Допустимость: {solution.feasible}")
        logger.info(f"Итераций: {solver.iterations}")
        logger.info(f"Принятых ходов: {solver.accepted_moves}")
        
        # Логируем сводку ресурсов
        logger.info(f"Сводка ресурсов: {resource_summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при решении: {e}")
        monitor.log_current_usage()
        return False

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='SA VRP решатель')
    
    parser.add_argument('--orders', required=True, help='Путь к файлу заказов')
    parser.add_argument('--couriers', required=True, help='Путь к файлу курьеров')
    parser.add_argument('--durations_db', required=True, help='Путь к SQLite базе расстояний')
    parser.add_argument('--output', default='solution.json', help='Путь для сохранения решения')
    parser.add_argument('--data_dir', default='sa_vrp_solver/data', help='Директория для данных')
    parser.add_argument('--max_workers', type=int, default=4, help='Максимальное количество процессов')
    parser.add_argument('--time_budget', type=int, default=2400, help='Бюджет времени в секундах (40 мин)')
    parser.add_argument('--seed', type=int, help='Seed для воспроизводимости')
    parser.add_argument('--force_prepare', action='store_true', help='Принудительная подготовка данных')
    parser.add_argument('--preprocessing_only', action='store_true', help='Только подготовка данных')
    
    args = parser.parse_args()
    
    # Создаем директории
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверяем входные файлы
    for file_path in [args.orders, args.couriers, args.durations_db]:
        if not Path(file_path).exists():
            logger.error(f"Файл не найден: {file_path}")
            return 1
    
    logger.info("=== SA VRP Решатель ===")
    logger.info(f"Входные файлы: orders={args.orders}, couriers={args.couriers}, durations_db={args.durations_db}")
    logger.info(f"Параметры: workers={args.max_workers}, time_budget={args.time_budget}s, seed={args.seed}")
    
    # Подготавливаем данные
    if not prepare_data_if_needed(args.orders, args.couriers, args.durations_db, 
                                 data_dir, args.force_prepare):
        logger.error("Не удалось подготовить данные")
        return 1
    
    if args.preprocessing_only:
        logger.info("Подготовка данных завершена")
        return 0
    
    # Запускаем решатель
    success = run_sa_solver(
        data_dir=str(data_dir),
        durations_db_path=args.durations_db,
        max_workers=args.max_workers,
        time_budget=args.time_budget,
        output_path=args.output,
        seed=args.seed
    )
    
    if success:
        logger.info("Решение завершено успешно")
        return 0
    else:
        logger.error("Решение завершено с ошибкой")
        return 1

if __name__ == "__main__":
    exit(main())
