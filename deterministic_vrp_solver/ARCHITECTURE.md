# Модульная архитектура VRP Solver

## Принципы SOLID

Проект реорганизован в соответствии с принципами SOLID:

### 1. Single Responsibility Principle (SRP)
Каждый модуль отвечает за одну задачу:
- `distance_provider.py` - только получение расстояний
- `polygon_validator.py` - только валидация полигонов
- `ortools_solver.py` - только OR-Tools решение
- `pyomo_solver.py` - только Pyomo решение
- `heuristic_solver.py` - только эвристическое решение

### 2. Open/Closed Principle (OCP)
Система открыта для расширения, закрыта для модификации:
- Новые решатели добавляются через интерфейсы
- Новые провайдеры расстояний через `IDistanceProvider`

### 3. Liskov Substitution Principle (LSP)
Все реализации интерфейсов взаимозаменяемы:
- `ORToolsAssignmentSolver`, `PyomoAssignmentSolver` и `HeuristicAssignmentSolver` взаимозаменяемы
- `SQLiteDistanceProvider` можно заменить на другой провайдер

### 4. Interface Segregation Principle (ISP)
Интерфейсы разделены на специфичные:
- `IDistanceProvider` - только расстояния
- `IPolygonAssignmentSolver` - только назначение
- `IPolygonValidator` - только валидация

### 5. Dependency Inversion Principle (DIP)
Зависимости инвертированы через интерфейсы:
- Решатели зависят от `IDistanceProvider`, а не от конкретной БД
- Фабрика создает решатели через интерфейсы

## Структура модулей

```
deterministic_vrp_solver/
├── interfaces.py              # Интерфейсы (абстракции)
├── distance_provider.py       # Провайдер расстояний
├── polygon_validator.py       # Валидатор полигонов
├── ortools_solver.py          # OR-Tools решатель
├── pyomo_solver.py            # Pyomo решатель
├── heuristic_solver.py        # Эвристический решатель
├── solver_factory.py          # Фабрика решателей
├── polygon_optimizer.py       # TSP оптимизатор
├── route_optimizer.py         # Оптимизатор маршрутов
├── solution_generator.py      # Генератор решений
├── utils.py                   # Утилиты
├── pyomo_example.py           # Пример использования Pyomo
└── main.py                    # Основной скрипт
```

## Решатели оптимизации

### 1. OR-Tools (Google)
- **Тип**: Constraint Programming (CP-SAT)
- **Преимущества**: Быстрый, хорошо документированный
- **Недостатки**: Менее гибкий, только Google решатели

### 2. Pyomo (Python Optimization Modeling Objects)
- **Тип**: Mathematical Programming
- **Преимущества**: 
  - Гибкий и расширяемый
  - Множество решателей (CBC, GLPK, CPLEX, Gurobi)
  - Нативный Python API
  - Академическая поддержка
- **Недостатки**: Медленнее OR-Tools для некоторых задач

### 3. Эвристический решатель
- **Тип**: Жадный алгоритм + локальный поиск
- **Преимущества**: Быстрый, всегда находит решение
- **Недостатки**: Не гарантирует оптимальность

## Сравнение решателей

| Критерий | OR-Tools | Pyomo | Эвристика |
|----------|----------|-------|-----------|
| Скорость | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Качество решения | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Гибкость | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Простота использования | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Поддержка решателей | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Использование Pyomo

### Установка
```bash
pip install pyomo cbc glpk
```

### Базовое использование
```python
from solver_factory import AssignmentSolverFactory

# Создаем Pyomo решатель с CBC
solver = AssignmentSolverFactory.create_solver(
    solver_type='pyomo',
    distance_provider=distance_provider,
    warehouse_id=1,
    solver_name='cbc'  # или 'glpk', 'cplex', 'gurobi'
)

# Решаем задачу
assignment = solver.solve(polygons_df, couriers_df)
```

### Гибридный подход
```python
# Pyomo + эвристика как fallback
solver = AssignmentSolverFactory.create_solver(
    solver_type='pyomo_hybrid',
    distance_provider=distance_provider,
    warehouse_id=1,
    solver_name='cbc'
)
```

### Релаксация для больших задач
```python
# Pyomo с релаксацией для больших задач
pyomo_solver = PyomoAssignmentSolver(distance_provider, warehouse_id, 'cbc')
assignment = pyomo_solver.solve_with_relaxation(polygons_df, couriers_df)
```

## Преимущества новой архитектуры

### 1. Модульность
- Каждый компонент можно тестировать отдельно
- Легко добавлять новые алгоритмы
- Простое переключение между решателями

### 2. Расширяемость
- Новые провайдеры расстояний (Redis, PostgreSQL)
- Новые решатели (генетические алгоритмы, нейронные сети)
- Новые валидаторы

### 3. Тестируемость
- Моки для интерфейсов
- Изолированное тестирование компонентов
- Легкая замена зависимостей

### 4. Производительность
- Кэширование расстояний
- Параллельная обработка полигонов
- Оптимизированные запросы к БД

### 5. Надежность
- Обработка ошибок на каждом уровне
- Валидация входных данных
- Graceful degradation при сбоях

## Использование

### Базовое использование
```python
from distance_provider import SQLiteDistanceProvider
from solver_factory import AssignmentSolverFactory

# Создаем провайдер расстояний
distance_provider = SQLiteDistanceProvider(conn)

# Создаем решатель через фабрику
solver = AssignmentSolverFactory.create_solver(
    solver_type='pyomo_hybrid',  # или 'ortools', 'heuristic'
    distance_provider=distance_provider,
    warehouse_id=1,
    solver_name='cbc'  # для Pyomo
)

# Решаем задачу
assignment = solver.solve(polygons_df, couriers_df)
```

### Добавление нового решателя
```python
from interfaces import IPolygonAssignmentSolver

class MyCustomSolver(IPolygonAssignmentSolver):
    def solve(self, polygons_df, couriers_df, max_time_per_courier=43200):
        # Ваша реализация
        pass

# Добавляем в фабрику
class AssignmentSolverFactory:
    @staticmethod
    def create_solver(solver_type, distance_provider, warehouse_id=1):
        if solver_type == 'custom':
            return MyCustomSolver(distance_provider, warehouse_id)
        # ...
```

## Решение проблем из логов

### 1. Segmentation Fault
- Разделение ответственности снижает нагрузку на память
- Изолированные компоненты легче отлаживать
- Graceful handling ошибок

### 2. Неназначаемые полигоны
- Ранняя валидация и фильтрация
- Гибридный подход (Pyomo + эвристика)
- Релаксация для больших задач
- Детальная статистика проблем

### 3. Производительность
- Кэширование расстояний
- Параллельная обработка
- Оптимизированные запросы к БД
- Выбор оптимального решателя

## Мониторинг и отладка

### Логирование
- Детальные логи на каждом этапе
- Статистика производительности
- Отслеживание ошибок

### Метрики
- Время выполнения каждого компонента
- Статистика кэша
- Количество назначенных полигонов
- Сравнение решателей

### Валидация
- Проверка на каждом этапе
- Детальная отчетность ошибок
- Graceful degradation
