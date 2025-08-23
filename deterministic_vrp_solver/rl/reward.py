from typing import List


def calculate_reward(
    courier_time: float,
    total_time: float,
    max_time: int,
    orders: int,
    all_courier_times: List[float],
    remaining_polygons: int = 0,
    violations: int = 0,
) -> float:
    """Shaped reward, согласованный с глобальной целью.

    Компоненты:
    - покрытие заказов (бонус за большее число заказов в полигоне)
    - штраф за время шага
    - штраф около/за превышение лимита курьера
    - штраф за оставшиеся полигоны (малый локальный градиент к убыванию остатка)
    - штраф за текущие нарушения (подталкивает к балансировке)
    - лёгкая справедливость (выравнивание загрузки)
    """
    # Масштабы подбираем так, чтобы совпадать по порядку с calculate_global_objective
    w_orders = 1.0  # бонус за покрытие
    w_time = 1.0 / 1500.0  # штраф за время шага
    w_remaining = -0.5  # каждый оставшийся полигон плохо
    w_viol = -50.0  # текущее число нарушений сильно плохо

    coverage_bonus = w_orders * max(0, orders)
    time_penalty = w_time * max(0.0, float(total_time))

    ratio = (courier_time / max_time) if max_time > 0 else 0.0
    if ratio > 1.0:
        over_penalty = -400.0
    elif ratio > 0.95:
        over_penalty = -120.0
    elif ratio > 0.85:
        over_penalty = -30.0
    else:
        over_penalty = 0.0

    # Небольшой градиент к уменьшению остатка и нарушений
    remaining_term = w_remaining * float(max(0, remaining_polygons))
    violation_term = w_viol * float(max(0, violations))

    fairness_bonus = 0.0
    if all_courier_times:
        avg_time = sum(all_courier_times) / len(all_courier_times)
        if avg_time > 0:
            diff = abs(courier_time - avg_time) / avg_time
            if diff < 0.1:
                fairness_bonus = 6.0
            elif diff < 0.2:
                fairness_bonus = 3.0

    return coverage_bonus - time_penalty + over_penalty + remaining_term + violation_term + fairness_bonus


