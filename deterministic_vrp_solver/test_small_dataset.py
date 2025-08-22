#!/usr/bin/env python3
"""
Создание простого микротестового набора данных
"""

import json
import sqlite3
import random
import polars as pl
from pathlib import Path

def create_microtest_dataset():
    """Создает простой микротестовый набор данных"""
    
    print("🔧 Создание простого микротеста...")
    
    # Создаем директорию для тестовых данных
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Загружаем оригинальные данные
    with open('../ml_ozon_logistic/ml_ozon_logistic_dataSetOrders.json', 'r') as f:
        orders_data = json.load(f)
    with open('../ml_ozon_logistic/ml_ozon_logistic_dataSetCouriers.json', 'r') as f:
        couriers_data = json.load(f)
    
    # Преобразуем в DataFrame
    orders_df = pl.DataFrame(orders_data['Orders'])
    couriers_df = pl.DataFrame(couriers_data['Couriers'])
    
    print(f"Оригинальные данные: {len(orders_df)} заказов, {len(couriers_df)} курьеров")
    
    # Берем первые 300 заказов и 50 курьеров
    small_orders = orders_df.head(300)
    small_couriers = couriers_df.head(50)
    
    print(f"Микротест: {len(small_orders)} заказов, {len(small_couriers)} курьеров")
    
    # Получаем ID заказов
    order_ids = small_orders['ID'].to_list()
    
    # Создаем простую базу расстояний
    print("🗄️ Создание базы расстояний...")
    test_db_path = test_data_dir / "test_durations.sqlite"
    
    if test_db_path.exists():
        test_db_path.unlink()
    
    test_conn = sqlite3.connect(test_db_path)
    test_cursor = test_conn.cursor()
    
    # Создаем таблицу
    test_cursor.execute("CREATE TABLE dists (f INTEGER, t INTEGER, d INTEGER)")
    
    # Создаем расстояния между всеми точками (включая склад)
    warehouse_id = 1
    all_points = [warehouse_id] + order_ids
    
    for from_id in all_points:
        for to_id in all_points:
            if from_id != to_id:
                # Простые расстояния от 100 до 1000 секунд
                distance = random.randint(100, 1000)
                test_cursor.execute("INSERT INTO dists (f, t, d) VALUES (?, ?, ?)", 
                                  (from_id, to_id, distance))
    
    test_conn.commit()
    test_conn.close()
    
    # Сохраняем тестовые данные
    print("💾 Сохранение данных...")
    
    small_orders_json = {"Orders": small_orders.to_dicts()}
    with open(test_data_dir / "test_orders.json", 'w') as f:
        json.dump(small_orders_json, f, indent=2)
    
    # Создаем полную структуру данных курьеров с информацией о складе
    small_couriers_json = {
        "CourierTimeWork": {
            "TSStart": 28800,
            "TSEnd": 72000
        },
        "Warehouse": {
            "ID": 1,
            "Lat": 55.7558,
            "Long": 37.6176,
            "MpId": 0,
            "ServiceTime": 300
        },
        "Couriers": small_couriers.to_dicts()
    }
    with open(test_data_dir / "test_couriers.json", 'w') as f:
        json.dump(small_couriers_json, f, indent=2)
    
    print(f"✅ Микротест создан:")
    print(f"  {len(small_orders)} заказов, {len(small_couriers)} курьеров")
    print(f"  База расстояний: {len(all_points)} точек")
    
    return True

if __name__ == "__main__":
    create_microtest_dataset()
