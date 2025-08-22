#!/usr/bin/env python3
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Header
nb.cells.append(nbf.v4.new_markdown_cell("""# Анализ микрополигонов и сервисных времен VRP

Анализ диаметра микрополигонов и персональных сервисных времен курьеров"""))

# Imports
nb.cells.append(nbf.v4.new_code_cell("""import sqlite3
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import json

print("Библиотеки загружены")"""))

# Data Loading
nb.cells.append(nbf.v4.new_code_cell("""# Загрузка данных
print("Загружаем данные...")

# Orders
with open("ml_ozon_logistic/ml_ozon_logistic_dataSetOrders.json", 'r') as f:
    orders_data = json.load(f)
orders_df = pl.DataFrame(orders_data['Orders'], orient="row")

# Couriers
with open("ml_ozon_logistic/ml_ozon_logistic_dataSetCouriers.json", 'r') as f:
    couriers_data = json.load(f)
couriers_df = pl.DataFrame(couriers_data['Couriers'], orient="row")
warehouse_info = couriers_data['Warehouse']

print(f"✅ Заказы загружены: {len(orders_df)} записей")
print(f"✅ Курьеры загружены: {len(couriers_df)} записей")
print(f"🏢 Склад: ID={warehouse_info['ID']}, координаты=({warehouse_info['Lat']:.4f}, {warehouse_info['Long']:.4f})")"""))

# Polygon Analysis - Diameter
nb.cells.append(nbf.v4.new_code_cell("""# Анализ диаметра микрополигонов
print("=== АНАЛИЗ ДИАМЕТРА МИКРОПОЛИГОНОВ ===")

# Подключение к SQLite
db_path = "durations.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Получаем все уникальные микрополигоны
unique_polygons = orders_df.select('MpId').unique().to_series().to_list()
print(f"📊 Анализируем {len(unique_polygons)} микрополигонов")

# Анализ каждого полигона
polygon_diameters = []

for mp_id in unique_polygons:
    # Получаем все заказы в этом полигоне
    polygon_orders = orders_df.filter(pl.col('MpId') == mp_id)
    order_ids = polygon_orders.select('ID').to_series().to_list()
    
    if len(order_ids) <= 1:
        # Один заказ - диаметр = 0
        polygon_diameters.append({
            'mp_id': mp_id,
            'order_count': len(order_ids),
            'max_distance': 0,
            'avg_distance': 0,
            'min_distance': 0
        })
        continue
    
    # Получаем все расстояния между заказами в полигоне
    placeholders = ','.join(['?' for _ in order_ids])
    query = f"SELECT f, t, d FROM dists WHERE f IN ({placeholders}) AND t IN ({placeholders}) AND f != t"
    
    # Параметры для IN clause (дважды - для f и t)
    params = order_ids + order_ids
    
    cursor.execute(query, params)
    distances = cursor.fetchall()
    
    if distances:
        dist_values = [d[2] for d in distances]
        polygon_diameters.append({
            'mp_id': mp_id,
            'order_count': len(order_ids),
            'max_distance': max(dist_values),
            'avg_distance': sum(dist_values) / len(dist_values),
            'min_distance': min(dist_values)
        })
    else:
        polygon_diameters.append({
            'mp_id': mp_id,
            'order_count': len(order_ids),
            'max_distance': 0,
            'avg_distance': 0,
            'min_distance': 0
        })

# Создаем DataFrame
diameters_df = pl.DataFrame(polygon_diameters, orient="row")
print(f"✅ Проанализированы все полигоны")

# Статистики диаметров
print(f"\\n📏 Статистики диаметров полигонов:")
print(f"   • Максимальный диаметр: {diameters_df['max_distance'].max()} сек")
print(f"   • Средний диаметр: {diameters_df['max_distance'].mean():.1f} сек")
print(f"   • Медиана диаметра: {diameters_df['max_distance'].median():.1f} сек")
print(f"   • Минимальный диаметр: {diameters_df['max_distance'].min()} сек")
print(f"   • Стандартное отклонение: {diameters_df['max_distance'].std():.1f} сек")

# Квантили диаметров
print(f"\\n📈 Квантили диаметров:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    value = diameters_df['max_distance'].quantile(q)
    print(f"   • {q*100}%: {value:.1f} сек")

conn.close()"""))

# Service Times Analysis
nb.cells.append(nbf.v4.new_code_cell("""# Анализ персональных сервисных времен
print("=== АНАЛИЗ СЕРВИСНЫХ ВРЕМЕН ===")

# Создаем DataFrame сервисных времен
service_times = []
for row in couriers_df.iter_rows(named=True):
    for service in row['ServiceTimeInMps']:
        service_times.append({
            'courier_id': row['ID'],
            'mp_id': service['MpID'],
            'service_time': service['ServiceTime']
        })

service_df = pl.DataFrame(service_times, orient="row")
print(f"📊 Общее количество записей сервисных времен: {len(service_df)}")

# Общие статистики
print(f"\\n⏱️ Общие статистики сервисных времен:")
print(f"   • Минимум: {service_df['service_time'].min()} сек")
print(f"   • Максимум: {service_df['service_time'].max()} сек")
print(f"   • Среднее: {service_df['service_time'].mean():.1f} сек")
print(f"   • Медиана: {service_df['service_time'].median():.1f} сек")
print(f"   • Стандартное отклонение: {service_df['service_time'].std():.1f} сек")

# Квантили
print(f"\\n📈 Квантили сервисных времен:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    value = service_df['service_time'].quantile(q)
    print(f"   • {q*100}%: {value:.1f} сек")

# Анализ по курьерам
print(f"\\n🚚 Анализ по курьерам:")
courier_stats = service_df.group_by('courier_id').agg([
    pl.col('service_time').mean().alias('avg_service_time'),
    pl.col('service_time').std().alias('std_service_time'),
    pl.col('service_time').min().alias('min_service_time'),
    pl.col('service_time').max().alias('max_service_time'),
    pl.col('service_time').count().alias('polygon_count')
])

print(f"   • Среднее время на курьера: {courier_stats['avg_service_time'].mean():.1f} сек")
print(f"   • Медиана времени на курьера: {courier_stats['avg_service_time'].median():.1f} сек")
print(f"   • Стандартное отклонение между курьерами: {courier_stats['avg_service_time'].std():.1f} сек")
print(f"   • Разброс времени на курьера: {courier_stats['max_service_time'].mean() - courier_stats['min_service_time'].mean():.1f} сек")

# Анализ по полигонам
print(f"\\n🏘️ Анализ по полигонам:")
polygon_stats = service_df.group_by('mp_id').agg([
    pl.col('service_time').mean().alias('avg_service_time'),
    pl.col('service_time').std().alias('std_service_time'),
    pl.col('service_time').min().alias('min_service_time'),
    pl.col('service_time').max().alias('max_service_time'),
    pl.col('service_time').count().alias('courier_count')
])

print(f"   • Среднее время на полигон: {polygon_stats['avg_service_time'].mean():.1f} сек")
print(f"   • Медиана времени на полигон: {polygon_stats['avg_service_time'].median():.1f} сек")
print(f"   • Стандартное отклонение между полигонами: {polygon_stats['avg_service_time'].std():.1f} сек")
print(f"   • Разброс времени на полигон: {polygon_stats['max_service_time'].mean() - polygon_stats['min_service_time'].mean():.1f} сек")"""))

# Visualizations - Diameter Analysis
nb.cells.append(nbf.v4.new_code_cell("""# Визуализация анализа диаметров
print("Создаем визуализации диаметров...")

# Convert to pandas for plotly
diameters_pd = diameters_df.to_pandas()

# 1. Distribution of polygon diameters
fig1 = go.Figure()
fig1.add_trace(go.Histogram(
    x=diameters_pd['max_distance'],
    nbinsx=50,
    name='Распределение диаметров',
    marker_color='lightblue'
))
fig1.update_layout(title="Распределение диаметров микрополигонов",
                  xaxis_title="Диаметр (секунды)",
                  yaxis_title="Количество полигонов")
fig1.show()

# 2. Diameter vs Order Count
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=diameters_pd['order_count'],
    y=diameters_pd['max_distance'],
    mode='markers',
    name='Диаметр vs Количество заказов',
    marker=dict(size=5, color='red', opacity=0.6)
))
fig2.update_layout(title="Зависимость диаметра от количества заказов",
                  xaxis_title="Количество заказов в полигоне",
                  yaxis_title="Диаметр полигона (секунды)")
fig2.show()

# 3. Top 20 largest diameters
fig3 = go.Figure()
top_diameters = diameters_df.sort('max_distance', descending=True).head(20)
top_diameters_pd = top_diameters.to_pandas()
fig3.add_trace(go.Bar(
    x=top_diameters_pd['mp_id'],
    y=top_diameters_pd['max_distance'],
    name='Топ-20 диаметров',
    marker_color='orange'
))
fig3.update_layout(title="Топ-20 полигонов с наибольшим диаметром",
                  xaxis_title="ID полигона",
                  yaxis_title="Диаметр (секунды)")
fig3.show()

# 4. Average vs Max distance in polygons
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=diameters_pd['avg_distance'],
    y=diameters_pd['max_distance'],
    mode='markers',
    name='Среднее vs Максимум',
    marker=dict(size=5, color='green', opacity=0.6)
))
fig4.update_layout(title="Среднее vs Максимальное расстояние в полигонах",
                  xaxis_title="Среднее расстояние (секунды)",
                  yaxis_title="Максимальное расстояние (секунды)")
fig4.show()"""))

# Visualizations - Service Times
nb.cells.append(nbf.v4.new_code_cell("""# Визуализация сервисных времен
print("Создаем визуализации сервисных времен...")

# Convert to pandas for plotly
service_pd = service_df.to_pandas()
courier_stats_pd = courier_stats.to_pandas()
polygon_stats_pd = polygon_stats.to_pandas()

# 1. Distribution of service times
fig5 = go.Figure()
fig5.add_trace(go.Histogram(
    x=service_pd['service_time'],
    nbinsx=100,
    name='Распределение сервисных времен',
    marker_color='lightgreen'
))
fig5.update_layout(title="Распределение сервисных времен",
                  xaxis_title="Время (секунды)",
                  yaxis_title="Количество")
fig5.show()

# 2. Service times by courier
fig6 = go.Figure()
fig6.add_trace(go.Box(
    y=service_pd['service_time'],
    x=service_pd['courier_id'],
    name='Сервисные времена по курьерам',
    boxpoints='outliers'
))
fig6.update_layout(title="Распределение сервисных времен по курьерам",
                  xaxis_title="ID курьера",
                  yaxis_title="Время (секунды)")
fig6.show()

# 3. Average service time by courier
fig7 = go.Figure()
fig7.add_trace(go.Bar(
    x=courier_stats_pd['courier_id'],
    y=courier_stats_pd['avg_service_time'],
    name='Среднее время на курьера',
    marker_color='purple'
))
fig7.update_layout(title="Среднее сервисное время по курьерам",
                  xaxis_title="ID курьера",
                  yaxis_title="Среднее время (секунды)")
fig7.show()

# 4. Service time variation by polygon
fig8 = go.Figure()
fig8.add_trace(go.Scatter(
    x=polygon_stats_pd['avg_service_time'],
    y=polygon_stats_pd['std_service_time'],
    mode='markers',
    name='Среднее vs Стандартное отклонение',
    marker=dict(size=5, color='brown', opacity=0.6)
))
fig8.update_layout(title="Вариация сервисных времен по полигонам",
                  xaxis_title="Среднее время (секунды)",
                  yaxis_title="Стандартное отклонение (секунды)")
fig8.show()"""))

# Summary Analysis
nb.cells.append(nbf.v4.new_code_cell("""# Сводный анализ
print("=== СВОДНЫЙ АНАЛИЗ ===")

print(f"📊 ДИАМЕТРЫ ПОЛИГОНОВ:")
print(f"   • Полигонов с диаметром > 1000 сек: {(diameters_df['max_distance'] > 1000).sum()}")
print(f"   • Полигонов с диаметром > 2000 сек: {(diameters_df['max_distance'] > 2000).sum()}")
print(f"   • Полигонов с диаметром > 5000 сек: {(diameters_df['max_distance'] > 5000).sum()}")

print(f"\\n⏱️ СЕРВИСНЫЕ ВРЕМЕНА:")
print(f"   • Записей с временем > 100 сек: {(service_df['service_time'] > 100).sum()}")
print(f"   • Записей с временем > 200 сек: {(service_df['service_time'] > 200).sum()}")
print(f"   • Записей с временем > 300 сек: {(service_df['service_time'] > 300).sum()}")

# Критические полигоны
critical_polygons = diameters_df.filter(pl.col('max_distance') > 2000)
if len(critical_polygons) > 0:
    print(f"\\n⚠️ КРИТИЧЕСКИЕ ПОЛИГОНЫ (диаметр > 2000 сек):")
    print(critical_polygons.sort('max_distance', descending=True).head(10))

# Проблемные сервисные времена
problematic_times = service_df.filter(pl.col('service_time') > 200)
if len(problematic_times) > 0:
    print(f"\\n⚠️ ПРОБЛЕМНЫЕ СЕРВИСНЫЕ ВРЕМЕНА (> 200 сек):")
    print(problematic_times.sort('service_time', descending=True).head(10))

print(f"\\n🎯 ВЫВОДЫ:")
print(f"   • Средний диаметр полигона: {diameters_df['max_distance'].mean():.1f} сек")
print(f"   • Среднее сервисное время: {service_df['service_time'].mean():.1f} сек")
print(f"   • Соотношение диаметр/сервис: {diameters_df['max_distance'].mean() / service_df['service_time'].mean():.1f}")"""))

# Save notebook
nbf.write(nb, 'polygon_analysis.ipynb')
print("✅ Ноутбук анализа полигонов создан: polygon_analysis.ipynb")
