#!/usr/bin/env python3
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Header
nb.cells.append(nbf.v4.new_markdown_cell("""# Анализ сервисных времен курьеров

Содержательные визуализации для относительной оценки эффективности курьеров с использованием Polars"""))

# Imports
nb.cells.append(nbf.v4.new_code_cell("""import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("Библиотеки загружены")"""))

# Data Loading
nb.cells.append(nbf.v4.new_code_cell("""# Загрузка данных
logger.info("Загружаем данные...")

# Couriers
with open("ml_ozon_logistic/ml_ozon_logistic_dataSetCouriers.json", 'r') as f:
    couriers_data = json.load(f)
couriers_df = pl.DataFrame(couriers_data['Couriers'], orient="row")

logger.info(f"✅ Курьеры загружены: {len(couriers_df)} записей")"""))

# Service Times Analysis
nb.cells.append(nbf.v4.new_code_cell("""# Анализ сервисных времен
logger.info("=== АНАЛИЗ СЕРВИСНЫХ ВРЕМЕН КУРЬЕРОВ ===")

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
logger.info(f"📊 Общее количество записей сервисных времен: {len(service_df)}")

# Вычисляем среднее время для каждого полигона
polygon_avg_times = service_df.group_by('mp_id').agg([
    pl.col('service_time').mean().alias('avg_service_time'),
    pl.col('service_time').count().alias('courier_count')
])

logger.info(f"🏘️ Уникальных полигонов: {len(polygon_avg_times)}")

# Объединяем с исходными данными для сравнения
comparison_df = service_df.join(polygon_avg_times, on='mp_id', how='left')

# Добавляем флаги сравнения со средним
comparison_df = comparison_df.with_columns([
    (pl.col('service_time') < pl.col('avg_service_time')).alias('faster_than_avg'),
    (pl.col('service_time') == pl.col('avg_service_time')).alias('equal_to_avg'),
    (pl.col('service_time') > pl.col('avg_service_time')).alias('slower_than_avg')
])

logger.info("✅ Добавлены флаги сравнения со средним")"""))

# Courier Performance Analysis
nb.cells.append(nbf.v4.new_code_cell("""# Анализ эффективности курьеров
logger.info("=== АНАЛИЗ ЭФФЕКТИВНОСТИ КУРЬЕРОВ ===")

# Группируем по курьерам
courier_performance = comparison_df.group_by('courier_id').agg([
    pl.col('faster_than_avg').sum().alias('faster_count'),
    pl.col('equal_to_avg').sum().alias('equal_count'),
    pl.col('slower_than_avg').sum().alias('slower_count'),
    pl.col('mp_id').n_unique().alias('total_polygons'),
    pl.col('service_time').mean().alias('avg_service_time'),
    pl.col('service_time').std().alias('std_service_time'),
    pl.col('service_time').min().alias('min_service_time'),
    pl.col('service_time').max().alias('max_service_time')
])

# Вычисляем процент быстрых полигонов
courier_performance = courier_performance.with_columns([
    (pl.col('faster_count') / pl.col('total_polygons') * 100).alias('faster_percentage'),
    (pl.col('slower_count') / pl.col('total_polygons') * 100).alias('slower_percentage')
])

logger.info(f"📊 Анализ по курьерам:")
logger.info(f"   • Средний процент быстрых полигонов: {courier_performance['faster_percentage'].mean():.1f}%")
logger.info(f"   • Медиана процента быстрых полигонов: {courier_performance['faster_percentage'].median():.1f}%")
logger.info(f"   • Стандартное отклонение: {courier_performance['faster_percentage'].std():.1f}%")"""))

# Polygon Analysis
nb.cells.append(nbf.v4.new_code_cell("""# Анализ по полигонам
logger.info("=== АНАЛИЗ ПО ПОЛИГОНАМ ===")

polygon_analysis = comparison_df.group_by('mp_id').agg([
    pl.col('faster_than_avg').sum().alias('faster_couriers'),
    pl.col('slower_than_avg').sum().alias('slower_couriers'),
    pl.col('courier_id').n_unique().alias('total_couriers'),
    pl.col('avg_service_time').first().alias('avg_service_time')
])

polygon_analysis = polygon_analysis.with_columns([
    (pl.col('faster_couriers') / pl.col('total_couriers') * 100).alias('faster_courier_percentage')
])

logger.info(f"🏘️ Анализ по полигонам:")
logger.info(f"   • Среднее время обслуживания по полигонам: {polygon_analysis['avg_service_time'].mean():.1f} сек")
logger.info(f"   • Полигонов с временем >200 сек: {(polygon_analysis['avg_service_time'] > 200).sum()}")
logger.info(f"   • Полигонов с временем <100 сек: {(polygon_analysis['avg_service_time'] < 100).sum()}")"""))

# Visualization 1: Heatmap - Service Time Matrix
nb.cells.append(nbf.v4.new_code_cell("""# 1. Тепловая карта: Матрица времени обслуживания курьеров по полигонам
logger.info("1. Создаем тепловую карту времени обслуживания...")

# Выбираем топ-30 курьеров и полигонов для читаемости
top_couriers = courier_performance.sort('avg_service_time', descending=False).head(30)['courier_id'].to_list()
top_polygons = polygon_analysis.sort('avg_service_time', descending=False).head(30)['mp_id'].to_list()

# Фильтруем данные для тепловой карты
heatmap_data = comparison_df.filter(
    pl.col('courier_id').is_in(top_couriers) & 
    pl.col('mp_id').is_in(top_polygons)
)

# Получаем уникальных курьеров и полигоны
couriers = sorted(heatmap_data['courier_id'].unique().to_list())
polygons = sorted(heatmap_data['mp_id'].unique().to_list())

# Создаем матрицу времени обслуживания
z_values = []
for courier in couriers:
    row = []
    for polygon in polygons:
        # Найти время обслуживания для данной пары курьер-полигон
        value = heatmap_data.filter(
            (pl.col('courier_id') == courier) & (pl.col('mp_id') == polygon)
        )
        if len(value) > 0:
            row.append(float(value['service_time'][0]))
        else:
            row.append(None)  # Нет данных
    z_values.append(row)

x_labels = [str(x) for x in polygons]
y_labels = [str(y) for y in couriers]

fig1 = go.Figure(data=go.Heatmap(
    z=z_values,
    x=x_labels,
    y=y_labels,
    colorscale='Viridis',
    text=z_values,
    texttemplate="%{text:.0f}",
    textfont={"size": 8},
    colorbar=dict(title="Время (сек)")
))
fig1.update_layout(
    title="Тепловая карта: Время обслуживания курьеров по полигонам (сек)<br>Топ-30 курьеров и полигонов",
    xaxis_title="Полигоны",
    yaxis_title="Курьеры",
    width=800,
    height=600
)
fig1.show()"""))

# Visualization 2: Heatmap - Relative Performance Matrix
nb.cells.append(nbf.v4.new_code_cell("""# 2. Тепловая карта: Относительная производительность курьеров
logger.info("2. Создаем тепловую карту относительной производительности...")

# Создаем матрицу относительной производительности
# Группируем по полигонам и курьерам для анализа
polygon_courier_matrix = comparison_df.group_by(['mp_id', 'courier_id']).agg([
    pl.col('service_time').first().alias('service_time'),
    pl.col('avg_service_time').first().alias('avg_service_time')
]).with_columns([
    # Относительное время: отношение времени курьера к среднему времени полигона
    (pl.col('service_time') / pl.col('avg_service_time')).alias('relative_time')
])

# Выбираем топ-25 полигонов по количеству курьеров
top_polygons_by_couriers = comparison_df.group_by('mp_id').agg([
    pl.col('courier_id').n_unique().alias('courier_count')
]).sort('courier_count', descending=True).head(25)['mp_id'].to_list()

# Выбираем топ-25 курьеров по количеству полигонов
top_couriers_by_polygons = comparison_df.group_by('courier_id').agg([
    pl.col('mp_id').n_unique().alias('polygon_count')
]).sort('polygon_count', descending=True).head(25)['courier_id'].to_list()

# Фильтруем данные
matrix_data = polygon_courier_matrix.filter(
    pl.col('mp_id').is_in(top_polygons_by_couriers) & 
    pl.col('courier_id').is_in(top_couriers_by_polygons)
)

# Создаем матрицу для тепловой карты
# Получаем уникальных курьеров и полигоны
couriers = sorted(matrix_data['courier_id'].unique().to_list())
polygons = sorted(matrix_data['mp_id'].unique().to_list())

# Создаем матрицу относительных времен
z_values = []
for courier in couriers:
    row = []
    for polygon in polygons:
        # Найти относительное время для данной пары курьер-полигон
        value = matrix_data.filter(
            (pl.col('courier_id') == courier) & (pl.col('mp_id') == polygon)
        )
        if len(value) > 0:
            row.append(float(value['relative_time'][0]))
        else:
            row.append(None)  # Нет данных
    z_values.append(row)

x_labels = [str(x) for x in polygons]
y_labels = [str(y) for y in couriers]

fig2 = go.Figure(data=go.Heatmap(
    z=z_values,
    x=x_labels,
    y=y_labels,
    colorscale='RdYlBu_r',
    text=z_values,
    texttemplate="%{text:.2f}",
    textfont={"size": 8},
    colorbar=dict(title="Относительное время<br>(1.0 = среднее)")
))
fig2.update_layout(
    title="Тепловая карта: Относительная производительность курьеров<br>(<1.0 быстрее среднего, >1.0 медленнее среднего)",
    xaxis_title="Полигоны",
    yaxis_title="Курьеры",
    width=800,
    height=600
)
fig2.show()"""))

# Visualization 3: Distribution of Average Service Time per Courier
nb.cells.append(nbf.v4.new_code_cell("""# 3. Распределение среднего времени обслуживания курьеров
logger.info("3. Анализируем распределение среднего времени обслуживания...")

courier_avg_times = courier_performance.select('avg_service_time').to_series().to_list()

fig3 = go.Figure()
fig3.add_trace(go.Histogram(
    x=courier_avg_times,
    nbinsx=30,
    name='Среднее время обслуживания',
    marker_color='blue'
))
fig3.update_layout(
    title="Распределение курьеров по среднему времени обслуживания",
    xaxis_title="Среднее время обслуживания (сек)",
    yaxis_title="Количество курьеров",
    showlegend=False,
    plot_bgcolor='white'
)
fig3.show()"""))

# Visualization 4: Polygon Variability
nb.cells.append(nbf.v4.new_code_cell("""# 4. Топ-15 полигонов с наибольшей вариативностью
logger.info("4. Анализируем полигоны с наибольшей вариативностью...")

polygon_variability = comparison_df.group_by('mp_id').agg([
    pl.col('service_time').std().alias('std_time'),
    pl.col('service_time').mean().alias('mean_time'),
    pl.col('courier_id').n_unique().alias('courier_count')
])

polygon_variability = polygon_variability.with_columns([
    (pl.col('std_time') / pl.col('mean_time') * 100).alias('cv_percent')
])

top_variable = polygon_variability.sort('cv_percent', descending=True).head(15)

polygon_ids = top_variable.select('mp_id').to_series().to_list()
cv_values = top_variable.select('cv_percent').to_series().to_list()

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=polygon_ids,
    y=cv_values,
    name='Коэффициент вариации (%)',
    marker_color='orange'
))
fig4.update_layout(
    title="Топ-15 полигонов с наибольшей вариативностью времени курьеров",
    xaxis_title="ID полигона",
    yaxis_title="Коэффициент вариации (%)",
    showlegend=False,
    plot_bgcolor='white'
)
fig4.show()"""))

# Visualization 5: Polygon Average Service Time Distribution
nb.cells.append(nbf.v4.new_code_cell("""# 5. Распределение среднего времени обслуживания по полигонам
logger.info("5. Анализируем распределение времени обслуживания по полигонам...")

polygon_avg_times = polygon_analysis.select('avg_service_time').to_series().to_list()

fig5 = go.Figure()
fig5.add_trace(go.Histogram(
    x=polygon_avg_times,
    nbinsx=30,
    name='Среднее время обслуживания',
    marker_color='purple'
))
fig5.update_layout(
    title="Распределение полигонов по среднему времени обслуживания",
    xaxis_title="Среднее время обслуживания (сек)",
    yaxis_title="Количество полигонов",
    showlegend=False,
    plot_bgcolor='white'
)
fig5.show()

logger.info("✅ Создано 5 содержательных визуализаций!")"""))

# Summary Analysis
nb.cells.append(nbf.v4.new_code_cell("""# Сводный анализ
logger.info("=== СВОДНЫЙ АНАЛИЗ ===")

logger.info(f"📊 ОБЩАЯ СТАТИСТИКА:")
logger.info(f"   • Всего курьеров: {len(courier_performance)}")
logger.info(f"   • Всего полигонов: {len(polygon_analysis)}")
logger.info(f"   • Всего записей сервисных времен: {len(service_df)}")

logger.info(f"\\n🚚 ВРЕМЯ ОБСЛУЖИВАНИЯ КУРЬЕРОВ:")
logger.info(f"   • Среднее время обслуживания: {courier_performance['avg_service_time'].mean():.1f} сек")
logger.info(f"   • Медиана времени обслуживания: {courier_performance['avg_service_time'].median():.1f} сек")
logger.info(f"   • Максимальное время обслуживания: {courier_performance['avg_service_time'].max():.1f} сек")
logger.info(f"   • Минимальное время обслуживания: {courier_performance['avg_service_time'].min():.1f} сек")
logger.info(f"   • Стандартное отклонение: {courier_performance['avg_service_time'].std():.1f} сек")

logger.info(f"\\n🏘️ ВРЕМЯ ОБСЛУЖИВАНИЯ ПО ПОЛИГОНАМ:")
logger.info(f"   • Среднее время по полигонам: {polygon_analysis['avg_service_time'].mean():.1f} сек")
logger.info(f"   • Медиана времени по полигонам: {polygon_analysis['avg_service_time'].median():.1f} сек")
logger.info(f"   • Максимальное время в полигоне: {polygon_analysis['avg_service_time'].max():.1f} сек")
logger.info(f"   • Минимальное время в полигоне: {polygon_analysis['avg_service_time'].min():.1f} сек")
logger.info(f"   • Стандартное отклонение по полигонам: {polygon_analysis['avg_service_time'].std():.1f} сек")

logger.info(f"\\n📈 ВАРИАТИВНОСТЬ:")
logger.info(f"   • Курьеров с временем >300 сек: {len(courier_performance.filter(pl.col('avg_service_time') > 300))}")
logger.info(f"   • Курьеров с временем <60 сек: {len(courier_performance.filter(pl.col('avg_service_time') < 60))}")
logger.info(f"   • Полигонов с временем >300 сек: {len(polygon_analysis.filter(pl.col('avg_service_time') > 300))}")
logger.info(f"   • Полигонов с временем <60 сек: {len(polygon_analysis.filter(pl.col('avg_service_time') < 60))}")"""))

# Save notebook
nbf.write(nb, 'courier_service_analysis.ipynb')
print("✅ Jupyter notebook с содержательными визуализациями и тепловыми картами создан: courier_service_analysis.ipynb")
