#!/bin/bash

# Скрипт запуска детерминированного VRP решателя

echo "Запуск детерминированного VRP решателя..."

# Проверяем наличие необходимых файлов
if [ ! -f "../ml_ozon_logistic/ml_ozon_logistic_dataSetOrders.json" ]; then
    echo "Ошибка: Файл заказов не найден"
    exit 1
fi

if [ ! -f "../ml_ozon_logistic/ml_ozon_logistic_dataSetCouriers.json" ]; then
    echo "Ошибка: Файл курьеров не найден"
    exit 1
fi

if [ ! -f "../durations.sqlite" ]; then
    echo "Ошибка: База данных расстояний не найдена"
    exit 1
fi

# Устанавливаем зависимости
echo "Установка зависимостей..."
pip install -r requirements.txt

# Запускаем решатель
echo "Запуск алгоритма..."
python main.py

echo "Завершено!"
