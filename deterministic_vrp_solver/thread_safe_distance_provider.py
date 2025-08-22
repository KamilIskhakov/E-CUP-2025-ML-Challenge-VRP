"""
Thread-safe провайдер расстояний для multiprocessing
"""

import sqlite3
import threading
import logging
from typing import List, Dict, Optional
from interfaces import IDistanceProvider

logger = logging.getLogger(__name__)

class ThreadSafeDistanceProvider(IDistanceProvider):
    """Thread-safe провайдер расстояний для работы с multiprocessing"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._setup_connection()
    
    def _setup_connection(self):
        """Настройка основного соединения"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=1000000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=1073741824")
        self.conn.execute("PRAGMA page_size=65536")
    
    def get_connection(self) -> sqlite3.Connection:
        """Получить thread-local соединение"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=1000000")
            self._local.connection.execute("PRAGMA temp_store=MEMORY")
        return self._local.connection
    
    def get_distance(self, from_id: int, to_id: int) -> int:
        """Получить расстояние между двумя точками"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT d FROM dists WHERE f = ? AND t = ?",
            (from_id, to_id)
        )
        result = cursor.fetchone()
        distance = result[0] if result else 0
        return distance if distance > 0 else 999999
    
    def get_distances_batch(self, from_ids: List[int], to_ids: List[int]) -> List[int]:
        """Получить расстояния для батча точек"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Создаем временные таблицы для быстрого поиска
        cursor.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_from (id INTEGER PRIMARY KEY)
        """)
        cursor.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_to (id INTEGER PRIMARY KEY)
        """)
        
        # Вставляем данные
        cursor.executemany("INSERT OR REPLACE INTO temp_from (id) VALUES (?)", 
                          [(id,) for id in from_ids])
        cursor.executemany("INSERT OR REPLACE INTO temp_to (id) VALUES (?)", 
                          [(id,) for id in to_ids])
        
        # Выполняем запрос
        cursor.execute("""
            SELECT d.d 
            FROM dists d
            INNER JOIN temp_from f ON d.f = f.id
            INNER JOIN temp_to t ON d.t = t.id
            ORDER BY f.id, t.id
        """)
        
        results = [row[0] for row in cursor.fetchall()]
        
        # Очищаем временные таблицы
        cursor.execute("DELETE FROM temp_from")
        cursor.execute("DELETE FROM temp_to")
        
        return results
    
    def close_all_connections(self):
        """Закрыть все соединения"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
        if hasattr(self, 'conn'):
            self.conn.close()

class ProcessSafeDistanceProvider:
    """Process-safe провайдер для multiprocessing"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_distance(self, from_id: int, to_id: int) -> int:
        """Получить расстояние (создает новое соединение для каждого процесса)"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=1000000")
        
        cursor = conn.cursor()
        cursor.execute(
            "SELECT d FROM dists WHERE f = ? AND t = ?",
            (from_id, to_id)
        )
        result = cursor.fetchone()
        distance = result[0] if result else 0
        
        conn.close()
        return distance if distance > 0 else 999999
    
    def get_distances_batch(self, from_ids: List[int], to_ids: List[int]) -> List[int]:
        """Получить расстояния для батча (создает новое соединение)"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=1000000")
        
        cursor = conn.cursor()
        
        # Создаем временные таблицы
        cursor.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_from (id INTEGER PRIMARY KEY)
        """)
        cursor.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_to (id INTEGER PRIMARY KEY)
        """)
        
        # Вставляем данные
        cursor.executemany("INSERT OR REPLACE INTO temp_from (id) VALUES (?)", 
                          [(id,) for id in from_ids])
        cursor.executemany("INSERT OR REPLACE INTO temp_to (id) VALUES (?)", 
                          [(id,) for id in to_ids])
        
        # Выполняем запрос
        cursor.execute("""
            SELECT d.d 
            FROM dists d
            INNER JOIN temp_from f ON d.f = f.id
            INNER JOIN temp_to t ON d.t = t.id
            ORDER BY f.id, t.id
        """)
        
        results = [row[0] for row in cursor.fetchall()]
        
        # Очищаем временные таблицы
        cursor.execute("DELETE FROM temp_from")
        cursor.execute("DELETE FROM temp_to")
        
        conn.close()
        return results
