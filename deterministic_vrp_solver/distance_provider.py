"""
Провайдер расстояний - отвечает только за получение расстояний
"""

import sqlite3
import logging
from typing import List, Dict
from interfaces import IDistanceProvider

logger = logging.getLogger(__name__)

class SQLiteDistanceProvider(IDistanceProvider):
    """Провайдер расстояний из SQLite базы данных"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.distance_cache = {}
        self.portal_id_mapping = {}  # Маппинг Portal ID -> индекс
        self._setup_connection()
        self._build_portal_mapping()
    
    def _setup_connection(self):
        """Настройка соединения для оптимальной производительности"""
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=1000000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=1073741824")
        self.conn.execute("PRAGMA page_size=65536")
    
    def _build_portal_mapping(self):
        """Создает маппинг Portal IDs для правильной работы с большими ID"""
        cursor = self.conn.cursor()
        
        # Получаем все уникальные Portal IDs из базы
        cursor.execute("SELECT DISTINCT f FROM dists UNION SELECT DISTINCT t FROM dists ORDER BY f")
        portal_ids = [row[0] for row in cursor.fetchall()]
        
        # Создаем маппинг Portal ID -> индекс
        for idx, portal_id in enumerate(portal_ids):
            self.portal_id_mapping[portal_id] = idx
        
        logger.info(f"Создан маппинг для {len(self.portal_id_mapping)} Portal IDs")
        logger.debug(f"Диапазон Portal IDs: {min(portal_ids)} - {max(portal_ids)}")
    
    def get_distance(self, from_id: int, to_id: int) -> int:
        """Получить расстояние между двумя точками с кэшированием"""
        key = (from_id, to_id)
        if key not in self.distance_cache:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT d FROM dists WHERE f = ? AND t = ?",
                (from_id, to_id)
            )
            result = cursor.fetchone()
            distance = result[0] if result else 0
            # Если расстояние 0, значит нет пути - возвращаем большое значение как штраф
            self.distance_cache[key] = distance if distance > 0 else 999999
        return self.distance_cache[key]
    
    def get_distances_batch(self, from_ids: List[int], to_ids: List[int]) -> List[int]:
        """Получить расстояния для батча точек"""
        cursor = self.conn.cursor()
        
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
    
    def clear_cache(self):
        """Очистить кэш расстояний"""
        self.distance_cache.clear()
        logger.debug("Кэш расстояний очищен")
    
    def get_cache_stats(self) -> Dict:
        """Получить статистику кэша"""
        return {
            'cache_size': len(self.distance_cache),
            'cache_hits': getattr(self, '_cache_hits', 0),
            'cache_misses': getattr(self, '_cache_misses', 0)
        }
