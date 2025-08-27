from pathlib import Path
from typing import Dict, List
import sqlite3


def build_ports_database(ports_db_path: Path, polygon_ports: Dict[int, List[int]], durations_conn: sqlite3.Connection) -> None:
    if ports_db_path.exists():
        ports_db_path.unlink()
    conn = sqlite3.connect(str(ports_db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS port_distances (from_port INTEGER, to_port INTEGER, distance REAL)")
    all_ports = sorted({int(p) for ports in polygon_ports.values() for p in ports})
    dcur = durations_conn.cursor()
    dcur.execute("CREATE TEMP TABLE IF NOT EXISTS temp_ports (port_id INTEGER PRIMARY KEY)")
    dcur.executemany("INSERT OR IGNORE INTO temp_ports (port_id) VALUES (?)", [(pid,) for pid in all_ports])
    query = (
        "SELECT d.f AS from_port, d.t AS to_port, COALESCE(NULLIF(d.d, 0), 0) AS distance "
        "FROM dists d "
        "JOIN temp_ports pf ON pf.port_id = d.f "
        "JOIN temp_ports pt ON pt.port_id = d.t"
    )
    cur.execute("BEGIN")
    for chunk in dcur.execute(query):
        cur.execute("INSERT INTO port_distances (from_port, to_port, distance) VALUES (?, ?, ?)", chunk)
    conn.commit()
    dcur.execute("DELETE FROM temp_ports")
    conn.close()


def build_warehouse_ports_database(warehouse_ports_db_path: Path, polygon_ports: Dict[int, List[int]], durations_conn: sqlite3.Connection, warehouse_id: int) -> None:
    if warehouse_ports_db_path.exists():
        warehouse_ports_db_path.unlink()
    conn = sqlite3.connect(str(warehouse_ports_db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS warehouse_port_distances (port_id INTEGER PRIMARY KEY, distance REAL)")
    all_ports = sorted({int(p) for ports in polygon_ports.values() for p in ports})
    dcur = durations_conn.cursor()
    dcur.execute("CREATE TEMP TABLE IF NOT EXISTS temp_ports2 (port_id INTEGER PRIMARY KEY)")
    dcur.executemany("INSERT OR IGNORE INTO temp_ports2 (port_id) VALUES (?)", [(pid,) for pid in all_ports])
    query = (
        "SELECT p.port_id, COALESCE(NULLIF(d.d, 0), 0) AS distance "
        "FROM temp_ports2 p LEFT JOIN dists d ON d.f = ? AND d.t = p.port_id"
    )
    cur.execute("BEGIN")
    for row in dcur.execute(query, (int(warehouse_id),)):
        cur.execute("INSERT OR REPLACE INTO warehouse_port_distances (port_id, distance) VALUES (?, ?)", row)
    conn.commit()
    dcur.execute("DELETE FROM temp_ports2")
    conn.close()


