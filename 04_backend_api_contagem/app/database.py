from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "06_dados_e_sessoes" / "contador_caixas.sqlite3"


class SessionStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS counting_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    source TEXT NOT NULL,
                    line_config TEXT NOT NULL,
                    total INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS count_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    centroid_x INTEGER NOT NULL,
                    centroid_y INTEGER NOT NULL,
                    total_after_event INTEGER NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES counting_sessions(id)
                );
                """
            )

    def create_session(self, source: str, line_config: dict[str, float]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO counting_sessions (started_at, source, line_config, total, status)
                VALUES (?, ?, ?, 0, 'running')
                """,
                (now_iso(), source, json.dumps(line_config, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def finish_session(self, session_id: int, total: int) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE counting_sessions
                SET ended_at = ?, total = ?, status = 'finished'
                WHERE id = ?
                """,
                (now_iso(), total, session_id),
            )

    def update_total(self, session_id: int, total: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE counting_sessions SET total = ? WHERE id = ?",
                (total, session_id),
            )

    def insert_event(
        self,
        session_id: int,
        track_id: int,
        direction: str,
        centroid: tuple[int, int],
        total: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO count_events (
                    session_id,
                    created_at,
                    track_id,
                    direction,
                    centroid_x,
                    centroid_y,
                    total_after_event
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, now_iso(), track_id, direction, centroid[0], centroid[1], total),
            )

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, started_at, ended_at, source, line_config, total, status
                FROM counting_sessions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_events(self, session_id: int, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, track_id, direction, centroid_x, centroid_y, total_after_event
                FROM count_events
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
