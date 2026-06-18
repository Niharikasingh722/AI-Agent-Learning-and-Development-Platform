"""
Persistent memory layer using SQLite.
Stores conversation history and agent result cache across sessions.
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                tools_used  TEXT,
                timestamp   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);

            CREATE TABLE IF NOT EXISTS agent_cache (
                cache_key  TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                label       TEXT,
                created_at  TEXT NOT NULL,
                last_active TEXT NOT NULL
            );
        """)


def new_session(label: str = "") -> str:
    session_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, label, created_at, last_active) VALUES (?, ?, ?, ?)",
            (session_id, label or f"Session {session_id}", now, now),
        )
    return session_id


def list_sessions() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT session_id, label, created_at, last_active "
            "FROM sessions ORDER BY last_active DESC LIMIT 20"
        ).fetchall()
    return [dict(r) for r in rows]


def save_message(
    session_id: str,
    role: str,
    content: str,
    tools_used: list | None = None,
) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, tools_used, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(tools_used) if tools_used else None, now),
        )
        conn.execute(
            "UPDATE sessions SET last_active = ? WHERE session_id = ?",
            (now, session_id),
        )


def load_history(session_id: str, limit: int = 30) -> list[dict]:
    """Return messages ordered oldest-first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, tools_used FROM conversations "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    result = []
    for r in reversed(rows):
        result.append({
            "role": r["role"],
            "content": r["content"],
            "tools_used": json.loads(r["tools_used"]) if r["tools_used"] else None,
        })
    return result


def cache_set(key: str, value: dict) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO agent_cache (cache_key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), now),
        )


def cache_get(key: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM agent_cache WHERE cache_key = ?", (key,)
        ).fetchone()
    return json.loads(row["value"]) if row else None
