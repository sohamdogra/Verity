"""SQLite storage via the stdlib sqlite3 module. One short-lived connection per call."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS families (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    safe_word_hash  TEXT NOT NULL,
    trusted_phone   TEXT NOT NULL,
    alert_contact   TEXT NOT NULL,
    alert_phone     TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS challenges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id   INTEGER NOT NULL REFERENCES families(id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    answer_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id            INTEGER REFERENCES families(id) ON DELETE CASCADE,
    timestamp            TEXT NOT NULL,
    event_type           TEXT NOT NULL,
    synthetic_likelihood REAL,
    band                 TEXT,
    verification_result  TEXT,
    notes                TEXT
);

CREATE TABLE IF NOT EXISTS verification_attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id  INTEGER NOT NULL REFERENCES families(id) ON DELETE CASCADE,
    timestamp  REAL NOT NULL,
    passed     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_family ON events(family_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_attempts_family ON verification_attempts(family_id, timestamp);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        # Migration for databases created before SMS alerts existed.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(families)")}
        if "alert_phone" not in columns:
            conn.execute("ALTER TABLE families ADD COLUMN alert_phone TEXT")


# ---- families -------------------------------------------------------------------


def get_family(conn: sqlite3.Connection, family_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM families WHERE id = ?", (family_id,)).fetchone()


def latest_family(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM families ORDER BY updated_at DESC, id DESC LIMIT 1").fetchone()


def get_challenges(conn: sqlite3.Connection, family_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM challenges WHERE family_id = ? ORDER BY id", (family_id,)
    ).fetchall()


# ---- events ---------------------------------------------------------------------


def log_event(
    conn: sqlite3.Connection,
    family_id: int | None,
    event_type: str,
    *,
    synthetic_likelihood: float | None = None,
    band: str | None = None,
    verification_result: str | None = None,
    notes: str | None = None,
) -> sqlite3.Row:
    cur = conn.execute(
        """INSERT INTO events (family_id, timestamp, event_type, synthetic_likelihood, band,
                               verification_result, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (family_id, now_iso(), event_type, synthetic_likelihood, band, verification_result, notes),
    )
    return conn.execute("SELECT * FROM events WHERE id = ?", (cur.lastrowid,)).fetchone()


def list_events(conn: sqlite3.Connection, family_id: int, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM events WHERE family_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
        (family_id, limit),
    ).fetchall()


# ---- verification rate limiting ----------------------------------------------


def record_attempt(conn: sqlite3.Connection, family_id: int, passed: bool, ts: float) -> None:
    conn.execute(
        "INSERT INTO verification_attempts (family_id, timestamp, passed) VALUES (?, ?, ?)",
        (family_id, ts, int(passed)),
    )


def recent_failures(conn: sqlite3.Connection, family_id: int, since: float) -> list[float]:
    """Timestamps of failed attempts after `since` and after the most recent success."""
    last_success = conn.execute(
        "SELECT MAX(timestamp) FROM verification_attempts WHERE family_id = ? AND passed = 1",
        (family_id,),
    ).fetchone()[0]
    cutoff = max(since, last_success or 0.0)
    rows = conn.execute(
        """SELECT timestamp FROM verification_attempts
           WHERE family_id = ? AND passed = 0 AND timestamp > ? ORDER BY timestamp""",
        (family_id, cutoff),
    ).fetchall()
    return [r[0] for r in rows]
