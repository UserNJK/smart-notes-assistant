"""SQLite persistence for notes.

Starter notes keep their original IDs (N001-N012). User notes get IDs U0001, U0002, ...
(a different prefix), so they can never collide with starter IDs. Notes are never deleted,
so a U-number is never reused.
"""
import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "notes.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    id         TEXT UNIQUE NOT NULL,
    text       TEXT NOT NULL,
    category   TEXT NOT NULL,
    confidence REAL NOT NULL,
    uncertain  INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


class NoteStore:
    def __init__(self, path=None):
        self.path = str(path or os.environ.get("NOTES_DB", DEFAULT_DB))
        with self._connect() as con:
            con.execute(SCHEMA)

    def _connect(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def add(self, text: str, prediction: dict) -> dict:
        with self._connect() as con:
            # Next id = highest existing U-number + 1, computed inside the INSERT so it is atomic.
            # (Not the AUTOINCREMENT seq: SQLite also advances that on ignored duplicate starter inserts.)
            cur = con.execute(
                """INSERT INTO notes (id, text, category, confidence, uncertain) VALUES (
                       (SELECT printf('U%04d', COALESCE(MAX(CAST(SUBSTR(id, 2) AS INTEGER)), 0) + 1)
                        FROM notes WHERE id LIKE 'U%'), ?, ?, ?, ?)""",
                (text, prediction["category"], prediction["confidence"], int(prediction["uncertain"])),
            )
            note_id = con.execute("SELECT id FROM notes WHERE seq = ?", (cur.lastrowid,)).fetchone()["id"]
        return self.get(note_id)

    def add_with_id(self, note_id: str, text: str, prediction: dict) -> bool:
        """Insert a note with a fixed id. Returns False if the id already exists (no duplicate)."""
        with self._connect() as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO notes (id, text, category, confidence, uncertain) VALUES (?, ?, ?, ?, ?)",
                (note_id, text, prediction["category"], prediction["confidence"], int(prediction["uncertain"])),
            )
            return cur.rowcount == 1

    def get(self, note_id: str):
        with self._connect() as con:
            row = con.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return _to_dict(row) if row else None

    def list(self, category=None, uncertain=None) -> list:
        query, args = "SELECT * FROM notes WHERE 1=1", []
        if category:
            query += " AND category = ?"
            args.append(category)
        if uncertain is not None:
            query += " AND uncertain = ?"
            args.append(int(uncertain))
        with self._connect() as con:
            rows = con.execute(query + " ORDER BY seq", args).fetchall()
        return [_to_dict(r) for r in rows]


def _to_dict(row) -> dict:
    return {
        "id": row["id"],
        "text": row["text"],
        "category": row["category"],
        "confidence": row["confidence"],
        "uncertain": bool(row["uncertain"]),
        "created_at": row["created_at"],
    }
