"""Small helpers for reading the supplied datasets."""
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv(name: str):
    """Return (texts, labels). labels is None for files without a category column."""
    with open(DATA_DIR / name, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text"] for r in rows]
    labels = [r["category"] for r in rows] if rows and "category" in rows[0] else None
    return texts, labels


def load_starter_notes():
    with open(DATA_DIR / "starter_notes.json", encoding="utf-8") as f:
        return json.load(f)
