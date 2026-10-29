"""Load data/starter_notes.json (N001-N012) into the notes database. Safe to run twice.

    python load_starter.py
"""
from app.classifier import NoteClassifier
from app.main import load_starter_into
from app.store import NoteStore

if __name__ == "__main__":
    store = NoteStore()
    print(load_starter_into(store, NoteClassifier()), "->", store.path)
